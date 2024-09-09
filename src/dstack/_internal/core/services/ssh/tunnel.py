import abc
import asyncio
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Union, BinaryIO

from dstack._internal.core.errors import SSHError
from dstack._internal.core.models.instances import SSHConnectionParams
from dstack._internal.core.services.ssh import get_ssh_error
from dstack._internal.core.services.ssh.client import get_ssh_client_info
from dstack._internal.utils.logging import get_logger
from dstack._internal.utils.path import FilePath, FilePathOrContent, PathLike
from dstack._internal.utils.ssh import normalize_path

logger = get_logger(__name__)
SSH_TIMEOUT = 15
SSH_DEFAULT_OPTIONS = {
    "StrictHostKeyChecking": "no",
    "UserKnownHostsFile": "/dev/null",
    "ExitOnForwardFailure": "yes",
    "StreamLocalBindUnlink": "yes",
    "ConnectTimeout": "3",
}


class Socket(abc.ABC):
    @abc.abstractmethod
    def render(self) -> str:
        pass


@dataclass
class UnixSocket(Socket):
    path: PathLike

    def render(self) -> str:
        return str(self.path)


@dataclass
class IPSocket(Socket):
    host: str
    port: int

    def render(self) -> str:
        if ":" in self.host:  # assuming IPv6
            return f"[{self.host}]:{self.port}"
        return f"{self.host}:{self.port}"


@dataclass
class SocketPair:
    local: Socket
    remote: Socket


class SSHTunnel:
    def __init__(
        self,
        destination: str,
        identity: FilePathOrContent,
        forwarded_sockets: Iterable[SocketPair] = (),
        reverse_forwarded_sockets: Iterable[SocketPair] = (),
        control_sock_path: Optional[PathLike] = None,
        options: Dict[str, str] = SSH_DEFAULT_OPTIONS,
        ssh_config_path: Union[PathLike, Literal["none"]] = "none",
        port: Optional[int] = None,
        ssh_proxy: Optional[SSHConnectionParams] = None,
    ):
        """
        :param forwarded_sockets: Connections to the specified local sockets will be
            forwarded to their corresponding remote sockets
        :param reverse_forwarded_sockets: Connections to the specified remote sockets
            will be forwarded to their corresponding local sockets
        """
        self.destination = destination
        self.forwarded_sockets = list(forwarded_sockets)
        self.reverse_forwarded_sockets = list(reverse_forwarded_sockets)
        self.options = options
        self.port = port
        self.ssh_config_path = normalize_path(ssh_config_path)
        self.ssh_proxy = ssh_proxy
        temp_dir = tempfile.TemporaryDirectory(delete=False)
        self.temp_dir = temp_dir
        if control_sock_path is None:
            control_sock_path = os.path.join(temp_dir.name, "control.sock")
        self.control_sock_path = normalize_path(control_sock_path)
        if isinstance(identity, FilePath):
            identity_path = identity.path
        else:
            identity_path = os.path.join(temp_dir.name, "identity")
            with open(
                identity_path, opener=lambda path, flags: os.open(path, flags, 0o600), mode="w"
            ) as f:
                f.write(identity.content)
        self.identity_path = normalize_path(identity_path)
        self.log_path = normalize_path(os.path.join(temp_dir.name, "tunnel.log"))
        self.ssh_client_info = get_ssh_client_info()
        self.ssh_exec_path = str(self.ssh_client_info.path)
        self.proc: Optional[subprocess.Popen] = None

    @property
    def _use_background_mode(self) -> bool:
        return (
            self.ssh_client_info.supports_background_mode
            and self.ssh_client_info.supports_control_socket
        )

    def open_command(self) -> List[str]:
        # Some information about how `ssh(1)` handles options:
        # 1. Command-line options override config options regardless of the order of the arguments:
        #   `ssh -S sock2 -F config` with `ControlPath sock1` in the config -> the control socket
        #   path is `sock2`.
        # 2. First argument wins:
        #   `ssh -S sock2 -S sock1` -> the control socket path is `sock2`.
        # 3. `~` is not expanded in the arguments, but expanded in the config file.
        command = [
            self.ssh_exec_path,
            "-F",
            self.ssh_config_path,
            "-i",
            self.identity_path,
            "-N",  # do not run commands on remote
        ]
        if self._use_background_mode:
            # It's safe to use ControlMaster even if the ssh client does not support multiplexing
            # as long as we don't allow more than one tunnel to the specific host to be running.
            # We use this feature for control only (see :meth:`close_command`).
            # NB: As a precaution, we use `-o ControlMaster=auto` instead of `-M` (which means
            # `-o ControlMaster=yes`) to avoid spawning uncontrollable ssh instances if more than
            # one tunnel is started.
            command += [
                "-f",  # go to background after successful authentication
                "-o",
                "ControlMaster=auto",
                "-S",
                self.control_sock_path,
            ]
        else:
            # Debug logs are used to determine when the ssh client is successfully authenticated.
            command += ["-v"]
        if self.port is not None:
            command += ["-p", str(self.port)]
        for k, v in self.options.items():
            command += ["-o", f"{k}={v}"]
        if proxy_command := self.proxy_command():
            command += ["-o", "ProxyCommand=" + shlex.join(proxy_command)]
        for socket_pair in self.forwarded_sockets:
            command += ["-L", f"{socket_pair.local.render()}:{socket_pair.remote.render()}"]
        for socket_pair in self.reverse_forwarded_sockets:
            command += ["-R", f"{socket_pair.remote.render()}:{socket_pair.local.render()}"]
        command += [self.destination]
        return command

    def close_command(self) -> List[str]:
        return [self.ssh_exec_path, "-S", self.control_sock_path, "-O", "exit", self.destination]

    def check_command(self) -> List[str]:
        return [self.ssh_exec_path, "-S", self.control_sock_path, "-O", "check", self.destination]

    def exec_command(self) -> List[str]:
        return [self.ssh_exec_path, "-S", self.control_sock_path, self.destination]

    def proxy_command(self) -> Optional[List[str]]:
        if self.ssh_proxy is None:
            return None
        return [
            self.ssh_exec_path,
            "-i",
            self.identity_path,
            "-W",
            "%h:%p",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-p",
            str(self.ssh_proxy.port),
            f"{self.ssh_proxy.username}@{self.ssh_proxy.hostname}",
        ]

    def open(self) -> None:
        # We cannot use `stderr=subprocess.PIPE` here since the forked process (daemon) does not
        # close standard streams if ProxyJump is used, therefore we will wait EOF from the pipe
        # as long as the daemon exists.
        self._remove_log_file()
        with open(self.log_path, "wb") as f:
            try:
                if self._use_background_mode:
                    ok = self._open_in_background(f)
                else:
                    ok = self._open_in_foreground(f)
            except subprocess.TimeoutExpired as e:
                msg = f"SSH tunnel to {self.destination} did not open in {SSH_TIMEOUT} seconds"
                logger.debug(msg)
                raise SSHError(msg) from e
        if ok:
            return
        try:
            stderr = self._read_log_file()
        except OSError:
            stderr = b"???"
        logger.debug("SSH tunnel failed: %s", stderr)
        raise get_ssh_error(stderr)

    def _open_in_background(self, stderr: BinaryIO) -> bool:
        # We cannot use `stderr=subprocess.PIPE` here since the forked process (daemon) does not
        # close standard streams (it calls `daemon(3)` with `noclose` set to `1`), therefore
        # we will wait EOF from the pipe as long as the daemon exists.
        r = subprocess.run(
            self.open_command(),
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            timeout=SSH_TIMEOUT,
        )
        return r.returncode == 0

    def _open_in_foreground(self, stderr: BinaryIO) -> bool:
        cmd = self.open_command()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr)
        self.proc = proc
        with open(self.log_path, "rb") as f:
            for _ in range(SSH_TIMEOUT):
                while line := f.readline():
                    if b"Entering interactive session" in line:
                        return True
                if proc.poll() is not None:
                    return False
                time.sleep(1)
        proc.terminate()
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=SSH_TIMEOUT)

    async def aopen(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._remove_log_file)
        proc = await asyncio.create_subprocess_exec(
            *self.open_command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            await asyncio.wait_for(proc.communicate(), SSH_TIMEOUT)
        except asyncio.TimeoutError as e:
            proc.kill()
            msg = f"SSH tunnel to {self.destination} did not open in {SSH_TIMEOUT} seconds"
            logger.debug(msg)
            raise SSHError(msg) from e
        if proc.returncode == 0:
            return
        stderr = await loop.run_in_executor(None, self._read_log_file)
        logger.debug("SSH tunnel failed: %s", stderr)
        raise get_ssh_error(stderr)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
                return
            except subprocess.TimeoutExpired:
                pass
            logger.warning("Killing SSH tunnel process %d", self.proc.pid)
            self.proc.kill()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                logger.warning("Failed to SSH tunnel process %d", self.proc.pid)
        else:
            subprocess.run(
                self.close_command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    async def aclose(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            *self.close_command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        await proc.wait()

    async def acheck(self) -> bool:
        proc = await asyncio.create_subprocess_exec(
            *self.check_command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        await proc.wait()
        ok = proc.returncode == 0
        return ok

    async def aexec(self, command: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            *self.exec_command(), command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise SSHError(stderr.decode())
        return stdout.decode()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self) -> None:
        self.close()
        for _ in range(3):
            try:
                self.temp_dir.cleanup()
                return
            except OSError:
                time.sleep(1)
        logger.warning("Failed to remove SSH tunnel temp dir: %s", self.temp_dir.name)

    def _read_log_file(self) -> bytes:
        with open(self.log_path, "rb") as f:
            return f.read()

    def _remove_log_file(self) -> None:
        try:
            os.remove(self.log_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.debug("Failed to remove SSH tunnel log file %s: %s", self.log_path, e)


def ports_to_forwarded_sockets(
    ports: Dict[int, int], bind_local: str = "localhost"
) -> List[SocketPair]:
    """
    Converts remote->local ports mapping to List[SocketPair] suitable for SSHTunnel
    """
    return [
        SocketPair(
            local=IPSocket(host=bind_local, port=port_local),
            remote=IPSocket(host="localhost", port=port_remote),
        )
        for port_remote, port_local in ports.items()
    ]
