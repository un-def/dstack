import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from dstack._internal.compat import IS_WINDOWS
from dstack._internal.core.errors import SSHError
from dstack._internal.utils.path import PathLike


@dataclass
class SSHClientInfo:
    # Path to `ssh` executable
    path: Path
    # Full version including portable suffix, e.g., "9.6p1"
    version: str
    # Base version not including portable suffix, e.g., (9, 6)
    version_tuple: Tuple[int, ...]
    # True if OpenSSH_for_Windows (Microsoft's OpenSSH Portable fork)
    for_windows: bool
    # Supports Control{Master,Path,Persist} directives, but only for control purposes
    # (e.g., `ssh -O exit`), cannot be used for connection multiplexing
    supports_control_socket: bool
    # Supports Control{Master,Path,Persist} for connection multiplexing
    supports_multiplexing: bool
    # Supports ForkAfterAuthentication (`ssh -f`)
    supports_background_mode: bool

    RAW_VERSION_REGEX = re.compile(
        r"OpenSSH_(?P<for_windows>for_Windows_)?(?P<base_version>[\d.]+)(?P<portable_version>p\d+)?",
        flags=re.I,
    )

    @classmethod
    def from_raw_version(cls, raw_version: str, path: Path) -> "SSHClientInfo":
        match = cls.RAW_VERSION_REGEX.match(raw_version)
        if not match:
            raise ValueError("no match")
        for_windows, base_version, portable_version = match.group(
            "for_windows", "base_version", "portable_version"
        )
        if portable_version:
            version = f"{base_version}{portable_version}"
        else:
            version = base_version
        return cls(
            path=path,
            version=version,
            version_tuple=tuple(map(int, base_version.split("."))),
            for_windows=bool(for_windows),
            supports_control_socket=(not for_windows),
            supports_multiplexing=(not IS_WINDOWS),
            supports_background_mode=(not for_windows),
        )


def inspect_ssh_client(path: PathLike) -> SSHClientInfo:
    """
    Inspects various aspects of a given SSH client — version, "flavor", features — by executing
    and parsing `ssh -V`.

    :param path: a path of the ssh executable.
    :return: :class:`SSHClientInfo` named tuple.
    :raise dstack._internal.core.errors.SSHError: if path does not exist, `ssh -V` returns
        non-zero exit status, or `ssh -V` output does not match the pattern.
    """
    path = Path(path).resolve()
    try:
        cp = subprocess.run(
            [path, "-V"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise SSHError(f"failed to execute `{path} -V`: {e}") from e
    output = cp.stderr
    if cp.returncode != 0:
        raise SSHError(f"`{path} -V` returned non-zero exit status {cp.returncode}: {output}")
    try:
        return SSHClientInfo.from_raw_version(output, path)
    except ValueError:
        raise SSHError(f"failed to parse `{path} -V` output: {output}")


def find_ssh_client() -> Optional[Path]:
    path_str = os.getenv("DSTACK_SSH_CLIENT")
    if path_str:
        return Path(path_str)
    if not IS_WINDOWS:
        path_str = shutil.which("ssh")
        if path_str:
            return Path(path_str)
        return None
    # First, we check for ssh bundled with Git for Windows (MSYS2/MinGW-w64-built OpenSSH Portable)
    # as a preferred client. It supports ForkAfterAuthentication; ControlMaster is only partially
    # supported, we don't use it.
    git_path_str = shutil.which("git")
    if git_path_str:
        # C:\Program Files\Git\cmd\git.exe -> C:\Program Files\Git\usr\bin\ssh.exe
        path = Path(git_path_str).parent.parent / "usr" / "bin" / "ssh.exe"
        if path.exists():
            return path
    # Then we check for OpenSSH for Windows (Microsoft's fork of OpenSSH Portable).
    # It does not support some features, namely ControlMaster and ForkAfterAuthentication.
    windir_str = os.getenv("WINDIR")
    if windir_str:
        path = Path(windir_str) / "System32" / "OpenSSH" / "ssh.exe"
        if path.exists():
            return path
    # Finally, we check for any ssh client in PATH. It can be anything, it can be not compatible,
    # so we use it only as a last resort.
    path_str = shutil.which("ssh")
    if path_str:
        return Path(path_str)
    return None


_ssh_client_info: Optional[SSHClientInfo] = None


def get_ssh_client_info() -> SSHClientInfo:
    """
    Returns :class:`SSHClientInfo` for the default SSH client. The result is cached.

    :return: :class:`SSHClientInfo` named tuple.
    :raise dstack._internal.core.errors.SSHError: if no ssh client found or the underlying
    :func:`inspect_ssh_client` raises an error.
    """
    global _ssh_client_info
    if _ssh_client_info is not None:
        return _ssh_client_info
    path = find_ssh_client()
    if path is None:
        if IS_WINDOWS:
            msg = "SSH client not found, install Git for Windows."
        else:
            msg = "SSH client not found."
        raise SSHError(msg)
    _ssh_client_info = inspect_ssh_client(path)
    return _ssh_client_info
