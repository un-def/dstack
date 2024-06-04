from dstack._internal.core.backends.base.config import BackendConfig
from dstack._internal.core.models.backends.gcp import AnyGCPCreds, GCPStoredConfig


class GCPConfig(GCPStoredConfig, BackendConfig):
    creds: AnyGCPCreds

    @property
    def vpc_id(self) -> str:
        vpc_name = self.vpc_name
        if vpc_name is None:
            vpc_name = "default"
        return f"projects/{self.project_id}/global/networks/{vpc_name}"
