from pydantic import BaseModel, ConfigDict
import torch


class SAEConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    batch_size: int = 4096
    d_model: int = 2048
    expansion_factor: int = 8
    k: int = 128

    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion_factor


class TestConfig(SAEConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    preheat_repeat: int = 3
    timing_repeat: int = 20
