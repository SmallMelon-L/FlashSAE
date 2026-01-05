from pydantic import BaseModel, ConfigDict
import torch


class SAEConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    batch_size: int = 4096
    d_model: int = 2048
    expansion_factor: int = 8
    topk: int = 128

    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion_factor


class LorsaConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    batch_size: int = 16
    seq_len: int = 2048
    d_model: int = 4096
    n_qk_heads: int = 128
    d_qk_head: int = 128
    expansion_factor: int = 8
    topk: int = 128

    causal: bool = False
    softmax_scale: float | None = None

    @property
    def n_ov_heads(self) -> int:
        return self.d_model * self.expansion_factor

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert self.n_ov_heads % self.n_qk_heads == 0, (
            "n_ov_heads must be divisible by n_qk_heads"
        )

        if self.softmax_scale is None:
            self.softmax_scale = 1 / self.d_qk_head**0.5


class TestConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    preheat_repeat: int = 3
    timing_repeat: int = 10


class LorsaTestConfig(LorsaConfig, TestConfig):
    pass


class SAETestConfig(SAEConfig, TestConfig):
    pass
