import torch
from types import TracebackType


class CUDATimer:
    def __init__(self, times_dict: dict[str, float], name: str, enable: bool = True):
        self.enable: bool = enable
        if not self.enable:
            return
        self.times_dict: dict[str, float] = times_dict
        self.name: str = name
        self.starter: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
        self.ender: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if not self.enable:
            return self
        torch.cuda.synchronize()
        self.starter.record()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if not self.enable:
            return False
        self.ender.record()
        torch.cuda.synchronize()
        elapsed: float = self.starter.elapsed_time(self.ender)  # pyright: ignore[reportUnknownMemberType]
        if self.name not in self.times_dict:
            self.times_dict[self.name] = elapsed
        else:
            self.times_dict[self.name] += elapsed
        return False
