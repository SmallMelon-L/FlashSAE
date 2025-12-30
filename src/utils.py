import torch
from types import TracebackType
from tqdm import tqdm

from .config import SAEConfig, TestConfig


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


def sort_topk_result(topk_indices: torch.Tensor, topk_values: torch.Tensor):
    topk_indices_sorted, sort_perm = torch.sort(topk_indices, dim=-1)
    topk_values_sorted = torch.gather(topk_values, dim=-1, index=sort_perm)
    return topk_indices_sorted, topk_values_sorted


def build_sparse_csr_from_topk(
    topk_indices_sorted: torch.Tensor, topk_values_sorted: torch.Tensor, n_cols: int
):
    n_rows: int = topk_indices_sorted.shape[0]
    nnz_per_row: int = topk_indices_sorted.shape[1]

    crow_indices: torch.Tensor = torch.arange(
        0,
        (n_rows + 1) * nnz_per_row,
        nnz_per_row,
        device=topk_indices_sorted.device,
        dtype=topk_indices_sorted.dtype,
    )
    col_indices: torch.Tensor = topk_indices_sorted.reshape(-1)
    values: torch.Tensor = topk_values_sorted.reshape(-1)

    sparse_csr_tensor: torch.Tensor = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, size=(n_rows, n_cols)
    )

    return sparse_csr_tensor


def build_sparse_coo_from_topk(
    topk_indices_sorted: torch.Tensor, topk_values_sorted: torch.Tensor, n_cols: int
):
    n_rows: int = topk_indices_sorted.shape[0]
    nnz_per_row: int = topk_indices_sorted.shape[1]

    row_indices: torch.Tensor = torch.arange(
        0, n_rows, device=topk_indices.device, dtype=topk_indices.dtype
    ).repeat_interleave(nnz_per_row)
    col_indices: torch.Tensor = topk_indices_sorted.reshape(-1)
    values: torch.Tensor = topk_values_sorted.reshape(-1)
    sparse_coo_tensor: torch.Tensor = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices], dim=0),
        values,
        size=(n_rows, n_cols),
    )
    return sparse_coo_tensor


def generate_data(
    config: SAEConfig,
) -> dict[str, torch.Tensor]:
    activations: torch.Tensor = torch.randn(
        config.batch_size, config.d_model, device=config.device, dtype=config.dtype
    )
    W_E: torch.Tensor = torch.randn(
        config.d_model, config.d_sae, device=config.device, dtype=config.dtype
    )
    W_E /= W_E.norm(dim=0, keepdim=True)
    b_E: torch.Tensor = torch.randn(
        config.d_sae, device=config.device, dtype=config.dtype
    )
    b_E /= b_E.norm(dim=-1, keepdim=True)
    W_D: torch.Tensor = torch.randn(
        config.d_sae, config.d_model, device=config.device, dtype=config.dtype
    )
    W_D /= W_D.norm(dim=1, keepdim=True)
    b_D: torch.Tensor = torch.randn(
        config.d_model, device=config.device, dtype=config.dtype
    )
    b_D /= b_D.norm(dim=-1, keepdim=True)

    return {
        "activations": activations,
        "W_E": W_E,
        "b_E": b_E,
        "W_D": W_D,
        "b_D": b_D,
        "k": config.k,
    }


def sae_fwd_bwd(fwd_func, activations, W_E, b_E, W_D, b_D, k, times={}):
    activations = activations.clone().detach().requires_grad_(False)
    W_E = W_E.clone().detach().requires_grad_(True)
    W_D = W_D.clone().detach().requires_grad_(True)
    b_E = b_E.clone().detach().requires_grad_(True)
    b_D = b_D.clone().detach().requires_grad_(True)

    reconstruct, times = fwd_func(
        activations, W_E, b_E, W_D, b_D, k, times, enable_timer=True
    )

    loss = (reconstruct - activations).pow(2).sum(dim=-1).mean()
    loss.backward()

    result_dict = {
        "reconstruct": reconstruct,
        "W_E.grad": W_E.grad,
        "W_D.grad": W_D.grad,
        "b_E.grad": b_E.grad,
        "b_D.grad": b_D.grad,
    }

    return result_dict, times


def sae_fwd_bwd_time(fwd_func, activations, W_E, b_E, W_D, b_D, k, times={}):
    activations = activations.clone().detach().requires_grad_(False)
    W_E = W_E.clone().detach().requires_grad_(True)
    W_D = W_D.clone().detach().requires_grad_(True)
    b_E = b_E.clone().detach().requires_grad_(True)
    b_D = b_D.clone().detach().requires_grad_(True)

    with CUDATimer(times, "forward"):
        reconstruct, times = fwd_func(
            activations, W_E, b_E, W_D, b_D, k, times, enable_timer=False
        )
    with CUDATimer(times, "compute_loss"):
        loss = (reconstruct - activations).pow(2).sum(dim=-1).mean()
    with CUDATimer(times, "backward"):
        loss.backward()

    result_dict = {
        "reconstruct": reconstruct,
        "W_E.grad": W_E.grad,
        "W_D.grad": W_D.grad,
        "b_E.grad": b_E.grad,
        "b_D.grad": b_D.grad,
    }
    return result_dict, times


def compare_reconstruct(
    sae_fwd_1,
    sae_fwd_2,
    config: SAEConfig,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    verbose: bool = True,
):
    data_dict = generate_data(config)
    reconstruct_1, _ = sae_fwd_1(**data_dict)
    reconstruct_2, _ = sae_fwd_2(**data_dict)
    if verbose:
        print(
            f"reconstruct is Same: {torch.allclose(reconstruct_1.to(config.dtype), reconstruct_2.to(config.dtype), atol=atol, rtol=rtol)}"
        )
        diff = (reconstruct_1.to(config.dtype) - reconstruct_2.to(config.dtype)).abs()
        print(f"max={diff.max():.2e}, mean={diff.mean():.2e}")
    else:
        assert torch.allclose(
            reconstruct_1.to(config.dtype),
            reconstruct_2.to(config.dtype),
            atol=atol,
            rtol=rtol,
        )


def compare_reconstruct_grad(
    sae_fwd_1,
    sae_fwd_2,
    config: SAEConfig,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    verbose: bool = True,
):
    data_dict = generate_data(config)

    result_dict_1, _ = sae_fwd_bwd(
        fwd_func=sae_fwd_1,
        **data_dict,
    )

    result_dict_2, _ = sae_fwd_bwd(
        fwd_func=sae_fwd_2,
        **data_dict,
    )

    if verbose:
        for name in result_dict_1.keys():
            print(
                f"{name:11s} is Same: {torch.allclose(result_dict_1[name].to(config.dtype), result_dict_2[name].to(config.dtype), atol=atol, rtol=rtol)}"
            )
            diff = (
                result_dict_1[name].to(config.dtype)
                - result_dict_2[name].to(config.dtype)
            ).abs()
            print(f"max={diff.max():.2e}, mean={diff.mean():.2e}")
    else:
        for name in result_dict_1.keys():
            assert torch.allclose(
                result_dict_1[name].to(config.dtype),
                result_dict_2[name].to(config.dtype),
                atol=atol,
                rtol=rtol,
            )


def compare_latency(
    sae_fwd_1,
    sae_fwd_2,
    config: TestConfig,
    verbose: bool = True,
):
    data_dict = generate_data(config)

    if verbose:
        iters = tqdm(range(config.preheat_repeat))
    else:
        iters = range(config.preheat_repeat)

    for _ in iters:
        sae_fwd_bwd(
            fwd_func=sae_fwd_1,
            **data_dict,
        )
        sae_fwd_bwd(
            fwd_func=sae_fwd_2,
            **data_dict,
        )

    fwd_bwd_times_1 = {}
    fwd_bwd_times_2 = {}
    if verbose:
        iters = tqdm(range(config.timing_repeat))
    else:
        iters = range(config.timing_repeat)
    for _ in iters:
        result_dict_1, fwd_bwd_times_1 = sae_fwd_bwd_time(
            fwd_func=sae_fwd_1,
            **data_dict,
            times=fwd_bwd_times_1,
        )

        result_dict_2, fwd_bwd_times_2 = sae_fwd_bwd_time(
            fwd_func=sae_fwd_2,
            **data_dict,
            times=fwd_bwd_times_2,
        )

    fwd_bwd_times_1 = {
        name: fwd_bwd_times_1[name] / config.timing_repeat
        for name in fwd_bwd_times_1.keys()
    }
    fwd_bwd_times_2 = {
        name: fwd_bwd_times_2[name] / config.timing_repeat
        for name in fwd_bwd_times_2.keys()
    }

    if verbose:
        print("=" * 100)
        print(f"{'name':32s} {'sae_fwd_1':>17s} {'sae_fwd_2':>17s} {'speedup':>17s}")
        print("-" * 100)
        all_time_1 = 0
        all_time_2 = 0
        for name in fwd_bwd_times_1.keys():
            if name not in fwd_bwd_times_2.keys():
                continue
            t1 = fwd_bwd_times_1[name]
            t2 = fwd_bwd_times_2[name]
            speedup = t1 / t2 if t2 > 0 else float("inf")
            print(f"{name:32s} {t1:>17.3f}ms {t2:>17.3f}ms {speedup:>17.2f}x")
            all_time_1 += t1
            all_time_2 += t2
        print("-" * 100)
        total_speedup = all_time_1 / all_time_2 if all_time_2 > 0 else float("inf")
        print(
            f"{'total':32s} {all_time_1:>17.3f}ms {all_time_2:>17.3f}ms {total_speedup:>17.2f}x"
        )
        print("=" * 100)

    times_1 = {}
    times_2 = {}

    if verbose:
        iters = tqdm(range(config.timing_repeat))
    else:
        iters = range(config.timing_repeat)

    for _ in iters:
        result_dict_1, times_1 = sae_fwd_bwd(
            fwd_func=sae_fwd_1,
            **data_dict,
            times=times_1,
        )

        result_dict_2, times_2 = sae_fwd_bwd(
            fwd_func=sae_fwd_2,
            **data_dict,
            times=times_2,
        )

    times_1 = {name: times_1[name] / config.timing_repeat for name in times_1.keys()}
    times_2 = {name: times_2[name] / config.timing_repeat for name in times_2.keys()}

    if verbose:
        print("=" * 100)
        print(f"{'name':32s} {'sae_fwd_1':>17s}")
        print("-" * 100)
        for name in times_1.keys():
            t1 = times_1[name]
            print(f"{name:32s} {t1:>17.3f}ms")
        print("-" * 100)
        print(f"{'name':32s} {'sae_fwd_2':>17s}")
        print("-" * 100)
        for name in times_2.keys():
            t2 = times_2[name]
            print(f"{name:32s} {t2:>17.3f}ms")
        print("-" * 100)

    return fwd_bwd_times_1, fwd_bwd_times_2, times_1, times_2
