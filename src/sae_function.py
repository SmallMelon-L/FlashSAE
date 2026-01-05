from typing import Any, Literal
from functools import partial

import torch
import torch.sparse
import torch.autograd.function

from .utils.time import CUDATimer
from .utils.ops import (
    sort_topk_result,
    build_sparse_csr_from_topk,
    build_sparse_coo_from_topk,
)
from .kernel import (
    masked_matmul,
    masked_matmul_with_col_sum,
)


def topk_sae(
    activations: torch.Tensor,  # [batch_size, d_model]
    W_E: torch.Tensor,  # [d_model, d_sae]
    b_E: torch.Tensor,  # [d_sae]
    W_D: torch.Tensor,  # [d_sae, d_model]
    b_D: torch.Tensor,  # [d_model]
    topk: int,
    times: dict[str, float] = {},
    enable_timer: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    with CUDATimer(times, "fwd_encode", enable=enable_timer):
        hidden_pre = activations @ W_E + b_E  # [batch_size, d_sae]

    with CUDATimer(times, "fwd_topk", enable=enable_timer):
        topk_values, topk_indices = torch.topk(hidden_pre, k=topk, dim=-1, sorted=False)
        feature_acts = torch.zeros_like(hidden_pre)
        feature_acts.scatter_(dim=1, index=topk_indices, src=topk_values)

    with CUDATimer(times, "fwd_decode", enable=enable_timer):
        reconstruct = feature_acts @ W_D + b_D

    return reconstruct, times


def topk_sae_fwd_sparse(
    activations: torch.Tensor,
    W_E: torch.Tensor,
    b_E: torch.Tensor,
    W_D: torch.Tensor,
    b_D: torch.Tensor,
    topk: int,
    times: dict[str, float] = {},
    *,
    sparse_format: Literal["csr", "coo"] = "csr",
    enable_timer: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    with CUDATimer(times, "fwd_encode", enable=enable_timer):
        hidden_pre = activations @ W_E + b_E

    with CUDATimer(times, "fwd_topk", enable=enable_timer):
        topk_values, topk_indices = torch.topk(hidden_pre, k=topk, dim=-1, sorted=False)

    with CUDATimer(times, "fwd_build_feature_acts", enable=enable_timer):
        feature_acts = torch.zeros_like(hidden_pre)
        feature_acts.scatter_(dim=1, index=topk_indices, src=topk_values)

    with CUDATimer(times, "fwd_feature_acts_to_sparse", enable=enable_timer):
        if sparse_format == "csr":
            feature_acts_sparse = feature_acts.to_sparse_csr()
        elif sparse_format == "coo":
            feature_acts_sparse = feature_acts.to_sparse_coo()

    with CUDATimer(times, "fwd_sparse_decode", enable=enable_timer):
        reconstruct = torch.sparse.mm(feature_acts_sparse, W_D) + b_D

    return reconstruct, times


topk_sae_fwd_sparse_coo = partial(topk_sae_fwd_sparse, sparse_format="coo")
topk_sae_fwd_sparse_csr = partial(topk_sae_fwd_sparse, sparse_format="csr")


def topk_sae_fwd_sparse_fused(
    activations: torch.Tensor,
    W_E: torch.Tensor,
    b_E: torch.Tensor,
    W_D: torch.Tensor,
    b_D: torch.Tensor,
    topk: int,
    times: dict[str, float] = {},
    *,
    sparse_format: Literal["csr", "coo"] = "csr",
    enable_timer: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    d_sae: int = W_E.shape[1]

    with CUDATimer(times, "fwd_encode", enable=enable_timer):
        hidden_pre = activations @ W_E + b_E

    with CUDATimer(times, "fwd_topk", enable=enable_timer):
        topk_values, topk_indices = torch.topk(hidden_pre, k=topk, dim=-1, sorted=False)

    with CUDATimer(times, "fwd_sort_topk_result", enable=enable_timer):
        topk_indices_sorted, topk_values_sorted = sort_topk_result(
            topk_indices, topk_values
        )

    with CUDATimer(times, "fwd_build_sparse_feature_acts", enable=enable_timer):
        if sparse_format == "csr":
            feature_acts_sparse = build_sparse_csr_from_topk(
                topk_indices_sorted, topk_values_sorted, d_sae
            )
        elif sparse_format == "coo":
            feature_acts_sparse = build_sparse_coo_from_topk(
                topk_indices_sorted, topk_values_sorted, d_sae
            )

    with CUDATimer(times, "fwd_sparse_decode", enable=enable_timer):
        reconstruct = torch.sparse.mm(feature_acts_sparse, W_D) + b_D

    return reconstruct, times


topk_sae_fwd_sparse_coo_fused = partial(topk_sae_fwd_sparse_fused, sparse_format="coo")
topk_sae_fwd_sparse_csr_fused = partial(topk_sae_fwd_sparse_fused, sparse_format="csr")


class TopKSparseFusedSAE(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        activations: torch.Tensor,
        W_E: torch.Tensor,
        b_E: torch.Tensor,
        W_D: torch.Tensor,
        b_D: torch.Tensor,
        topk: int,
        times: dict[str, float] = {},
        enable_timer: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        d_sae = W_E.shape[1]

        with CUDATimer(times, "fwd_encode", enable=enable_timer):
            hidden_pre = activations @ W_E + b_E

        with CUDATimer(times, "fwd_topk", enable=enable_timer):
            topk_values, topk_indices = torch.topk(
                hidden_pre, k=topk, dim=-1, sorted=False
            )

        with CUDATimer(times, "fwd_sort_topk_result", enable=enable_timer):
            topk_indices_sorted, topk_values_sorted = sort_topk_result(
                topk_indices, topk_values
            )

        with CUDATimer(times, "fwd_build_sparse_feature_acts", enable=enable_timer):
            feature_acts_sparse = build_sparse_csr_from_topk(
                topk_indices_sorted, topk_values_sorted, d_sae
            )

        with CUDATimer(times, "fwd_sparse_decode", enable=enable_timer):
            reconstruct = torch.sparse.mm(feature_acts_sparse, W_D) + b_D

        ctx.save_for_backward(
            activations,
            W_E,
            b_E,
            W_D,
            b_D,
            feature_acts_sparse,
            topk_indices_sorted,
            topk_values_sorted,
        )
        ctx.topk = topk
        ctx.times = times
        ctx.enable_timer = enable_timer

        return reconstruct, times

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx,
        grad_output: torch.Tensor,
        _: Any,
    ) -> tuple[
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
    ]:
        (
            activations,
            W_E,
            b_E,
            W_D,
            b_D,
            feature_acts_sparse,
            topk_indices_sorted,
            topk_values_sorted,
        ) = ctx.saved_tensors
        topk = ctx.topk
        times = ctx.times
        enable_timer = ctx.enable_timer

        batch_size, d_model = activations.shape
        d_sae = W_E.shape[1]

        grad_W_E = grad_b_E = grad_W_D = grad_b_D = None

        # grad_b_D
        with CUDATimer(times, "bwd_grad_b_D", enable=enable_timer):
            grad_b_D = grad_output.sum(dim=0)

        # grad_W_D
        with CUDATimer(times, "bwd_grad_W_D_transpose", enable=enable_timer):
            feature_acts_sparse_T = feature_acts_sparse.to_sparse_coo().T
            # feature_acts_sparse_T = feature_acts_sparse.transpose(0, 1)
        with CUDATimer(times, "bwd_grad_W_D_to_sparse", enable=enable_timer):
            feature_acts_sparse_T = feature_acts_sparse_T.to_sparse_csr()
        with CUDATimer(times, "bwd_grad_W_D_mul", enable=enable_timer):
            grad_W_D = feature_acts_sparse_T @ grad_output

        with CUDATimer(times, "bwd_masked_matmul", enable=enable_timer):
            grad_topk_values, grad_b_E = masked_matmul_with_col_sum(
                grad_output, W_D, topk_indices_sorted
            )

        with CUDATimer(times, "bwd_grad_hidden_pre", enable=enable_timer):
            crow_indices = torch.arange(
                0,
                (batch_size + 1) * topk,
                topk,
                device=grad_topk_values.device,
                dtype=torch.int32,
            )
            col_indices = topk_indices_sorted.reshape(-1).to(torch.int32)
            grad_hidden_pre = torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                grad_topk_values.reshape(-1),
                size=(batch_size, d_sae),
            )

        # grad_W_E
        with CUDATimer(times, "bwd_grad_W_E_transpose", enable=enable_timer):
            grad_hidden_pre_T = grad_hidden_pre.to_sparse_coo().T
        with CUDATimer(times, "bwd_grad_W_E_to_sparse", enable=enable_timer):
            grad_hidden_pre_T = grad_hidden_pre_T.to_sparse_csr()
        with CUDATimer(times, "bwd_grad_W_E_mul", enable=enable_timer):
            grad_W_E = torch.sparse.mm(grad_hidden_pre_T, activations).T

        return None, grad_W_E, grad_b_E, grad_W_D, grad_b_D, None, None, None


def topk_sae_sparse_fused(
    activations: torch.Tensor,
    W_E: torch.Tensor,
    b_E: torch.Tensor,
    W_D: torch.Tensor,
    b_D: torch.Tensor,
    topk: int,
    times: dict[str, float] = {},
    enable_timer: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    return TopKSparseFusedSAE.apply(
        activations, W_E, b_E, W_D, b_D, topk, times, enable_timer
    )
