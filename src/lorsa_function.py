import math
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

from .kernel import (
    masked_matmul,
    flash_attn_forward,
    flash_attn_backward,
)
from .utils.time import CUDATimer
from .utils.ops import (
    sort_topk_result,
    build_sparse_csr_from_topk,
)


def _compute_qkv(
    x: torch.Tensor,
    W_Q: torch.Tensor,
    b_Q: torch.Tensor,
    W_K: torch.Tensor,
    b_K: torch.Tensor,
    W_V: torch.Tensor,
    b_V: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # x: (batch, seq_len, d_model)
    # W_Q: (n_qk_heads, d_model, d_qk_head)
    # W_K: (n_qk_heads, d_model, d_qk_head)
    # W_V: (n_ov_heads, d_model)
    batch_size, seq_len, d_model = x.shape
    n_qk_heads, _, d_qk_head = W_Q.shape

    # q: einsum("bsd,Qdq->bsQq") equivalent
    # W_Q: (Q, d, q) -> permute to (d, Q, q) -> reshape to (d, Q*q)
    q = (x @ W_Q.permute(1, 0, 2).reshape(d_model, -1)).view(
        batch_size, seq_len, n_qk_heads, d_qk_head
    ) + b_Q

    # k: einsum("bsd,Kdk->bsKk") equivalent
    k = (x @ W_K.permute(1, 0, 2).reshape(d_model, -1)).view(
        batch_size, seq_len, n_qk_heads, d_qk_head
    ) + b_K

    # v: einsum("bsd,Vd->bsV") equivalent
    v = x @ W_V.T + b_V

    return q, k, v


def topk_lorsa(
    activations: torch.Tensor,
    W_Q: torch.Tensor,
    b_Q: torch.Tensor,
    W_K: torch.Tensor,
    b_K: torch.Tensor,
    W_V: torch.Tensor,
    b_V: torch.Tensor,
    W_O: torch.Tensor,
    b_O: torch.Tensor,
    topk: int,
    causal: bool = False,
    softmax_scale: float | None = None,
    times: dict[str, float] = {},
    enable_timer: bool = True,
) -> torch.Tensor:
    with CUDATimer(times, "fwd_compute_qkv", enable=enable_timer):
        q, k, v = _compute_qkv(activations, W_Q, b_Q, W_K, b_K, W_V, b_V)

        query = q.permute(0, 2, 1, 3)
        key = k.permute(0, 2, 1, 3)
        value = v.reshape(*k.shape[:3], -1).permute(0, 2, 1, 3)

    with CUDATimer(times, "fwd_attn_scores", enable=enable_timer):
        attn_scores = (
            torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
        )  # (batch, n_heads, seqlen_q, seqlen_k)

    with CUDATimer(times, "fwd_create_causal_mask", enable=enable_timer):
        if causal:
            mask = torch.tril(
                torch.ones(
                    query.shape[2], key.shape[2], device=query.device, dtype=torch.bool
                )
            )

    with CUDATimer(times, "fwd_mask", enable=enable_timer):
        if causal:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

    with CUDATimer(times, "fwd_softmax", enable=enable_timer):
        attn_probs = F.softmax(attn_scores, dim=-1)

    with CUDATimer(times, "fwd_compute_hidden_pre", enable=enable_timer):
        hidden_pre = (
            torch.matmul(attn_probs, value).transpose(1, 2).reshape(*v.shape)
        )  # (batch, seqlen, n_ov_heads)

    with CUDATimer(times, "fwd_topk", enable=enable_timer):
        topk_values, topk_indices = torch.topk(hidden_pre, k=topk, dim=-1, sorted=False)
        feature_acts = torch.zeros_like(hidden_pre)
        feature_acts.scatter_(dim=2, index=topk_indices, src=topk_values)

    with CUDATimer(times, "fwd_decode", enable=enable_timer):
        output = feature_acts @ W_O + b_O

    return output, times


def topk_lorsa_torch_flash(
    activations: torch.Tensor,
    W_Q: torch.Tensor,
    b_Q: torch.Tensor,
    W_K: torch.Tensor,
    b_K: torch.Tensor,
    W_V: torch.Tensor,
    b_V: torch.Tensor,
    W_O: torch.Tensor,
    b_O: torch.Tensor,
    topk: int,
    causal: bool = False,
    softmax_scale: float | None = None,
    times: dict[str, float] = {},
    enable_timer: bool = True,
) -> torch.Tensor:
    with CUDATimer(times, "fwd_compute_qkv", enable=enable_timer):
        q, k, v = _compute_qkv(activations, W_Q, b_Q, W_K, b_K, W_V, b_V)

        query = q.permute(0, 2, 1, 3)
        key = k.permute(0, 2, 1, 3)
        value = v.reshape(*k.shape[:3], -1).permute(0, 2, 1, 3)

    with CUDATimer(times, "fwd_compute_hidden_pre", enable=enable_timer):
        with sdpa_kernel(
            backends=[
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            z = F.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=softmax_scale,
                is_causal=causal,
            )
        hidden_pre = z.permute(0, 2, 1, 3).reshape(*v.shape)

    with CUDATimer(times, "fwd_topk", enable=enable_timer):
        topk_values, topk_indices = torch.topk(hidden_pre, k=topk, dim=-1, sorted=False)
        feature_acts = torch.zeros_like(hidden_pre)
        feature_acts.scatter_(dim=2, index=topk_indices, src=topk_values)

    with CUDATimer(times, "fwd_decode", enable=enable_timer):
        output = feature_acts @ W_O + b_O

    return output, times


def topk_lorsa_custom_flash(
    activations: torch.Tensor,
    W_Q: torch.Tensor,
    b_Q: torch.Tensor,
    W_K: torch.Tensor,
    b_K: torch.Tensor,
    W_V: torch.Tensor,
    b_V: torch.Tensor,
    W_O: torch.Tensor,
    b_O: torch.Tensor,
    topk: int,
    causal: bool = False,
    softmax_scale: float | None = None,
    times: dict[str, float] = {},
    enable_timer: bool = True,
) -> torch.Tensor:
    with CUDATimer(times, "fwd_compute_qkv", enable=enable_timer):
        q, k, v = _compute_qkv(activations, W_Q, b_Q, W_K, b_K, W_V, b_V)
        value = v.reshape(*k.shape[:3], -1)  # (batch, seq_len, n_qk_heads, d_v_head)

    with CUDATimer(times, "fwd_compute_hidden_pre", enable=enable_timer):
        z = flash_attn_func(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            value.to(torch.bfloat16),
            causal,
            softmax_scale,
        )
        hidden_pre = z.reshape(*v.shape).to(W_O.dtype)

    with CUDATimer(times, "fwd_topk", enable=enable_timer):
        topk_values, topk_indices = torch.topk(hidden_pre, k=topk, dim=-1, sorted=False)
        feature_acts = torch.zeros_like(hidden_pre)
        feature_acts.scatter_(dim=2, index=topk_indices, src=topk_values)

    with CUDATimer(times, "fwd_decode", enable=enable_timer):
        output = feature_acts @ W_O + b_O

    return output, times


def topk_lorsa_custom_flash_sparse_decode(
    activations: torch.Tensor,
    W_Q: torch.Tensor,
    b_Q: torch.Tensor,
    W_K: torch.Tensor,
    b_K: torch.Tensor,
    W_V: torch.Tensor,
    b_V: torch.Tensor,
    W_O: torch.Tensor,
    b_O: torch.Tensor,
    topk: int,
    causal: bool = False,
    softmax_scale: float | None = None,
    times: dict[str, float] = {},
    enable_timer: bool = True,
) -> torch.Tensor:
    d_sae = W_O.shape[0]

    with CUDATimer(times, "fwd_compute_qkv", enable=enable_timer):
        q, k, v = _compute_qkv(activations, W_Q, b_Q, W_K, b_K, W_V, b_V)
        value = v.reshape(*k.shape[:3], -1)  # (batch, seq_len, n_qk_heads, d_v_head)

    with CUDATimer(times, "fwd_compute_hidden_pre", enable=enable_timer):
        z = flash_attn_func(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            value.to(torch.bfloat16),
            causal,
            softmax_scale,
        )
        hidden_pre = z.reshape(*v.shape).to(W_O.dtype)

    with CUDATimer(times, "fwd_topk_select_decode", enable=enable_timer):
        hidden_pre_flattened = hidden_pre.reshape(-1, d_sae)
        output, times = topk_decode_sparse_fused(
            hidden_pre_flattened, W_O, b_O, topk, times, enable_timer
        )
        output = output.reshape(*hidden_pre.shape[:2], -1)

    return output, times


class TopKSparseFusedDecode(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        hidden_pre: torch.Tensor,
        W_D: torch.Tensor,
        b_D: torch.Tensor,
        topk: int,
        times: dict[str, float] = {},
        enable_timer: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        d_sae = W_D.shape[0]

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
            with torch.amp.autocast(device_type="cuda", enabled=False):
                output = torch.sparse.mm(feature_acts_sparse, W_D) + b_D

        ctx.save_for_backward(
            hidden_pre,
            W_D,
            b_D,
            feature_acts_sparse,
            topk_indices_sorted,
            topk_values_sorted,
        )
        ctx.times = times
        ctx.enable_timer = enable_timer

        return output, times

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx,
        grad_output: torch.Tensor,
        _: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        (
            hidden_pre,
            W_D,
            b_D,
            feature_acts_sparse,
            topk_indices_sorted,
            topk_values_sorted,
        ) = ctx.saved_tensors
        times = ctx.times
        enable_timer = ctx.enable_timer

        grad_hidden_pre = grad_W_D = grad_b_D = None

        # grad_b_D
        with CUDATimer(times, "bwd_grad_b_D", enable=enable_timer):
            grad_b_D = grad_output.sum(dim=0)

        # grad_W_D
        with CUDATimer(times, "bwd_feature_acts_transpose", enable=enable_timer):
            feature_acts_sparse_T = feature_acts_sparse.to_sparse_coo().T

        with CUDATimer(times, "bwd_feature_acts_to_sparse_csr", enable=enable_timer):
            feature_acts_sparse_T = feature_acts_sparse_T.to_sparse_csr()

        with CUDATimer(times, "bwd_grad_W_D_mul", enable=enable_timer):
            with torch.amp.autocast(device_type="cuda", enabled=False):
                grad_W_D = torch.sparse.mm(feature_acts_sparse_T, grad_output)

        with CUDATimer(times, "bwd_masked_matmul", enable=enable_timer):
            grad_topk_values = masked_matmul(grad_output, W_D, topk_indices_sorted)

        with CUDATimer(times, "bwd_build_grad_hidden_pre", enable=enable_timer):
            grad_hidden_pre = torch.zeros_like(hidden_pre)
            grad_hidden_pre.scatter_(
                dim=1, index=topk_indices_sorted, src=grad_topk_values
            )

        return grad_hidden_pre, grad_W_D, grad_b_D, None, None, None


topk_decode_sparse_fused = TopKSparseFusedDecode.apply


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, k, v, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k: (batch_size, seqlen_k, nheads, headdim)
        v: (batch_size, seqlen_k, nheads, headdim_v)  # headdim_v can be different from headdim
        Returns: (batch_size, seqlen_q, nheads, headdim_v)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, ctx.softmax_scale = flash_attn_forward(
            q, k, v, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            flash_attn_backward(
                do,  # [batch_size, seq_len_q, n_heads, headdim_v]
                q,
                k,
                v,
                o,
                lse,
                dq,
                dk,
                dv,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dk, dv, None, None, None


flash_attn_func = FlashAttnFunc.apply
