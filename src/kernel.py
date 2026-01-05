import math

import triton
import triton.language as tl

import torch


@triton.jit
def masked_matmul_kernel(
    # ptr
    output_ptr,  # [batch_size, k]
    a_ptr,  # [batch_size, d_model]
    b_ptr,  # [n_cols, d_model]  (e.g. W_D: [d_sae, d_model])
    indices_ptr,  # [batch_size, k]
    # shape
    batch_size,
    k,
    d_model,
    n_cols,
    # strides
    stride_a_batch,
    stride_a_d,
    stride_b_row,
    stride_b_d,
    stride_idx_batch,
    stride_idx_k,
    stride_out_batch,
    stride_out_k,
    # block size
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // k
    k_idx = pid % k

    if batch_id >= batch_size:
        return

    col_idx = tl.load(indices_ptr + batch_id * stride_idx_batch + k_idx * stride_idx_k)

    accumulator = 0.0

    for d_start in range(0, d_model, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask = d_offsets < d_model

        a_vals = tl.load(
            a_ptr + batch_id * stride_a_batch + d_offsets * stride_a_d,
            mask=mask,
            other=0.0,
        )

        b_vals = tl.load(
            b_ptr + col_idx * stride_b_row + d_offsets * stride_b_d,
            mask=mask,
            other=0.0,
        )

        accumulator += tl.sum(a_vals * b_vals)

    tl.store(
        output_ptr + batch_id * stride_out_batch + k_idx * stride_out_k, accumulator
    )


def masked_matmul(
    a: torch.Tensor, b: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    assert a.is_cuda, f"a must be on CUDA, got {a.device}"
    assert b.is_cuda, f"b must be on CUDA, got {b.device}"
    assert indices.is_cuda, f"indices must be on CUDA, got {indices.device}"
    assert a.dim() == 2 and b.dim() == 2 and indices.dim() == 2
    assert a.shape[0] == indices.shape[0]
    assert a.shape[1] == b.shape[1]

    batch_size, d_model = a.shape
    n_cols = b.shape[0]
    k = indices.shape[1]

    a = a.contiguous()
    b = b.contiguous()
    indices = indices.contiguous().to(torch.int64)

    output = torch.empty((batch_size, k), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_D = triton.next_power_of_2(min(d_model, 1024))

    grid = (batch_size * k,)

    masked_matmul_kernel[grid](
        output,
        a,
        b,
        indices,
        batch_size,
        k,
        d_model,
        n_cols,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return output


@triton.jit
def masked_matmul_with_col_sum_kernel(
    # ptr
    output_ptr,  # [batch_size, k]
    col_sum_ptr,  # [n_cols]
    a_ptr,  # [batch_size, d_model]
    b_ptr,  # [n_cols, d_model]
    indices_ptr,  # [batch_size, k]
    # shape
    batch_size,
    k,
    d_model,
    n_cols,
    # strides
    stride_a_batch,
    stride_a_d,
    stride_b_row,
    stride_b_d,
    stride_idx_batch,
    stride_idx_k,
    stride_out_batch,
    stride_out_k,
    # block size
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // k
    k_idx = pid % k

    if batch_id >= batch_size:
        return

    col_idx = tl.load(indices_ptr + batch_id * stride_idx_batch + k_idx * stride_idx_k)

    accumulator = 0.0

    for d_start in range(0, d_model, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask = d_offsets < d_model

        a_vals = tl.load(
            a_ptr + batch_id * stride_a_batch + d_offsets * stride_a_d,
            mask=mask,
            other=0.0,
        )

        b_vals = tl.load(
            b_ptr + col_idx * stride_b_row + d_offsets * stride_b_d,
            mask=mask,
            other=0.0,
        )

        accumulator += tl.sum(a_vals * b_vals)

    # 写入 output
    tl.store(
        output_ptr + batch_id * stride_out_batch + k_idx * stride_out_k, accumulator
    )

    # 用 atomic add 累加到 col_sum[col_idx]
    tl.atomic_add(col_sum_ptr + col_idx, accumulator)


def masked_matmul_with_col_sum(
    a: torch.Tensor, b: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    assert a.is_cuda, f"a must be on CUDA, got {a.device}"
    assert b.is_cuda, f"b must be on CUDA, got {b.device}"
    assert indices.is_cuda, f"indices must be on CUDA, got {indices.device}"
    assert a.dim() == 2 and b.dim() == 2 and indices.dim() == 2
    assert a.shape[0] == indices.shape[0]
    assert a.shape[1] == b.shape[1]

    batch_size, d_model = a.shape
    n_cols = b.shape[0]
    k = indices.shape[1]

    a = a.contiguous()
    b = b.contiguous()
    indices = indices.contiguous().to(torch.int64)

    output = torch.empty((batch_size, k), device=a.device, dtype=a.dtype)
    col_sum = torch.zeros(n_cols, device=a.device, dtype=a.dtype)

    BLOCK_SIZE_D = triton.next_power_of_2(min(d_model, 1024))

    grid = (batch_size * k,)

    masked_matmul_with_col_sum_kernel[grid](
        output,
        col_sum,
        a,
        b,
        indices,
        batch_size,
        k,
        d_model,
        n_cols,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return output, col_sum


@triton.jit
def triton_sparse_dense_matmul_kernel(
    sparse_indices_ptr,
    sparse_values_ptr,
    dense_ptr,
    out_ptr,
    stride_dn,
    stride_db,
    A,
    B,
    N,
    K,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    sparse_indices is shape (A, K)
    sparse_values is shape (A, K)
    dense is shape (N, B), contiguous along B
    out is shape (A, B)
    """

    pid = tl.program_id(0)

    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    sparse_indices = tl.load(
        sparse_indices_ptr + pid * K + offsets_k, mask=offsets_k < K
    )  # shape (K,)
    sparse_values = tl.load(
        sparse_values_ptr + pid * K + offsets_k, mask=offsets_k < K
    )  # shape (K,)

    accum = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)

    offsets_b = tl.arange(0, BLOCK_SIZE_B)

    for k in range(K):
        # workaround to do sparse_indices[k]
        i = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_SIZE_K) == k,
                sparse_indices,
                tl.zeros((BLOCK_SIZE_K,), dtype=tl.int64),
            )
        )
        # workaround to do sparse_values[k]
        v = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_SIZE_K) == k,
                sparse_values,
                tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32),
            )
        )

        tl.device_assert(i < N)
        if v != 0:
            accum += v * tl.load(
                dense_ptr + i * stride_dn + offsets_b * stride_db, mask=offsets_b < B
            )

    tl.store(
        out_ptr + pid * B + offsets_b, accum.to(sparse_values.dtype), mask=offsets_b < B
    )


def triton_sparse_dense_matmul(
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    """
    calculates sparse @ dense (i.e reducing along the uncollated dimension of sparse)
    dense must be contiguous along dim 0 (in other words, dense.T is contiguous)

    sparse_indices is shape (batch_size, k)
    sparse_values is shape (batch_size, k)
    dense is shape (d_sae, d_model)   ## TODO: check if this is correct, this is compatible with current situation

    output is shape (batch_size, d_model)
    """
    N = dense.shape[0]
    assert sparse_indices.shape == sparse_values.shape
    assert sparse_indices.is_contiguous()
    assert sparse_values.is_contiguous()
    assert dense.is_contiguous()  # contiguous along B

    A = sparse_indices.shape[0]
    K = sparse_indices.shape[1]
    B = dense.shape[1]

    out = torch.zeros(A, B, device=dense.device, dtype=sparse_values.dtype)

    triton_sparse_dense_matmul_kernel[(A,)](  # type: ignore
        sparse_indices,
        sparse_values,
        dense,
        out,
        stride_dn=dense.stride(0),
        stride_db=dense.stride(1),
        A=A,
        B=B,
        N=N,
        K=K,
        BLOCK_SIZE_K=triton.next_power_of_2(K),  # type: ignore
        BLOCK_SIZE_B=triton.next_power_of_2(B),  # type: ignore
    )
    return out


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "EVEN_HEADDIM_V": lambda args: args["headdim_v"] == args["BLOCK_HEADDIM_V"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    headdim_v,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_HEADDIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
    # Initialize pointers to Q, K, V
    q_ptrs = (
        Q
        + off_b * stride_qb
        + off_h * stride_qh
        + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K
        + off_b * stride_kb
        + off_h * stride_kh
        + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V
        + off_b * stride_vb
        + off_h * stride_vh
        + (offs_n[:, None] * stride_vn + offs_d_v[None, :])
    )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM_V], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k), input_precision="tf32x3")
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(
                offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf")
            )
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # -- update output accumulator --
        acc_o = acc_o * acc_o_scale[:, None]
        # load v with headdim_v
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM_V:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=offs_d_v[None, :] < headdim_v,
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM_V:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k)
                    & (offs_d_v[None, :] < headdim_v),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v, input_precision="tf32x3")

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output (using headdim_v)
    offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d_v[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM_V:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d_v[None, :] < headdim_v)
    else:
        if EVEN_HEADDIM_V:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d_v[None, :] < headdim_v),
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim_v,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
    # load (o and do have headdim_v dimension)
    o = tl.load(
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d_v[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d_v[None, :] < headdim_v),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d_v[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d_v[None, :] < headdim_v),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    offs_d_v,
    seqlen_k,
    headdim,
    headdim_v,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_HEADDIM_V: tl.constexpr,
):
    # Store dk (with headdim) and dv (with headdim_v) separately
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
        if EVEN_HEADDIM_V:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d_v[None, :] < headdim_v)
    else:
        if EVEN_HEADDIM:
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(
                dk_ptrs,
                dk,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            )
        if EVEN_HEADDIM_V:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(
                dv_ptrs,
                dv,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d_v[None, :] < headdim_v),
            )


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    headdim_v,
    ATOMIC_ADD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_HEADDIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
    # initialize pointers to value-like data
    # q, k, dq, dk use headdim; v, do, dv use headdim_v
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d_v[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d_v[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    # initialize dv (headdim_v) and dk (headdim)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d_v[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            offs_d_v,
            seqlen_k,
            headdim,
            headdim_v,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            EVEN_HEADDIM_V=EVEN_HEADDIM_V,
        )
        return
    # k (headdim) and v (headdim_v) stay in SRAM throughout
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        if EVEN_HEADDIM_V:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=offs_d_v[None, :] < headdim_v, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        if EVEN_HEADDIM_V:
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d_v[None, :] < headdim_v),
                other=0.0,
            )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q (headdim)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k), input_precision="tf32x3")
        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp(qk - lse_i[:, None])
        # load do (headdim_v)
        if EVEN_M & EVEN_HEADDIM_V:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q)
                & (offs_d_v[None, :] < headdim_v),
                other=0.0,
            )
        # compute dv = p^T @ do (headdim_v)
        dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="tf32x3")
        # compute dp = do @ v^T (using headdim_v)
        if not (EVEN_M & EVEN_HEADDIM_V):
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v), input_precision="tf32x3")
        if not EVEN_HEADDIM_V:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        Di = tl.load(D + offs_m_curr)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = ds^T @ q (headdim)
        dk += tl.dot(tl.trans(ds), q, input_precision="tf32x3")
        # compute dq = ds @ k (headdim)
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k, input_precision="tf32x3")
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k, input_precision="tf32x3")
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q)
                        & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k, input_precision="tf32x3")
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q)
                        & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:
            dq = tl.dot(ds, k, input_precision="tf32x3")
            if EVEN_M & EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q)
                        & (offs_d[None, :] < headdim),
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d_v[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        offs_d_v,
        seqlen_k,
        headdim,
        headdim_v,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
        EVEN_HEADDIM_V=EVEN_HEADDIM_V,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "IS_CAUSAL",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "EVEN_HEADDIM_V": lambda args: args["headdim_v"] == args["BLOCK_HEADDIM_V"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    headdim_v,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_HEADDIM_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_HEADDIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                headdim_v,
                ATOMIC_ADD=False,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                BLOCK_HEADDIM_V=BLOCK_HEADDIM_V,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                EVEN_HEADDIM_V=EVEN_HEADDIM_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            headdim_v,
            ATOMIC_ADD=True,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_HEADDIM_V=BLOCK_HEADDIM_V,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            EVEN_HEADDIM_V=EVEN_HEADDIM_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _flash_attn_forward(q, k, v, causal=False, softmax_scale=None):
    # shape constraints
    # q, k: (batch, seqlen, nheads, d)
    # v: (batch, seqlen_k, nheads, d_v) where d_v can be different from d
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, d_v = v.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape[:3] == (batch, seqlen_k, nheads)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty(
        (batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32
    )
    # output has shape (batch, seqlen_q, nheads, d_v)
    o = torch.empty((batch, seqlen_q, nheads, d_v), device=q.device, dtype=q.dtype)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_HEADDIM_V = max(triton.next_power_of_2(d_v), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        d_v,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        causal,
        BLOCK_HEADDIM,
        BLOCK_HEADDIM_V,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale


def _flash_attn_backward(
    do, q, k, v, o, lse, dq, dk, dv, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, d_v = v.shape
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_HEADDIM_V = max(triton.next_power_of_2(d_v), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d_v,
        BLOCK_M=128,
        BLOCK_HEADDIM_V=BLOCK_HEADDIM_V,
    )

    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        d_v,
        seqlen_q // 32,
        seqlen_k // 32,
        causal,
        BLOCK_HEADDIM,
        BLOCK_HEADDIM_V,
    )
    dq.copy_(dq_accum)


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
        o, lse, ctx.softmax_scale = _flash_attn_forward(
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
            _flash_attn_backward(
                do,
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
