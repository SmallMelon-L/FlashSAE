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
