import torch

from ..config import LorsaConfig
from .time import CUDATimer


def generate_lorsa_data(
    config: LorsaConfig,
) -> dict[str, torch.Tensor]:
    activations = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        device=config.device,
        dtype=config.dtype,
    )
    activations /= activations.norm(dim=-1, keepdim=True)
    W_Q = torch.randn(
        config.n_qk_heads,
        config.d_model,
        config.d_qk_head,
        device=config.device,
        dtype=config.dtype,
    )
    W_Q /= W_Q.norm(dim=1, keepdim=True)
    b_Q = torch.randn(
        config.n_qk_heads,
        config.d_qk_head,
        device=config.device,
        dtype=config.dtype,
    )
    b_Q /= b_Q.norm(dim=-1, keepdim=True)
    W_K = torch.randn(
        config.n_qk_heads,
        config.d_model,
        config.d_qk_head,
        device=config.device,
        dtype=config.dtype,
    )
    W_K /= W_K.norm(dim=1, keepdim=True)
    b_K = torch.randn(
        config.d_qk_head,
        device=config.device,
        dtype=config.dtype,
    )
    b_K /= b_K.norm(dim=-1, keepdim=True)
    W_V = torch.randn(
        config.n_ov_heads,
        config.d_model,
        device=config.device,
        dtype=config.dtype,
    )
    W_V /= W_V.norm(dim=1, keepdim=True)
    b_V = torch.randn(
        config.n_ov_heads,
        device=config.device,
        dtype=config.dtype,
    )
    b_V /= b_V.norm(dim=-1, keepdim=True)
    W_O = torch.randn(
        config.n_ov_heads,
        config.d_model,
        device=config.device,
        dtype=config.dtype,
    )
    W_O /= W_O.norm(dim=1, keepdim=True)
    b_O = torch.randn(
        config.d_model,
        device=config.device,
        dtype=config.dtype,
    )
    b_O /= b_O.norm(dim=-1, keepdim=True)
    return {
        "activations": activations,
        "W_Q": W_Q,
        "b_Q": b_Q,
        "W_K": W_K,
        "b_K": b_K,
        "W_V": W_V,
        "b_V": b_V,
        "W_O": W_O,
        "b_O": b_O,
        "topk": config.topk,
        "causal": config.causal,
        "softmax_scale": config.softmax_scale,
    }


def lorsa_fwd_bwd(
    fwd_func,
    activations,
    W_Q,
    b_Q,
    W_K,
    b_K,
    W_V,
    b_V,
    W_O,
    b_O,
    topk,
    causal,
    softmax_scale,
    times={},
    enable_total_timer: bool = True,
):
    activations = activations.clone().detach().requires_grad_(False)
    W_Q = W_Q.clone().detach().requires_grad_(True)
    b_Q = b_Q.clone().detach().requires_grad_(True)
    W_K = W_K.clone().detach().requires_grad_(True)
    b_K = b_K.clone().detach().requires_grad_(True)
    W_V = W_V.clone().detach().requires_grad_(True)
    b_V = b_V.clone().detach().requires_grad_(True)
    W_O = W_O.clone().detach().requires_grad_(True)
    b_O = b_O.clone().detach().requires_grad_(True)
    W_O = W_O.clone().detach().requires_grad_(True)
    b_O = b_O.clone().detach().requires_grad_(True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        with CUDATimer(times, "forward", enable=enable_total_timer):
            output, times = fwd_func(
                activations,
                W_Q,
                b_Q,
                W_K,
                b_K,
                W_V,
                b_V,
                W_O,
                b_O,
                topk,
                causal,
                softmax_scale,
                times,
                enable_timer=not enable_total_timer,
            )

        with CUDATimer(times, "compute_loss", enable=enable_total_timer):
            loss = (output - activations).pow(2).sum(dim=-1).mean()

    with CUDATimer(times, "backward", enable=enable_total_timer):
        loss.backward()

    result_dict = {
        "output": output,
        "W_Q.grad": W_Q.grad,
        "b_Q.grad": b_Q.grad,
        "W_K.grad": W_K.grad,
        "b_K.grad": b_K.grad,
        "W_V.grad": W_V.grad,
        "b_V.grad": b_V.grad,
        "W_O.grad": W_O.grad,
        "b_O.grad": b_O.grad,
    }

    return result_dict, times
