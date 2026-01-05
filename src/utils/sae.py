import torch

from ..config import SAEConfig
from .time import CUDATimer


def generate_sae_data(
    config: SAEConfig,
) -> dict[str, torch.Tensor]:
    activations: torch.Tensor = torch.randn(
        config.batch_size, config.d_model, device=config.device, dtype=config.dtype
    )
    activations /= activations.norm(dim=-1, keepdim=True)
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
        "topk": config.topk,
    }


def sae_fwd_bwd(fwd_func, activations, W_E, b_E, W_D, b_D, topk, times={}):
    activations = activations.clone().detach().requires_grad_(False)
    W_E = W_E.clone().detach().requires_grad_(True)
    W_D = W_D.clone().detach().requires_grad_(True)
    b_E = b_E.clone().detach().requires_grad_(True)
    b_D = b_D.clone().detach().requires_grad_(True)

    reconstruct, times = fwd_func(
        activations, W_E, b_E, W_D, b_D, topk, times, enable_timer=True
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


def sae_fwd_bwd_time(fwd_func, activations, W_E, b_E, W_D, b_D, topk, times={}):
    activations = activations.clone().detach().requires_grad_(False)
    W_E = W_E.clone().detach().requires_grad_(True)
    W_D = W_D.clone().detach().requires_grad_(True)
    b_E = b_E.clone().detach().requires_grad_(True)
    b_D = b_D.clone().detach().requires_grad_(True)

    with CUDATimer(times, "forward"):
        reconstruct, times = fwd_func(
            activations, W_E, b_E, W_D, b_D, topk, times, enable_timer=False
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
