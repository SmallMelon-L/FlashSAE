import torch
from tqdm import tqdm

from ..config import TestConfig, SAEConfig, SAETestConfig, LorsaConfig, LorsaTestConfig
from .sae import generate_sae_data, sae_fwd_bwd, sae_fwd_bwd_time


def compare_reconstruct(
    gen_data_func,
    fwd_func_1,
    fwd_func_2,
    config: TestConfig,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    verbose: bool = True,
):
    data_dict = gen_data_func(config)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        reconstruct_1, _ = fwd_func_1(**data_dict)
        reconstruct_2, _ = fwd_func_2(**data_dict)
    reconstruct_1 = reconstruct_1.to(config.dtype)
    reconstruct_2 = reconstruct_2.to(config.dtype)
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
    gen_data_func,
    fwd_func_1,
    fwd_func_2,
    fwd_bwd_func,
    config: TestConfig,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    verbose: bool = True,
):
    data_dict = gen_data_func(config)

    result_dict_1, _ = fwd_bwd_func(
        fwd_func=fwd_func_1,
        **data_dict,
    )

    result_dict_2, _ = fwd_bwd_func(
        fwd_func=fwd_func_2,
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
    gen_data_func,
    fwd_func_1,
    fwd_func_2,
    fwd_bwd_func,
    config: TestConfig,
    verbose: bool = True,
):
    data_dict = gen_data_func(config)

    if verbose:
        iters = tqdm(range(config.preheat_repeat))
    else:
        iters = range(config.preheat_repeat)

    for _ in iters:
        fwd_bwd_func(
            fwd_func=fwd_func_1,
            **data_dict,
        )
        fwd_bwd_func(
            fwd_func=fwd_func_2,
            **data_dict,
        )

    fwd_bwd_times_1 = {}
    fwd_bwd_times_2 = {}
    if verbose:
        iters = tqdm(range(config.timing_repeat))
    else:
        iters = range(config.timing_repeat)
    for _ in iters:
        result_dict_1, fwd_bwd_times_1 = fwd_bwd_func(
            fwd_func=fwd_func_1,
            **data_dict,
            times=fwd_bwd_times_1,
            enable_total_timer=True,
        )

        result_dict_2, fwd_bwd_times_2 = fwd_bwd_func(
            fwd_func=fwd_func_2,
            **data_dict,
            times=fwd_bwd_times_2,
            enable_total_timer=True,
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
        _, times_1 = fwd_bwd_func(
            fwd_func=fwd_func_1,
            **data_dict,
            times=times_1,
            enable_total_timer=False,
        )

        _, times_2 = fwd_bwd_func(
            fwd_func=fwd_func_2,
            **data_dict,
            times=times_2,
            enable_total_timer=False,
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
