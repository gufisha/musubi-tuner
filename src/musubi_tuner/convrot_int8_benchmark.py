"""Standalone ConvRot INT8 Linear smoke test and microbenchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn as nn

from musubi_tuner.modules.convrot_int8 import (
    CONVROT_INT8_ROTATION_BUFFER,
    CONVROT_INT8_SCALE_BUFFER,
    apply_convrot_int8_monkey_patch,
    effective_rotation_size,
    quantize_int8_rows,
    resolve_convrot_int8_backend,
    rotate,
    validate_rotation_size,
)


class BenchmarkLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def build_model(args, device: torch.device, dtype: torch.dtype, backend: str):
    rotation_size = effective_rotation_size(args.in_features, args.rotation_size)
    if args.in_features % 16 or args.out_features % 8 or rotation_size < 16:
        raise ValueError("benchmark shape is not eligible: in must be divisible by 16, out by 8, rotation >= 16")

    synchronize(device)
    started = time.perf_counter()
    weight = torch.randn(args.out_features, args.in_features, device=device, dtype=torch.float32) * args.weight_std
    codes, scales = quantize_int8_rows(rotate(weight, rotation_size))
    del weight
    bias = torch.zeros(args.out_features, device=device, dtype=dtype) if args.bias else None
    synchronize(device)
    conversion_seconds = time.perf_counter() - started

    with torch.device("meta"):
        model = BenchmarkLinear(args.in_features, args.out_features, args.bias, torch.device("meta"), dtype)
    state = {
        "linear.weight": codes,
        f"linear.{CONVROT_INT8_SCALE_BUFFER}": scales.contiguous().view(torch.uint8),
        f"linear.{CONVROT_INT8_ROTATION_BUFFER}": torch.tensor(rotation_size, dtype=torch.int64, device=device),
    }
    if bias is not None:
        state["linear.bias"] = bias
    apply_convrot_int8_monkey_patch(model, state, backend)
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False)
    return model, rotation_size, conversion_seconds


def run_benchmark(args) -> dict:
    validate_rotation_size(args.rotation_size)
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    backend = resolve_convrot_int8_backend(device, allow_bf16_fallback=args.allow_bf16_fallback)
    model, effective_rotation, conversion_seconds = build_model(args, device, dtype, backend)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn(args.rows, args.in_features, device=device, dtype=dtype, requires_grad=args.backward)

    def one_step(check_finite: bool = False) -> None:
        if x.grad is not None:
            x.grad = None
        output = model(x)
        if args.backward:
            output.float().square().mean().backward()
        if check_finite and not torch.isfinite(output).all():
            raise RuntimeError("benchmark produced non-finite output")

    one_step(check_finite=True)
    for _ in range(args.warmup):
        one_step()
    synchronize(device)

    samples_ms = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        one_step()
        synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "backend": backend,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "dtype": args.dtype,
        "shape": {"rows": args.rows, "in_features": args.in_features, "out_features": args.out_features},
        "requested_rotation_size": args.rotation_size,
        "effective_rotation_size": effective_rotation,
        "backward": args.backward,
        "conversion_seconds": conversion_seconds,
        "iterations": args.iterations,
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "peak_memory_bytes": peak_memory,
    }


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--in_features", type=int, default=5120)
    parser.add_argument("--out_features", type=int, default=5120)
    parser.add_argument("--rotation_size", type=int, default=256)
    parser.add_argument("--weight_std", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--no_backward", action="store_false", dest="backward")
    parser.add_argument("--allow_bf16_fallback", action="store_true")
    parser.set_defaults(backward=True)
    return parser


def main() -> None:
    args = setup_parser().parse_args()
    if args.rows < 1 or args.in_features < 1 or args.out_features < 1:
        raise ValueError("rows, in_features, and out_features must be positive")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    print(json.dumps(run_benchmark(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
