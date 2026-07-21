# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The INT8 GEMM structure and tuning configurations are adapted from
# kohya-ss/musubi-tuner PR #1008, which vendors the implementation from
# comfy-kitchen (itself derived from dxqb/OneTrainer and ComfyUI-Flux2-INT8):
# https://github.com/kohya-ss/musubi-tuner/pull/1008
#
# Modifications for the WAN ConvRot training path:
# - keep rotation and activation quantization in the existing WAN implementation
# - accept already-quantized, optionally padded activation rows
# - use independent, unwrapped output coordinates with complete tail masks
# - expose only the per-channel INT8 GEMM plus fused dequantization/bias epilogue

"""Fused Triton INT8 GEMM used by the WAN ConvRot forward path."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the optional runtime package
    triton = None
    tl = None
    HAS_TRITON = False


def is_recoverable_triton_error(exc: Exception) -> bool:
    """Return whether retrying with the legacy GEMM is safe.

    Import, compilation, and launch-configuration failures happen before a CUDA
    kernel executes, so the existing ``torch._int_mm`` path can safely take
    over. Runtime CUDA errors (including OOM and illegal accesses) are not
    recoverable here because the device context may already be unhealthy.
    """

    if isinstance(exc, (ImportError, ModuleNotFoundError, NotImplementedError)):
        return True

    recoverable_triton_base_names = {
        "AutotunerError",
        "CompilationError",
        "CompileTimeAssertionFailure",
        "InterpreterError",
        "OutOfResources",
        "PTXASError",
    }
    return any(
        error_type.__module__.startswith("triton.") and error_type.__name__ in recoverable_triton_base_names
        for error_type in type(exc).__mro__
    )


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config(
                {"block_m": 128, "block_n": 256, "block_k": 64, "group_size_m": 8},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"block_m": 64, "block_n": 256, "block_k": 32, "group_size_m": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"block_m": 128, "block_n": 128, "block_k": 32, "group_size_m": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"block_m": 128, "block_n": 64, "block_k": 32, "group_size_m": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"block_m": 64, "block_n": 128, "block_k": 32, "group_size_m": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"block_m": 128, "block_n": 32, "block_k": 32, "group_size_m": 8},
                num_stages=4,
                num_warps=4,
            ),
        ],
        key=["m", "n", "k"],
    )
    @triton.jit
    def _int8_matmul_dequant_per_row_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        bias_ptr,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        group_size_m: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        """Compute dequant(A_int8 @ B_int8) with row/channel scales."""

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(m, block_m)
        num_pid_n = tl.cdiv(n, block_n)
        num_pid_in_group = group_size_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size_m
        actual_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + (pid % actual_group_size_m)
        pid_n = (pid % num_pid_in_group) // actual_group_size_m

        # Keep these coordinates unwrapped. Modulo-based load coordinates are a
        # common GEMM optimization, but they must never be reused for stores: on
        # partial tiles that aliases tail rows/columns onto valid output entries.
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)
        for k_offset in range(0, k, block_k):
            k_remaining = k - k_offset
            a_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k_remaining)
            b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < n)
            a = tl.load(a_ptrs, mask=a_mask, other=0)
            b = tl.load(b_ptrs, mask=b_mask, other=0)
            accumulator += tl.dot(a, b)
            a_ptrs += block_k * stride_ak
            b_ptrs += block_k * stride_bk

        scale_a = tl.load(a_scale_ptr + offs_m, mask=offs_m < m, other=0.0)
        scale_b = tl.load(b_scale_ptr + offs_n, mask=offs_n < n, other=0.0)
        output = accumulator.to(tl.float32) * (scale_a[:, None] * scale_b[None, :])

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < n, other=0.0).to(tl.float32)
            output += bias[None, :]

        output_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        output_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        tl.store(output_ptrs, output, mask=output_mask)


def fused_int8_matmul_dequant(
    activation_codes: torch.Tensor,
    weight: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    rows: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Run INT8 GEMM with per-row/per-channel dequantization fused into the store.

    ``activation_codes`` may contain padded rows, but only the first ``rows``
    participate. ``weight`` is stored in normal Linear layout ``[N, K]`` and is
    read as the logical GEMM RHS ``[K, N]`` through its strides, without a
    transposed copy.
    """

    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if not activation_codes.is_cuda or not weight.is_cuda:
        raise RuntimeError("the fused INT8 kernel requires CUDA tensors")
    if activation_codes.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("activation_codes and weight must both be torch.int8")
    if activation_codes.ndim != 2 or weight.ndim != 2:
        raise ValueError("activation_codes and weight must both be two-dimensional")
    if activation_codes.shape[1] != weight.shape[1]:
        raise ValueError(f"INT8 GEMM reduction mismatch: activation K={activation_codes.shape[1]}, weight K={weight.shape[1]}")
    if rows < 1 or rows > activation_codes.shape[0]:
        raise ValueError(f"rows must be within [1, {activation_codes.shape[0]}], got {rows}")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported fused INT8 output dtype: {out_dtype}")

    m = rows
    n, k = weight.shape
    if activation_scales.numel() < m:
        raise ValueError("activation_scales does not cover all requested rows")
    if weight_scales.numel() != n:
        raise ValueError(f"weight_scales must contain {n} values, got {weight_scales.numel()}")
    if bias is not None and bias.numel() != n:
        raise ValueError(f"bias must contain {n} values, got {bias.numel()}")

    activation_scales = activation_scales.reshape(-1)
    weight_scales = weight_scales.reshape(-1)
    output = torch.empty((m, n), device=activation_codes.device, dtype=out_dtype)
    bias_ptr = bias if bias is not None else activation_scales

    def grid(meta):
        return (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]),)

    _int8_matmul_dequant_per_row_kernel[grid](
        a_ptr=activation_codes,
        b_ptr=weight,
        c_ptr=output,
        a_scale_ptr=activation_scales,
        b_scale_ptr=weight_scales,
        bias_ptr=bias_ptr,
        m=m,
        n=n,
        k=k,
        stride_am=activation_codes.stride(0),
        stride_ak=activation_codes.stride(1),
        stride_bk=weight.stride(1),
        stride_bn=weight.stride(0),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        has_bias=bias is not None,
    )
    return output


__all__ = ["HAS_TRITON", "fused_int8_matmul_dequant", "is_recoverable_triton_error"]
