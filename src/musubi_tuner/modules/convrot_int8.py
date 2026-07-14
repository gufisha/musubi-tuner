"""Experimental ConvRot W8A8 support for frozen Linear layers.

The W8A8 algorithm is derived from Ostris' MIT-licensed ConvRot implementation:
https://github.com/ostris/ai-toolkit/blob/0d53e5e1f9db022559f9f9cb8fd4f73b5f10e0c7/toolkit/util/convrot_quant.py

ConvRot itself is described in "ConvRot: Rotation-Based Plug-and-Play 4-bit
Quantization for Diffusion Transformers" (arXiv:2512.03673). The paper covers
W4A4 inference; the W8A8 training path here is experimental and must be measured
independently.

The implementation deliberately keeps quantized codes in ``nn.Linear.weight``.
That preserves module names and lets Musubi's existing block offloaders stream the
large INT8 payload without learning about a new tensor type. Per-output-channel
scales are stored as a uint8 byte view so model-wide dtype casts cannot alter them.
"""

# Portions derived from ai-toolkit:
#
# MIT License
# Copyright (c) 2024 Ostris, LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from types import MethodType
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from musubi_tuner.utils.device_utils import clean_memory_on_device, synchronize_device
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, get_split_weight_filenames

logger = logging.getLogger(__name__)

CONVROT_INT8_BACKEND_NATIVE = "native_int8"
CONVROT_INT8_BACKEND_FALLBACK = "bf16_fallback"
CONVROT_INT8_SCALE_BUFFER = "convrot_int8_scale"
CONVROT_INT8_ROTATION_BUFFER = "convrot_int8_rotation"

_hadamard_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_triton_ok: Optional[bool] = None
_int8_kernels = None
_probe_cache: dict[str, "ConvRotInt8Probe"] = {}


@dataclass(frozen=True)
class ConvRotInt8Probe:
    supported: bool
    device: str
    reason: str


def is_power_of_four(value: int) -> bool:
    if value < 1:
        return False
    while value > 1:
        if value % 4:
            return False
        value //= 4
    return True


def validate_rotation_size(rotation_size: int) -> None:
    if rotation_size < 16 or not is_power_of_four(rotation_size):
        raise ValueError(f"--convrot_int8_rotation_size must be a power of four and at least 16 (received {rotation_size})")


def largest_pow4_divisor(value: int) -> int:
    divisor = 1
    while value % (divisor * 4) == 0:
        divisor *= 4
    return divisor


def effective_rotation_size(in_features: int, requested: int) -> int:
    validate_rotation_size(requested)
    return min(requested, largest_pow4_divisor(in_features))


def regular_hadamard(rot_size: int, device: Union[str, torch.device], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the deterministic, symmetric regular-Hadamard ConvRot matrix."""
    if not is_power_of_four(rot_size) or rot_size < 4:
        raise ValueError(f"rotation size {rot_size} is not a power of four >= 4")

    device = torch.device(device)
    key = (rot_size, str(device), dtype)
    cached = _hadamard_cache.get(key)
    if cached is not None:
        return cached

    r4 = torch.tensor(
        [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    matrix = r4
    while matrix.shape[0] < rot_size:
        matrix = torch.kron(matrix, r4)
    if matrix.shape[0] != rot_size:
        raise ValueError(f"rotation size {rot_size} is not a power of four")
    matrix = (matrix / rot_size**0.5).to(device=device, dtype=dtype)
    _hadamard_cache[key] = matrix
    return matrix


def rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply the block regular-Hadamard rotation along the final dimension."""
    if rot_size == 1:
        return x
    if x.shape[-1] % rot_size:
        raise ValueError(f"last dimension {x.shape[-1]} is not divisible by rotation size {rot_size}")
    matrix = regular_hadamard(rot_size, x.device, x.dtype)
    original_shape = x.shape
    blocks = x.reshape(-1, original_shape[-1] // rot_size, rot_size)
    return torch.matmul(blocks, matrix).reshape(original_shape)


def quantize_int8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row INT8 quantization returning codes and FP32 scales."""
    x_float = x.float()
    scales = x_float.abs().amax(dim=1) / 127.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    codes = torch.round(x_float / scales.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
    return codes, scales


def _triton_available() -> bool:
    global _triton_ok
    if _triton_ok is None:
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401

            _triton_ok = True
        except Exception:
            _triton_ok = False
            logger.warning(
                "ConvRot INT8: Triton is unavailable; using Torch activation quantization and epilogue kernels. "
                "The result remains native INT8, but performance may be lower."
            )
    return _triton_ok


def _get_int8_kernels():
    global _int8_kernels
    if _int8_kernels is not None:
        return _int8_kernels

    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def int8_act_quant_kernel(x_ptr, q_ptr, scale_ptr, K, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        base = row * K
        maximum = tl.zeros((BLOCK_K,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offsets = k0 + tl.arange(0, BLOCK_K)
            values = tl.load(x_ptr + base + offsets, mask=offsets < K, other=0.0).to(tl.float32)
            maximum = tl.maximum(maximum, tl.abs(values))
        amax = tl.max(maximum, axis=0)
        scale = tl.where(amax > 0, amax / 127.0, 1.0)
        for k0 in range(0, K, BLOCK_K):
            offsets = k0 + tl.arange(0, BLOCK_K)
            mask = offsets < K
            values = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
            codes = libdevice.rint(values / scale)
            codes = tl.minimum(tl.maximum(codes, -127.0), 127.0)
            tl.store(q_ptr + base + offsets, codes.to(tl.int8), mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def int8_epilogue_kernel(
        i32_ptr,
        activation_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        output_ptr,
        N,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        column_block = tl.program_id(1)
        offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        accumulator = tl.load(i32_ptr + row * N + offsets, mask=mask, other=0).to(tl.float32)
        activation_scale = tl.load(activation_scale_ptr + row)
        weight_scale = tl.load(weight_scale_ptr + offsets, mask=mask, other=0.0)
        output = accumulator * (activation_scale * weight_scale)
        if HAS_BIAS:
            output += tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + row * N + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)

    _int8_kernels = (int8_act_quant_kernel, int8_epilogue_kernel)
    return _int8_kernels


def _activation_quant_block_size(features: int) -> int:
    """Choose a power-of-two Triton tile no larger than 2048."""
    return min(2048, 1 << (features - 1).bit_length())


@torch.library.custom_op("musubi_tuner::convrot_int8_act_quant", mutates_args=())
def _int8_act_quant_op(x: torch.Tensor) -> List[torch.Tensor]:
    rows, features = x.shape
    padded_rows = -(-rows // 32) * 32
    x = x.contiguous()
    codes = torch.empty(padded_rows, features, device=x.device, dtype=torch.int8)
    scales = torch.empty(padded_rows, device=x.device, dtype=torch.float32)
    if padded_rows != rows:
        codes[rows:].zero_()
        scales[rows:].fill_(1.0)
    kernel, _ = _get_int8_kernels()
    kernel[(rows,)](x, codes, scales, features, BLOCK_K=_activation_quant_block_size(features), num_warps=8)
    return [codes, scales]


@_int8_act_quant_op.register_fake
def _int8_act_quant_fake(x):
    rows, features = x.shape
    padded_rows = -(-rows // 32) * 32
    return [
        torch.empty(padded_rows, features, device=x.device, dtype=torch.int8),
        torch.empty(padded_rows, device=x.device, dtype=torch.float32),
    ]


@torch.library.custom_op("musubi_tuner::convrot_int8_epilogue", mutates_args=())
def _int8_epilogue_op(
    i32: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: str,
) -> torch.Tensor:
    rows, out_features = i32.shape
    output = torch.empty(rows, out_features, device=i32.device, dtype=getattr(torch, out_dtype))
    _, kernel = _get_int8_kernels()
    kernel[(rows, -(-out_features // 1024))](
        i32,
        activation_scales,
        weight_scales,
        bias if bias is not None else activation_scales,
        output,
        out_features,
        HAS_BIAS=bias is not None,
        BLOCK_N=1024,
        num_warps=4,
    )
    return output


@_int8_epilogue_op.register_fake
def _int8_epilogue_fake(i32, activation_scales, weight_scales, bias, out_dtype):
    return torch.empty(i32.shape, device=i32.device, dtype=getattr(torch, out_dtype))


def _int8_act_quant_padded(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _triton_available() and x.is_cuda:
        return tuple(_int8_act_quant_op(x))
    codes, scales = quantize_int8_rows(x)
    rows = codes.shape[0]
    padded_rows = -(-rows // 32) * 32
    if padded_rows != rows:
        codes = F.pad(codes, (0, 0, 0, padded_rows - rows))
        scales = F.pad(scales, (0, padded_rows - rows), value=1.0)
    return codes, scales


def _int8_epilogue(
    i32: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if _triton_available() and i32.is_cuda:
        return _int8_epilogue_op(i32, activation_scales, weight_scales, bias, str(out_dtype).split(".")[-1])
    output = i32.float() * weight_scales
    output = output * activation_scales.unsqueeze(1)
    if bias is not None:
        output = output + bias.float()
    return output.to(out_dtype)


@torch.library.custom_op("musubi_tuner::convrot_int8_linear_ste", mutates_args=())
def _int8_linear_ste_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scales_u8: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: str,
) -> torch.Tensor:
    rows = x.shape[0]
    activation_codes, activation_scales = _int8_act_quant_padded(x)
    i32 = torch._int_mm(activation_codes, weight.t())
    return _int8_epilogue(
        i32[:rows], activation_scales[:rows], weight_scales_u8.view(torch.float32), bias, getattr(torch, out_dtype)
    )


@_int8_linear_ste_op.register_fake
def _int8_linear_ste_fake(x, weight, weight_scales_u8, bias, out_dtype):
    return torch.empty(x.shape[0], weight.shape[0], device=x.device, dtype=getattr(torch, out_dtype))


def _int8_linear_ste_setup(ctx, inputs, output):
    _x, weight, weight_scales_u8, _bias, _out_dtype = inputs
    ctx.save_for_backward(weight, weight_scales_u8)


def _int8_linear_ste_backward(ctx, grad):
    weight, weight_scales_u8 = ctx.saved_tensors
    scales = weight_scales_u8.view(torch.float32).to(grad.dtype)
    dequantized_weight = weight.to(grad.dtype) * scales.unsqueeze(1)
    return grad @ dequantized_weight, None, None, None, None


_int8_linear_ste_op.register_autograd(_int8_linear_ste_backward, setup_context=_int8_linear_ste_setup)


def probe_convrot_int8(device: Union[str, torch.device], refresh: bool = False) -> ConvRotInt8Probe:
    """Run a real CUDA INT8 GEMM instead of relying on Torch/CUDA version gates."""
    device = torch.device(device)
    cache_key = str(device)
    if not refresh and cache_key in _probe_cache:
        return _probe_cache[cache_key]

    if device.type != "cuda":
        result = ConvRotInt8Probe(False, str(device), "device is not CUDA")
    elif not torch.cuda.is_available():
        result = ConvRotInt8Probe(False, str(device), "torch.cuda.is_available() is false")
    elif not hasattr(torch, "_int_mm"):
        result = ConvRotInt8Probe(False, str(device), "this PyTorch build does not expose torch._int_mm")
    else:
        try:
            left = torch.ones(32, 64, dtype=torch.int8, device=device)
            # Match the real Linear path: qweight is stored [out, in] and the
            # RHS is its column-major transposed view, not a row-major matrix.
            weight = torch.ones(32, 64, dtype=torch.int8, device=device)
            output = torch._int_mm(left, weight.t())
            torch.cuda.synchronize(device)
            if output.dtype != torch.int32 or output.shape != (32, 32):
                raise RuntimeError(f"unexpected output dtype/shape: {output.dtype}, {tuple(output.shape)}")
            if not torch.equal(output, torch.full_like(output, 64)):
                raise RuntimeError("INT8 GEMM smoke test returned an unexpected value")
            result = ConvRotInt8Probe(True, str(device), "native torch._int_mm smoke test passed")
        except Exception as exc:
            result = ConvRotInt8Probe(False, str(device), f"{type(exc).__name__}: {exc}")

    _probe_cache[cache_key] = result
    return result


def resolve_convrot_int8_backend(device: Union[str, torch.device], allow_bf16_fallback: bool = False) -> str:
    probe = probe_convrot_int8(device)
    if probe.supported:
        device_name = torch.cuda.get_device_name(torch.device(device))
        logger.info(
            "ConvRot INT8 preflight passed: device=%s, torch=%s, cuda=%s, gpu=%s",
            probe.device,
            torch.__version__,
            torch.version.cuda,
            device_name,
        )
        return CONVROT_INT8_BACKEND_NATIVE

    message = f"ConvRot INT8 preflight failed on {probe.device}: {probe.reason}"
    if not allow_bf16_fallback:
        raise RuntimeError(
            message + ". Native INT8 is required by default. Pass --convrot_int8_allow_bf16_fallback "
            "only when a dequantized compute fallback is intentional."
        )

    logger.warning("%s. EXPLICIT DEQUANTIZED BF16/FP16 FALLBACK ENABLED; THIS IS NOT AN INT8 SPEED TEST.", message)
    return CONVROT_INT8_BACKEND_FALLBACK


def can_quantize_linear_shape(in_features: int, out_features: int, requested_rotation_size: int) -> bool:
    rotation_size = effective_rotation_size(in_features, requested_rotation_size)
    return in_features % 16 == 0 and out_features % 8 == 0 and rotation_size >= 16


def _scale_buffer(module: nn.Linear) -> torch.Tensor:
    return getattr(module, CONVROT_INT8_SCALE_BUFFER)


def _dequantize_rotated_weight(module: nn.Linear, dtype: torch.dtype) -> torch.Tensor:
    scales = _scale_buffer(module).view(torch.float32)
    return (module.weight.float() * scales.unsqueeze(1)).to(dtype)


def convrot_int8_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Patched frozen-base Linear forward used before the standard LoRA branch."""
    if module.weight.device != x.device or _scale_buffer(module).device != x.device:
        raise RuntimeError(
            "ConvRot INT8 device mismatch: input, INT8 weight, and scales must be colocated "
            f"(input={x.device}, weight={module.weight.device}, scales={_scale_buffer(module).device})"
        )

    rotation_size = module._convrot_int8_rotation_size
    in_features = module.in_features
    original_shape = x.shape
    x_rotated = rotate(x, rotation_size).reshape(-1, in_features)
    backend = module._convrot_int8_backend

    if backend == CONVROT_INT8_BACKEND_NATIVE:
        if x.requires_grad:
            output = _int8_linear_ste_op(
                x_rotated,
                module.weight,
                _scale_buffer(module),
                module.bias,
                str(x.dtype).split(".")[-1],
            )
        else:
            rows = x_rotated.shape[0]
            activation_codes, activation_scales = _int8_act_quant_padded(x_rotated)
            i32 = torch._int_mm(activation_codes, module.weight.t())
            output = _int8_epilogue(
                i32[:rows],
                activation_scales[:rows],
                _scale_buffer(module).view(torch.float32),
                module.bias,
                x.dtype,
            )
    elif backend == CONVROT_INT8_BACKEND_FALLBACK:
        with torch.no_grad():
            activation_codes, activation_scales = quantize_int8_rows(x_rotated.detach())
            dequantized_activation = (activation_codes.float() * activation_scales.unsqueeze(1)).to(x.dtype)
            dequantized_weight = _dequantize_rotated_weight(module, x.dtype)
        activation = x_rotated + (dequantized_activation - x_rotated).detach() if x.requires_grad else dequantized_activation
        output = F.linear(activation, dequantized_weight, module.bias)
    else:
        raise RuntimeError(f"Unknown ConvRot INT8 backend: {backend}")

    return output.reshape(*original_shape[:-1], module.out_features)


def _patched_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return convrot_int8_linear_forward(self, x)


def apply_convrot_int8_monkey_patch(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    backend: str,
) -> nn.Module:
    """Patch only Linears represented by ConvRot scale/rotation state entries."""
    scale_suffix = f".{CONVROT_INT8_SCALE_BUFFER}"
    rotation_suffix = f".{CONVROT_INT8_ROTATION_BUFFER}"
    module_names = {key[: -len(scale_suffix)] for key in state_dict if key.endswith(scale_suffix)}
    modules = dict(model.named_modules())

    patched = 0
    for module_name in sorted(module_names):
        module = modules.get(module_name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"ConvRot state targets non-Linear or missing module: {module_name}")

        scale = state_dict[module_name + scale_suffix]
        rotation = state_dict[module_name + rotation_suffix]
        rotation_size = int(rotation.detach().cpu().item())

        weight_device = module.weight.device
        module.weight = nn.Parameter(
            torch.empty(module.weight.shape, dtype=torch.int8, device=weight_device),
            requires_grad=False,
        )
        module.register_buffer(
            CONVROT_INT8_SCALE_BUFFER,
            torch.empty(scale.shape, dtype=torch.uint8, device=weight_device),
            persistent=True,
        )
        module.register_buffer(
            CONVROT_INT8_ROTATION_BUFFER,
            torch.empty(rotation.shape, dtype=torch.int64, device=weight_device),
            persistent=True,
        )
        module._convrot_int8_rotation_size = rotation_size
        module._convrot_int8_backend = backend
        module._convrot_int8_enabled = True
        module.forward = MethodType(_patched_linear_forward, module)
        patched += 1

    if not patched:
        raise ValueError("ConvRot INT8 did not find any eligible WAN Linear layers to patch")
    logger.info("ConvRot INT8 monkey-patched %d Linear layers using backend=%s", patched, backend)
    return model


def _expand_model_files(model_files: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(model_files, str):
        model_files = [model_files]
    expanded: list[str] = []
    for model_file in model_files:
        split_files = get_split_weight_filenames(model_file)
        expanded.extend(split_files if split_files is not None else [model_file])
    return expanded


def load_safetensors_with_convrot_int8(
    model_files: Union[str, Sequence[str]],
    calc_device: Union[str, torch.device],
    loading_device: Union[str, torch.device],
    weight_dtype: torch.dtype,
    requested_rotation_size: int,
    target_keys: Optional[Sequence[str]] = None,
    exclude_keys: Optional[Sequence[str]] = None,
    disable_numpy_memmap: bool = False,
) -> dict[str, torch.Tensor]:
    """Load and quantize eligible weights one tensor at a time."""
    validate_rotation_size(requested_rotation_size)
    calc_device = torch.device(calc_device)
    loading_device = torch.device(loading_device)
    state_dict: dict[str, torch.Tensor] = {}
    optimized_count = 0
    skipped_shapes: set[tuple[int, ...]] = set()

    def is_target(key: str, value: torch.Tensor) -> bool:
        targeted = (target_keys is None or any(pattern in key for pattern in target_keys)) and key.endswith(".weight")
        excluded = exclude_keys is not None and any(pattern in key for pattern in exclude_keys)
        return targeted and not excluded and value.ndim == 2

    expanded_files = _expand_model_files(model_files)
    logger.info("Loading model files with ConvRot INT8: %s", expanded_files)
    for model_file in expanded_files:
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as file:
            for key in tqdm(file.keys(), desc=f"ConvRot loading {os.path.basename(model_file)}", unit="key"):
                value = file.get_tensor(key)
                if not is_target(key, value):
                    state_dict[key] = value.to(device=loading_device, dtype=weight_dtype)
                    continue

                if value.dtype.itemsize == 1:
                    raise ValueError(f"ConvRot INT8 requires FP16/BF16/FP32 input weights, but {key} is {value.dtype}")

                out_features, in_features = value.shape
                if not can_quantize_linear_shape(in_features, out_features, requested_rotation_size):
                    skipped_shapes.add(tuple(value.shape))
                    state_dict[key] = value.to(device=loading_device, dtype=weight_dtype)
                    continue

                rotation_size = effective_rotation_size(in_features, requested_rotation_size)
                value = value.to(device=calc_device, dtype=torch.float32)
                rotated = rotate(value, rotation_size)
                codes, scales = quantize_int8_rows(rotated)
                del value, rotated

                module_key = key[: -len(".weight")]
                state_dict[key] = codes.to(loading_device)
                state_dict[f"{module_key}.{CONVROT_INT8_SCALE_BUFFER}"] = scales.contiguous().view(torch.uint8).to(loading_device)
                state_dict[f"{module_key}.{CONVROT_INT8_ROTATION_BUFFER}"] = torch.tensor(
                    rotation_size, dtype=torch.int64, device=loading_device
                )
                optimized_count += 1

                if optimized_count % 8 == 0:
                    clean_memory_on_device(calc_device)

    if calc_device == loading_device:
        synchronize_device(calc_device)
    if skipped_shapes:
        logger.warning(
            "ConvRot INT8 left unsupported target shapes in %s: %s",
            weight_dtype,
            ", ".join(str(shape) for shape in sorted(skipped_shapes)),
        )
    logger.info(
        "ConvRot INT8 quantized %d Linear weights (requested rotation size=%d)",
        optimized_count,
        requested_rotation_size,
    )
    return state_dict


def convrot_int8_state_summary(model: nn.Module) -> dict[str, int]:
    """Return small diagnostics used by tests and benchmark logs."""
    layers = 0
    weight_bytes = 0
    scale_bytes = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) and getattr(module, "_convrot_int8_enabled", False):
            layers += 1
            weight_bytes += module.weight.numel() * module.weight.element_size()
            scales = _scale_buffer(module)
            scale_bytes += scales.numel() * scales.element_size()
    return {"layers": layers, "weight_bytes": weight_bytes, "scale_bytes": scale_bytes}
