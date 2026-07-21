import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import init_empty_weights

import musubi_tuner.modules.convrot_int8_kernels as convrot_int8_kernels
from musubi_tuner.modules.custom_offloading_utils import LoRAStreamOffloader, ModelOffloader
from musubi_tuner.networks.lora import LoRAModule
from musubi_tuner.modules.convrot_int8 import (
    CONVROT_INT8_BACKEND_FALLBACK,
    CONVROT_INT8_BACKEND_NATIVE,
    CONVROT_INT8_FORWARD_TORCH,
    CONVROT_INT8_FORWARD_TRITON,
    CONVROT_INT8_ROTATION_BUFFER,
    CONVROT_INT8_SCALE_BUFFER,
    _activation_quant_block_size,
    _legacy_int8_matmul_dequant,
    _native_int8_matmul_dequant,
    apply_convrot_int8_monkey_patch,
    can_quantize_linear_shape,
    convrot_int8_forward_kernel_summary,
    convrot_int8_state_summary,
    effective_rotation_size,
    load_safetensors_with_convrot_int8,
    probe_convrot_int8,
    quantize_int8_rows,
    regular_hadamard,
    reset_convrot_int8_forward_kernel_state,
    resolve_convrot_int8_backend,
    rotate,
    validate_rotation_size,
)


class TinyModel(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 32, bias: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(in_features, out_features, bias=bias)])

    def forward(self, x):
        return self.blocks[0](x)


def make_convrot_state(model: TinyModel, rotation_size: int = 64):
    state = copy.deepcopy(model.state_dict())
    weight = state["blocks.0.weight"]
    codes, scales = quantize_int8_rows(rotate(weight.float(), rotation_size))
    state["blocks.0.weight"] = codes
    state[f"blocks.0.{CONVROT_INT8_SCALE_BUFFER}"] = scales.contiguous().view(torch.uint8)
    state[f"blocks.0.{CONVROT_INT8_ROTATION_BUFFER}"] = torch.tensor(rotation_size, dtype=torch.int64)
    return state


def make_quantized_linear(
    backend: str,
    in_features: int = 64,
    out_features: int = 32,
    bias: bool = True,
) -> nn.Linear:
    model = TinyModel(in_features, out_features, bias=bias).float()
    state = make_convrot_state(model)
    apply_convrot_int8_monkey_patch(model, state, backend)
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False)
    return model.blocks[0]


@pytest.mark.parametrize("size", [4, 16, 64])
def test_regular_hadamard_is_symmetric_orthonormal_and_self_inverse(size):
    matrix = regular_hadamard(size, "cpu", torch.float64)
    identity = torch.eye(size, dtype=torch.float64)
    assert torch.equal(matrix, matrix.t())
    assert torch.allclose(matrix @ matrix.t(), identity, atol=1e-12, rtol=0)
    assert torch.allclose(matrix @ matrix, identity, atol=1e-12, rtol=0)


def test_rotation_size_validation_and_effective_size():
    validate_rotation_size(16)
    validate_rotation_size(256)
    for invalid in (0, 4, 8, 32, 100):
        with pytest.raises(ValueError):
            validate_rotation_size(invalid)

    assert effective_rotation_size(5120, 256) == 256
    assert effective_rotation_size(320, 256) == 64
    assert can_quantize_linear_shape(320, 128, 256)
    assert not can_quantize_linear_shape(96, 127, 256)
    assert _activation_quant_block_size(64) == 64
    assert _activation_quant_block_size(320) == 512
    assert _activation_quant_block_size(5120) == 2048


def test_rotate_roundtrip():
    torch.manual_seed(1)
    value = torch.randn(3, 2, 64, dtype=torch.float64)
    assert torch.allclose(rotate(rotate(value, 64), 64), value, atol=1e-12, rtol=1e-12)


def test_int8_row_quantization_handles_zero_and_extreme_rows():
    value = torch.tensor([[0.0, 0.0, 0.0], [-1e20, 0.0, 1e20]], dtype=torch.float32)
    codes, scales = quantize_int8_rows(value)
    assert codes.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert torch.equal(codes[0], torch.zeros(3, dtype=torch.int8))
    assert scales[0] == 1
    assert codes[1, 0] == -127
    assert codes[1, 2] == 127
    assert torch.isfinite(scales).all()


def test_forward_kernel_summary_starts_uninitialized():
    reset_convrot_int8_forward_kernel_state()
    assert convrot_int8_forward_kernel_summary() == {
        "kernel": "uninitialized",
        "fused_shapes": 0,
        "legacy_shapes": 0,
        "fallback_shapes": 0,
    }


def test_recoverable_triton_error_accepts_compilation_error_subclasses_only():
    compilation_error = type("CompilationError", (Exception,), {"__module__": "triton.compiler.errors"})
    unsupported_construct = type(
        "UnsupportedLanguageConstruct",
        (compilation_error,),
        {"__module__": "triton.compiler.errors"},
    )

    assert convrot_int8_kernels.is_recoverable_triton_error(unsupported_construct("unsupported"))
    assert not convrot_int8_kernels.is_recoverable_triton_error(RuntimeError("CUDA runtime failure"))
    assert not convrot_int8_kernels.is_recoverable_triton_error(torch.OutOfMemoryError("OOM"))


def test_wan_metadata_refresh_replaces_pre_forward_kernel_value(monkeypatch):
    # Upstream WAN T5 still evaluates this default at import time. Avoid trying
    # to initialize CUDA while exercising metadata on CPU-only test hosts.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    import musubi_tuner.wan_train_network as wan_train_network

    trainer = wan_train_network.WanNetworkTrainer()
    metadata = {"ss_convrot_int8_forward_kernel": "uninitialized"}
    monkeypatch.setattr(
        wan_train_network,
        "convrot_int8_forward_kernel_summary",
        lambda: {"kernel": "mixed"},
    )

    trainer.update_metadata_before_save(SimpleNamespace(convrot_int8_base=True), metadata)

    assert metadata["ss_convrot_int8_forward_kernel"] == "mixed"


def test_fallback_forward_backward_and_state_roundtrip():
    torch.manual_seed(2)
    source = TinyModel().float()
    optimized_state = make_convrot_state(source)

    model = TinyModel().float()
    apply_convrot_int8_monkey_patch(model, optimized_state, CONVROT_INT8_BACKEND_FALLBACK)
    info = model.load_state_dict(optimized_state, strict=True, assign=True)
    assert not info.missing_keys
    assert not info.unexpected_keys
    model.requires_grad_(False)

    layer = model.blocks[0]
    assert layer.weight.dtype == torch.int8
    assert not layer.weight.requires_grad
    assert getattr(layer, CONVROT_INT8_SCALE_BUFFER).dtype == torch.uint8
    assert convrot_int8_state_summary(model) == {
        "layers": 1,
        "weight_bytes": 32 * 64,
        "scale_bytes": 32 * 4,
    }

    x = torch.randn(2, 3, 64, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 3, 32)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.weight.grad is None

    saved = model.state_dict()
    reloaded = TinyModel().float()
    apply_convrot_int8_monkey_patch(reloaded, saved, CONVROT_INT8_BACKEND_FALLBACK)
    reloaded.load_state_dict(saved, strict=True, assign=True)
    reloaded.requires_grad_(False)
    assert torch.equal(model(x.detach()), reloaded(x.detach()))


def test_model_dtype_cast_preserves_integer_state():
    source = TinyModel().float()
    optimized_state = make_convrot_state(source)
    model = TinyModel().float()
    apply_convrot_int8_monkey_patch(model, optimized_state, CONVROT_INT8_BACKEND_FALLBACK)
    model.load_state_dict(optimized_state, strict=True, assign=True)
    model.to(dtype=torch.float16)
    layer = model.blocks[0]
    assert layer.weight.dtype == torch.int8
    assert getattr(layer, CONVROT_INT8_SCALE_BUFFER).dtype == torch.uint8
    assert getattr(layer, CONVROT_INT8_ROTATION_BUFFER).dtype == torch.int64
    assert layer.bias.dtype == torch.float16


def test_meta_model_strict_assign_preserves_int8_parameter_and_buffers():
    source = TinyModel().float()
    optimized_state = make_convrot_state(source)
    with init_empty_weights():
        model = TinyModel().float()

    assert model.blocks[0].weight.device.type == "meta"
    apply_convrot_int8_monkey_patch(model, optimized_state, CONVROT_INT8_BACKEND_FALLBACK)
    info = model.load_state_dict(optimized_state, strict=True, assign=True)
    assert not info.missing_keys
    assert not info.unexpected_keys
    layer = model.blocks[0]
    assert layer.weight.device.type == "cpu"
    assert layer.weight.dtype == torch.int8
    assert not layer.weight.requires_grad
    assert getattr(layer, CONVROT_INT8_SCALE_BUFFER).device.type == "cpu"


def test_standard_lora_wraps_patched_linear_and_saves_adapter_state_only():
    torch.manual_seed(4)
    source = TinyModel().float()
    optimized_state = make_convrot_state(source)
    model = TinyModel().float()
    apply_convrot_int8_monkey_patch(model, optimized_state, CONVROT_INT8_BACKEND_FALLBACK)
    model.load_state_dict(optimized_state, strict=True, assign=True)
    model.requires_grad_(False)

    adapter = LoRAModule("lora_unet_blocks_0", model.blocks[0], multiplier=1.0, lora_dim=4, alpha=4)
    adapter.apply_to()
    adapter.train()

    x = torch.randn(2, 64)
    output = model(x)
    output.square().mean().backward()
    assert adapter.lora_down.weight.grad is not None
    assert adapter.lora_up.weight.grad is not None
    assert model.blocks[0].weight.grad is None
    assert set(adapter.state_dict()) == {"alpha", "lora_down.weight", "lora_up.weight"}


def test_cpu_probe_is_strict_unless_fallback_is_explicit():
    result = probe_convrot_int8("cpu", refresh=True)
    assert not result.supported
    with pytest.raises(RuntimeError, match="Native INT8 is required"):
        resolve_convrot_int8_backend("cpu")
    assert resolve_convrot_int8_backend("cpu", allow_bf16_fallback=True) == CONVROT_INT8_BACKEND_FALLBACK


def test_streaming_safetensors_loader_quantizes_only_eligible_targets(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "tiny.safetensors"
    source = TinyModel().float().state_dict()
    source["head.weight"] = torch.randn(7, 64)
    safetensors.save_file(source, checkpoint)

    state = load_safetensors_with_convrot_int8(
        str(checkpoint),
        calc_device="cpu",
        loading_device="cpu",
        weight_dtype=torch.float32,
        requested_rotation_size=64,
        target_keys=["blocks"],
        exclude_keys=None,
    )
    assert state["blocks.0.weight"].dtype == torch.int8
    assert state[f"blocks.0.{CONVROT_INT8_SCALE_BUFFER}"].dtype == torch.uint8
    assert state[f"blocks.0.{CONVROT_INT8_ROTATION_BUFFER}"].item() == 64
    assert state["blocks.0.bias"].dtype == torch.float32
    assert state["head.weight"].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_cuda_forward_backward():
    device = torch.device("cuda")
    probe = probe_convrot_int8(device, refresh=True)
    if not probe.supported:
        pytest.skip(probe.reason)

    torch.manual_seed(3)
    source = TinyModel().to(device=device, dtype=torch.float16)
    optimized_state = make_convrot_state(source, rotation_size=64)
    model = TinyModel().to(device=device, dtype=torch.float16)
    apply_convrot_int8_monkey_patch(model, optimized_state, CONVROT_INT8_BACKEND_NATIVE)
    model.load_state_dict(optimized_state, strict=True, assign=True)
    model.requires_grad_(False)

    x = torch.randn(2, 5, 64, device=device, dtype=torch.float16, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 5, 32)
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def _make_native_cuda_model(device: torch.device, dtype: torch.dtype, bias: bool) -> TinyModel:
    source = TinyModel(bias=bias).to(device=device, dtype=dtype)
    optimized_state = make_convrot_state(source, rotation_size=64)
    model = TinyModel(bias=bias).to(device=device, dtype=dtype)
    apply_convrot_int8_monkey_patch(model, optimized_state, CONVROT_INT8_BACKEND_NATIVE)
    model.load_state_dict(optimized_state, strict=True, assign=True)
    model.requires_grad_(False)
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "dtype,rows,bias",
    [
        (torch.float16, 31, False),
        (torch.float16, 32, True),
        (torch.bfloat16, 33, True),
    ],
)
def test_fused_cuda_matches_legacy_output_and_unchanged_backward(monkeypatch, dtype, rows, bias):
    if not convrot_int8_kernels.HAS_TRITON:
        pytest.skip("Triton is required")

    device = torch.device("cuda")
    probe = probe_convrot_int8(device, refresh=True)
    if not probe.supported:
        pytest.skip(probe.reason)

    torch.manual_seed(31 + rows)
    model = _make_native_cuda_model(device, dtype, bias)
    upstream = torch.randn(rows, 32, device=device, dtype=dtype)

    reset_convrot_int8_forward_kernel_state()
    x_fused = torch.randn(rows, 64, device=device, dtype=dtype, requires_grad=True)
    output_fused = model(x_fused)
    output_fused.backward(upstream)
    fused_grad = x_fused.grad.detach().clone()
    assert convrot_int8_forward_kernel_summary()["kernel"] == CONVROT_INT8_FORWARD_TRITON

    reset_convrot_int8_forward_kernel_state()
    with monkeypatch.context() as patch:
        patch.setattr(convrot_int8_kernels, "HAS_TRITON", False)
        x_legacy = x_fused.detach().clone().requires_grad_(True)
        output_legacy = model(x_legacy)
        output_legacy.backward(upstream)
        legacy_grad = x_legacy.grad.detach().clone()
        assert convrot_int8_forward_kernel_summary()["kernel"] == CONVROT_INT8_FORWARD_TORCH

    torch.testing.assert_close(output_fused, output_legacy, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(fused_grad, legacy_grad, rtol=1e-4, atol=1e-4)
    output_cosine = F.cosine_similarity(output_fused.float().flatten(), output_legacy.float().flatten(), dim=0)
    grad_cosine = F.cosine_similarity(fused_grad.float().flatten(), legacy_grad.float().flatten(), dim=0)
    assert output_cosine >= 0.999999
    assert grad_cosine >= 0.999999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype,bias", [(torch.float16, True), (torch.bfloat16, False)])
def test_fused_cuda_no_grad_zero_and_outlier_inputs_match_legacy(monkeypatch, dtype, bias):
    if not convrot_int8_kernels.HAS_TRITON:
        pytest.skip("Triton is required")

    device = torch.device("cuda")
    probe = probe_convrot_int8(device, refresh=True)
    if not probe.supported:
        pytest.skip(probe.reason)

    torch.manual_seed(59)
    model = _make_native_cuda_model(device, dtype, bias)
    input_value = torch.randn(33, 64, device=device, dtype=dtype)
    input_value[0].zero_()
    input_value[1, ::2] = 1000
    input_value[1, 1::2] = -1000

    reset_convrot_int8_forward_kernel_state()
    with torch.no_grad():
        output_fused = model(input_value)
    assert convrot_int8_forward_kernel_summary()["kernel"] == CONVROT_INT8_FORWARD_TRITON

    reset_convrot_int8_forward_kernel_state()
    with monkeypatch.context() as patch, torch.no_grad():
        patch.setattr(convrot_int8_kernels, "HAS_TRITON", False)
        output_legacy = model(input_value)
    assert convrot_int8_forward_kernel_summary()["kernel"] == CONVROT_INT8_FORWARD_TORCH

    assert torch.isfinite(output_fused).all()
    torch.testing.assert_close(output_fused, output_legacy, rtol=1e-3, atol=1e-3)
    cosine = F.cosine_similarity(output_fused.float().flatten(), output_legacy.float().flatten(), dim=0)
    assert cosine >= 0.999999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_launch_failure_is_cached_as_legacy_fallback(monkeypatch):
    device = torch.device("cuda")
    probe = probe_convrot_int8(device, refresh=True)
    if not probe.supported:
        pytest.skip(probe.reason)

    model = _make_native_cuda_model(device, torch.float16, bias=True)
    x = torch.randn(33, 64, device=device, dtype=torch.float16)
    launch_count = 0

    def fail_fused_launch(*args, **kwargs):
        nonlocal launch_count
        launch_count += 1
        raise NotImplementedError("forced unsupported fused launch")

    reset_convrot_int8_forward_kernel_state()
    monkeypatch.setattr(convrot_int8_kernels, "HAS_TRITON", True)
    monkeypatch.setattr(convrot_int8_kernels, "fused_int8_matmul_dequant", fail_fused_launch)
    first = model(x)
    second = model(x)

    assert launch_count == 1
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert convrot_int8_forward_kernel_summary() == {
        "kernel": CONVROT_INT8_FORWARD_TORCH,
        "fused_shapes": 0,
        "legacy_shapes": 1,
        "fallback_shapes": 1,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("error", [RuntimeError("forced CUDA runtime failure"), torch.OutOfMemoryError("forced OOM")])
def test_fused_cuda_runtime_failures_propagate_without_legacy_retry(monkeypatch, error):
    device = torch.device("cuda")
    probe = probe_convrot_int8(device, refresh=True)
    if not probe.supported:
        pytest.skip(probe.reason)

    model = _make_native_cuda_model(device, torch.float16, bias=True)
    x = torch.randn(33, 64, device=device, dtype=torch.float16)

    def fail_fused_launch(*args, **kwargs):
        raise error

    reset_convrot_int8_forward_kernel_state()
    monkeypatch.setattr(convrot_int8_kernels, "HAS_TRITON", True)
    monkeypatch.setattr(convrot_int8_kernels, "fused_int8_matmul_dequant", fail_fused_launch)
    with pytest.raises(type(error), match=str(error)):
        model(x)
    assert convrot_int8_forward_kernel_summary()["kernel"] == "uninitialized"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_fused_cuda_lora_loss_and_gradients_match_legacy(monkeypatch, use_checkpoint):
    if not convrot_int8_kernels.HAS_TRITON:
        pytest.skip("Triton is required")

    device = torch.device("cuda")
    probe = probe_convrot_int8(device, refresh=True)
    if not probe.supported:
        pytest.skip(probe.reason)

    torch.manual_seed(83)
    source = TinyModel().to(device=device, dtype=torch.float16)
    optimized_state = make_convrot_state(source, rotation_size=64)

    def build_model_and_adapter():
        model = TinyModel().to(device=device, dtype=torch.float16)
        apply_convrot_int8_monkey_patch(model, optimized_state, CONVROT_INT8_BACKEND_NATIVE)
        model.load_state_dict(optimized_state, strict=True, assign=True)
        model.requires_grad_(False)
        adapter = LoRAModule("lora_unet_blocks_0", model.blocks[0], multiplier=1.0, lora_dim=4, alpha=4)
        adapter.to(device=device, dtype=torch.float16)
        adapter.apply_to()
        adapter.train()
        return model, adapter

    fused_model, fused_adapter = build_model_and_adapter()
    legacy_model, legacy_adapter = build_model_and_adapter()
    with torch.no_grad():
        fused_adapter.lora_up.weight.normal_(std=0.02)
    legacy_adapter.load_state_dict(fused_adapter.state_dict(), strict=True)

    input_value = torch.randn(33, 64, device=device, dtype=torch.float16)
    target = torch.randn(33, 32, device=device, dtype=torch.float32)

    def forward(model, value):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(model, value, use_reentrant=False)
        return model(value)

    reset_convrot_int8_forward_kernel_state()
    x_fused = input_value.detach().clone().requires_grad_(True)
    output_fused = forward(fused_model, x_fused)
    loss_fused = F.mse_loss(output_fused.float(), target)
    loss_fused.backward()
    assert convrot_int8_forward_kernel_summary()["kernel"] == CONVROT_INT8_FORWARD_TRITON

    reset_convrot_int8_forward_kernel_state()
    with monkeypatch.context() as patch:
        patch.setattr(convrot_int8_kernels, "HAS_TRITON", False)
        x_legacy = input_value.detach().clone().requires_grad_(True)
        output_legacy = forward(legacy_model, x_legacy)
        loss_legacy = F.mse_loss(output_legacy.float(), target)
        loss_legacy.backward()
        assert convrot_int8_forward_kernel_summary()["kernel"] == CONVROT_INT8_FORWARD_TORCH

    torch.testing.assert_close(output_fused, output_legacy, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(loss_fused, loss_legacy, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x_fused.grad, x_legacy.grad, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(
        fused_adapter.lora_down.weight.grad,
        legacy_adapter.lora_down.weight.grad,
        rtol=1e-3,
        atol=1e-5,
    )
    torch.testing.assert_close(
        fused_adapter.lora_up.weight.grad,
        legacy_adapter.lora_up.weight.grad,
        rtol=1e-3,
        atol=1e-5,
    )
    for fused, legacy in (
        (output_fused, output_legacy),
        (x_fused.grad, x_legacy.grad),
        (fused_adapter.lora_down.weight.grad, legacy_adapter.lora_down.weight.grad),
        (fused_adapter.lora_up.weight.grad, legacy_adapter.lora_up.weight.grad),
    ):
        reference_norm = torch.linalg.vector_norm(legacy.float())
        assert reference_norm > 0
        relative_l2 = torch.linalg.vector_norm(fused.float() - legacy.float()) / reference_norm
        assert relative_l2 <= 1e-3
        cosine = F.cosine_similarity(fused.float().flatten(), legacy.float().flatten(), dim=0)
        assert cosine >= 0.999999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "rows,in_features,out_features,dtype,bias",
    [
        (1031, 336, 520, torch.bfloat16, False),
        (33, 5120, 5120, torch.float16, True),
        (33, 5120, 13824, torch.float16, False),
        (33, 13824, 5120, torch.float16, True),
    ],
)
def test_fused_cuda_multi_tile_and_wan_shapes_match_legacy(rows, in_features, out_features, dtype, bias):
    if not convrot_int8_kernels.HAS_TRITON:
        pytest.skip("Triton is required")

    device = torch.device("cuda")
    probe = probe_convrot_int8(device, refresh=True)
    if not probe.supported:
        pytest.skip(probe.reason)

    torch.manual_seed(rows + in_features + out_features)
    padded_rows = -(-rows // 32) * 32
    activation_codes = torch.randint(-127, 128, (padded_rows, in_features), device=device, dtype=torch.int8)
    activation_scales = torch.rand(padded_rows, device=device, dtype=torch.float32).mul_(1e-3).add_(1e-5)
    weight = torch.randint(-127, 128, (out_features, in_features), device=device, dtype=torch.int8)
    weight_scales = torch.rand(out_features, device=device, dtype=torch.float32).mul_(1e-3).add_(1e-5)
    weight_scales_u8 = weight_scales.contiguous().view(torch.uint8)
    bias_value = torch.randn(out_features, device=device, dtype=dtype) if bias else None

    reset_convrot_int8_forward_kernel_state()
    output_fused = _native_int8_matmul_dequant(
        activation_codes,
        activation_scales,
        weight,
        weight_scales_u8,
        bias_value,
        rows,
        dtype,
    )
    assert convrot_int8_forward_kernel_summary()["kernel"] == CONVROT_INT8_FORWARD_TRITON
    output_legacy = _legacy_int8_matmul_dequant(
        activation_codes,
        activation_scales,
        weight,
        weight_scales_u8,
        bias_value,
        rows,
        dtype,
    )

    torch.testing.assert_close(output_fused, output_legacy, rtol=1e-3, atol=1e-3)
    cosine = F.cosine_similarity(output_fused.float().flatten(), output_legacy.float().flatten(), dim=0)
    assert cosine >= 0.999999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cuda_nonreentrant_checkpoint_backward():
    if not convrot_int8_kernels.HAS_TRITON:
        pytest.skip("Triton is required")

    device = torch.device("cuda")
    probe = probe_convrot_int8(device, refresh=True)
    if not probe.supported:
        pytest.skip(probe.reason)

    model = _make_native_cuda_model(device, torch.float16, bias=True)
    x = torch.randn(33, 64, device=device, dtype=torch.float16, requires_grad=True)
    reset_convrot_int8_forward_kernel_state()
    output = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=False)
    output.float().square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert convrot_int8_forward_kernel_summary()["kernel"] == CONVROT_INT8_FORWARD_TRITON


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_classic_block_swap_keeps_scales_resident_and_swaps_int8_weight():
    device = torch.device("cuda")
    blocks = nn.ModuleList([make_quantized_linear(CONVROT_INT8_BACKEND_NATIVE) for _ in range(2)])
    offloader = ModelOffloader(
        "test",
        blocks,
        num_blocks=2,
        blocks_to_swap=1,
        supports_backward=False,
        device=device,
    )
    offloader.prepare_block_devices_before_forward(blocks)
    assert blocks[0].weight.device == device
    assert blocks[1].weight.device.type == "cpu"
    assert all(getattr(block, CONVROT_INT8_SCALE_BUFFER).device == device for block in blocks)
    assert all(block.weight.dtype == torch.int8 for block in blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_h2d_ring_swap_streams_int8_weight_without_moving_scales():
    device = torch.device("cuda")
    blocks = nn.ModuleList([make_quantized_linear(CONVROT_INT8_BACKEND_NATIVE) for _ in range(4)])
    offloader = LoRAStreamOffloader(
        "test",
        blocks,
        num_blocks=4,
        blocks_to_swap=2,
        supports_backward=True,
        device=device,
        ring_size=1,
        use_pinned_memory=True,
    )
    offloader.prepare_block_devices_before_forward(blocks)
    first, second = offloader.stream_idx
    assert blocks[first].weight.device == device
    assert blocks[second].weight.device.type == "cpu"
    assert all(getattr(block, CONVROT_INT8_SCALE_BUFFER).device == device for block in blocks)

    offloader.wait_for_block(second)
    assert blocks[first].weight.device.type == "cpu"
    assert blocks[second].weight.device == device
    assert all(getattr(block, CONVROT_INT8_SCALE_BUFFER).device == device for block in blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cuda_h2d_ring_forward_backward():
    if not convrot_int8_kernels.HAS_TRITON:
        pytest.skip("Triton is required")

    device = torch.device("cuda")
    probe = probe_convrot_int8(device, refresh=True)
    if not probe.supported:
        pytest.skip(probe.reason)

    blocks = nn.ModuleList([make_quantized_linear(CONVROT_INT8_BACKEND_NATIVE, out_features=64) for _ in range(6)]).to(
        dtype=torch.float16
    )
    reference_blocks = copy.deepcopy(blocks).to(device=device)
    offloader = LoRAStreamOffloader(
        "test",
        blocks,
        num_blocks=6,
        blocks_to_swap=4,
        supports_backward=True,
        device=device,
        ring_size=2,
        use_pinned_memory=True,
    )
    offloader.prepare_block_devices_before_forward(blocks)
    assert offloader.S == 4
    assert offloader.B == 2

    input_value = torch.randn(33, 64, device=device, dtype=torch.float16)
    x_reference = input_value.detach().clone().requires_grad_(True)
    reference_output = x_reference
    for block in reference_blocks:
        reference_output = torch.utils.checkpoint.checkpoint(block, reference_output, use_reentrant=False)
    reference_output.square().mean().backward()

    reset_convrot_int8_forward_kernel_state()
    x_streamed = input_value.detach().clone().requires_grad_(True)
    streamed_output = x_streamed
    for block_idx, block in enumerate(blocks):
        offloader.wait_for_block(block_idx)
        streamed_output = torch.utils.checkpoint.checkpoint(block, streamed_output, use_reentrant=False)
        offloader.submit_move_blocks_forward(blocks, block_idx)
    assert set(offloader.in_slot) == set(offloader.stream_idx[-offloader.B :])
    streamed_output.square().mean().backward()

    assert set(offloader.in_slot) == set(offloader.stream_idx[: offloader.B])
    assert x_streamed.grad is not None
    assert torch.isfinite(x_streamed.grad).all()
    torch.testing.assert_close(streamed_output, reference_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x_streamed.grad, x_reference.grad, rtol=1e-4, atol=1e-4)
    assert convrot_int8_forward_kernel_summary()["kernel"] == CONVROT_INT8_FORWARD_TRITON
