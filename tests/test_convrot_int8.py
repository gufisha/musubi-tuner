import copy

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights

from musubi_tuner.modules.custom_offloading_utils import LoRAStreamOffloader, ModelOffloader
from musubi_tuner.networks.lora import LoRAModule
from musubi_tuner.modules.convrot_int8 import (
    CONVROT_INT8_BACKEND_FALLBACK,
    CONVROT_INT8_BACKEND_NATIVE,
    CONVROT_INT8_ROTATION_BUFFER,
    CONVROT_INT8_SCALE_BUFFER,
    _activation_quant_block_size,
    apply_convrot_int8_monkey_patch,
    can_quantize_linear_shape,
    convrot_int8_state_summary,
    effective_rotation_size,
    load_safetensors_with_convrot_int8,
    probe_convrot_int8,
    quantize_int8_rows,
    regular_hadamard,
    resolve_convrot_int8_backend,
    rotate,
    validate_rotation_size,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(64, 32)])

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


def make_quantized_linear(backend: str) -> nn.Linear:
    model = TinyModel().float()
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
