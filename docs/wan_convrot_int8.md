# Experimental WAN ConvRot INT8 training

This branch adds an experimental W8A8 base-weight mode for WAN LoRA training. It rotates eligible frozen block
`Linear` weights with a regular Hadamard transform, stores them as per-output-channel INT8, quantizes activations
per token, and uses `torch._int_mm` for the base forward pass. The trainable LoRA branch remains the standard
Musubi implementation.

The implementation is derived from Ostris' MIT-licensed
[`convrot_quant.py`](https://github.com/ostris/ai-toolkit/blob/0d53e5e1f9db022559f9f9cb8fd4f73b5f10e0c7/toolkit/util/convrot_quant.py).
The [ConvRot paper](https://arxiv.org/abs/2512.03673) evaluates W4A4 inference, not this W8A8 WAN training path.
Treat performance and quality as experimental until they are measured on your workload.

## Requirements and behavior

- Start from an FP16, BF16, or FP32 WAN checkpoint. Pre-quantized FP8/GGUF/Q8 inputs are rejected.
- Native mode requires a CUDA device on which a real `torch._int_mm` startup probe succeeds. This is capability
  based, so CUDA 12.8 and CUDA 13.0 environments are both accepted when the probe passes.
- The native path is strict by default and never silently dequantizes.
- Triton is optional. Without it, activation quantization and the scaling epilogue use Torch operations while the
  matrix multiplication remains native INT8. A performance warning is printed.
- `torch.compile`, `--base_weights` merging, and loading LoRAs during base conversion are not supported in v1.
- ConvRot and `--fp8_base` / `--fp8_scaled` are mutually exclusive.

## Flags

```text
--convrot_int8_base
--convrot_int8_rotation_size 256
--convrot_int8_allow_bf16_fallback
```

The rotation size defaults to `256` and must be a power of four of at least `16`. Each eligible layer uses the
largest power-of-four divisor of its input width that does not exceed the requested value. Unsupported target
shapes remain in the normal training dtype and are reported during loading.

The fallback flag is deliberately explicit. It runs fake-quantized W8A8 activations with a dequantized rotated
weight in the active FP16/BF16 compute dtype. It is useful for correctness checks, but it is **not** an INT8 speed
test. The selected backend is printed and written into the saved LoRA metadata.

## WAN 2.2 example

Use the normal WAN cache steps first. In an existing scaled-FP8 training command, replace:

```text
--fp8_base --fp8_scaled
```

with:

```text
--convrot_int8_base --convrot_int8_rotation_size=256
```

For example, a high-noise I2V run can use:

```bash
accelerate launch --num_cpu_threads_per_process=1 --mixed_precision=fp16 \
  src/musubi_tuner/wan_train_network.py \
  --task=i2v-A14B \
  --dit=/path/to/wan2.2_i2v_high_noise_14B_fp16.safetensors \
  --t5=/path/to/umt5-xxl-enc-bf16.safetensors \
  --vae=/path/to/wan_2.1_vae.safetensors \
  --dataset_config=dataset.toml \
  --min_timestep=900 --max_timestep=1000 \
  --mixed_precision=fp16 --sdpa --gradient_checkpointing \
  --convrot_int8_base --convrot_int8_rotation_size=256 \
  --blocks_to_swap=30 --use_pinned_memory_for_block_swap \
  --block_swap_h2d_only --block_swap_ring_size=2 \
  --network_module=networks.lora_wan --network_dim=64 --network_alpha=64 \
  --optimizer_type=schedulefree.RAdamScheduleFree --learning_rate=3e-5 \
  --output_dir=/path/to/output --output_name=my_lora_high
```

Use `--min_timestep=0 --max_timestep=900` and the low-noise checkpoint for a separate low-noise run. All other
training, caching, optimizer, SDPA, gradient-checkpointing, and block-swap flags can remain unchanged.

## Block swapping and adapter output

Quantized codes replace the existing two-dimensional `Linear.weight` parameter while preserving the `nn.Linear`
class and module path. Classic block swap and H2D-only ring swap therefore stream the INT8 weight through their
normal paths. The much smaller scales and rotation metadata remain resident with the block.

Only the ordinary LoRA parameters are trainable and saved. ConvRot weights, scales, and rotation state are base
model state and are not part of the adapter payload. The resulting LoRA is intended to load on the original
full-precision WAN checkpoint in Musubi or ComfyUI; inference does not require ConvRot conversion.

## RTX 5090 comparison runbook

Before loading the 14B model, run a representative native forward/backward smoke test:

```bash
python convrot_int8_benchmark.py \
  --device=cuda --dtype=fp16 --rows=1024 \
  --in_features=5120 --out_features=5120 --rotation_size=256
```

The command is strict by default, prints the selected backend and environment as JSON, and exits if the native
probe or computation fails. It is a linear microbenchmark, not a substitute for measuring complete training.

Run the same dataset, seed, resolution, frame count, rank/alpha, optimizer, attention backend, checkpointing, and
block-swap settings twice:

1. Baseline: `--fp8_base --fp8_scaled`
2. Candidate: `--convrot_int8_base --convrot_int8_rotation_size=256`

Keep compile disabled. Discard startup and warmup steps, then record median and p95 step time, peak VRAM, startup
conversion time, and final backend from the log. Set `MUSUBI_TUNER_OFFLOADER_DEBUG=1` to collect the existing H2D
ring transfer/stall diagnostics. Test CUDA 12.8 and CUDA 13.0 separately; the log must show that the native probe
passed in each environment used for a speed comparison.

After short matched low- and high-noise runs, render the adapters against the same original full-precision WAN
base using identical prompts, seeds, and inference settings. Compare convergence, motion, identity, detail, and
temporal stability. Do not infer output quality from weight error alone.

Saved metadata includes:

- `ss_convrot_int8_base`
- `ss_convrot_int8_rotation_size`
- `ss_convrot_int8_backend`
- Torch, CUDA, and GPU identity fields
