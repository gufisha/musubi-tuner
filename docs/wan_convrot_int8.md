# Experimental WAN ConvRot INT8 training

This branch adds an experimental W8A8 base-weight mode for WAN LoRA training. It rotates eligible frozen block
`Linear` weights with a regular Hadamard transform, stores them as per-output-channel INT8, quantizes activations
per token, and prefers a Triton INT8 GEMM that fuses per-row/per-channel scaling and bias into the FP16/BF16
output store. The existing `torch._int_mm` plus scaling epilogue remains the compatibility path. The trainable
LoRA branch remains the standard Musubi implementation.

The implementation is derived from Ostris' MIT-licensed
[`convrot_quant.py`](https://github.com/ostris/ai-toolkit/blob/0d53e5e1f9db022559f9f9cb8fd4f73b5f10e0c7/toolkit/util/convrot_quant.py).
The fused GEMM is adapted under Apache-2.0 from
[`kohya-ss/musubi-tuner#1008`](https://github.com/kohya-ss/musubi-tuner/pull/1008), which vendors the kernel from
Comfy Kitchen. Rotation, activation quantization, weight conversion, and backward remain the WAN implementation.
The [ConvRot paper](https://arxiv.org/abs/2512.03673) evaluates W4A4 inference, not this W8A8 WAN training path.
Treat performance and quality as experimental until they are measured on your workload.

## Requirements and behavior

- Start from an FP16, BF16, or FP32 WAN checkpoint. Pre-quantized FP8/GGUF/Q8 inputs are rejected.
- Native mode requires a CUDA device on which a real `torch._int_mm` startup probe succeeds. This is capability
  based, so CUDA 12.8 and CUDA 13.0 environments are both accepted when the probe passes.
- The native path is strict by default and never silently dequantizes.
- Triton is optional. When available, the preferred forward keeps INT32 accumulation inside the GEMM and writes
  the scaled FP16/BF16 result directly. Recoverable import, compile, or launch-configuration failures are cached per
  device/dtype/shape and that shape falls back to the previous native `torch._int_mm` path with a warning. CUDA
  runtime failures propagate instead of retrying on a potentially unhealthy device context. Without Triton,
  activation quantization and the scaling epilogue use Torch operations while matrix multiplication remains native
  INT8.
- Forward fusion does not change the ConvRot codes/scales or the FP16/BF16 input-gradient backward. It does not use
  the optional INT8-gradient mode from PR #1008.
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

The command is strict by default, prints the selected backend, the actual forward kernel, and the environment as
JSON, and exits if the native probe or computation fails. A fused performance result is valid only when
`forward_kernel.kernel` is `triton_fused` with zero fallback shapes. It is a linear microbenchmark, not a substitute
for measuring complete training.

Compare the candidate against ConvRot commit `0ed8cda` in `A-B-B-A` order. Each run must use the same cached
dataset and exactly 40 steps/four epochs. Keep seed `42`, FP16, xFormers, gradient checkpointing, rank/alpha
`64/64`, block swap `32`, H2D-only ring `2`, pinned memory, and compile disabled. Discard steps 1-10, then record
median and p95 step time, peak VRAM, losses, startup conversion time, GPU clocks/runtime conditions, and the final
forward-kernel diagnostic. Any candidate fallback makes that performance run invalid.

Before the end-to-end run, use `--no_backward` with the benchmark for `5120->5120`, `5120->13824`,
`13824->5120`, and `4096->5120`, including `--rows=14040`. The weighted median candidate forward must improve by
at least 15%. In both candidate training runs, median step time must improve by at least 5%, p95 and VRAM must not
regress, all losses must remain finite, median loss must stay within 1%, and the final LoRA-weight cosine versus the
matched baseline must be at least `0.999`.

After short matched low- and high-noise runs, render the adapters against the same original full-precision WAN
base using identical prompts, seeds, and inference settings. Compare convergence, motion, identity, detail, and
temporal stability. Do not infer output quality from weight error alone.

Saved metadata includes:

- `ss_convrot_int8_base`
- `ss_convrot_int8_rotation_size`
- `ss_convrot_int8_backend`
- `ss_convrot_int8_forward_kernel`
- Torch, CUDA, and GPU identity fields
