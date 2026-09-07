# Unsupported ONNX Operators by Hugging Face Repository

This document records operators currently unsupported by `onnx2webnn`, grouped by Hugging Face repository.

`scripts/generate_manifest.py` also reads it when regenerating `manifest.json`: a `## [org/repo](https://huggingface.co/org/repo)` section keeps that repo out of the sweep, and backticked `onnx/<component>*.onnx` references inside the section narrow the exclusion to those components (the rest of the repo stays in).

## [onnx-community/dpt-dinov2-small-kitti](https://huggingface.co/onnx-community/dpt-dinov2-small-kitti)

- ONNX model: `onnx/model*.onnx`
- Unsupported operators:
  - `Resize` (mode=cubic) — bicubic position-embedding interpolation; WebNN resample2d supports only nearest/linear

## [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)

- ONNX model: `onnx/model*.onnx`
- Unsupported operators:
  - `STFT`, `Atan`, `NonZero`, and others — vocoder signal-processing ops; `NonZero` has data-dependent output shapes and cannot be expressed in a static WebNN graph
  - the quantized export additionally uses the com.microsoft fusions `DynamicQuantizeLSTM`, `SkipLayerNormalization`, `FastGelu` and `FusedMatMul`

## [onnx-community/grounding-dino-tiny-ONNX](https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX)

- ONNX model: `onnx/model*.onnx`
- Unsupported operators:
  - `NonZero` — data-dependent output shape, not expressible in a static WebNN graph
  - `GridSample`, `EyeLike` — no WebNN equivalent yet

## [onnx-community/Janus-Pro-1B-ONNX](https://huggingface.co/onnx-community/Janus-Pro-1B-ONNX)

- `onnx/prepare_inputs_embeds*.onnx` — `NonZero` (image-token scatter); data-dependent shape (the other seven exports convert)

## [huggingworld/Qwen2.5-VL-3B-Instruct-ONNX](https://huggingface.co/huggingworld/Qwen2.5-VL-3B-Instruct-ONNX)

- `onnx/vision_encoder*.onnx` — `TopK`/`NonZero` in the window-sorting rotary code (data-dependent) and `MultiHeadAttention` (embed_tokens and the decoder convert)

## [Xenova/musicgen-small](https://huggingface.co/Xenova/musicgen-small)

- `onnx/build_delay_pattern_mask*.onnx` — `NonZero`; data-dependent shape
- `onnx/encodec_decode*.onnx` — pad amounts computed with float division and `Ceil` are not constant-folded; the quantized export also uses `DynamicQuantizeLSTM` (text_encoder and decoder_model_merged convert)

## [onnx-community/sam3-tracker-ONNX](https://huggingface.co/onnx-community/sam3-tracker-ONNX)

- `onnx/vision_encoder*.onnx` — shape propagation mismatch in the windowed attention layers (one operand tracked as 9216 rows instead of 9); open
- `onnx/prompt_encoder_mask_decoder*.onnx` — `Range` with a non-constant limit after the empty-point/box branches

## [Xenova/RTMO-l](https://huggingface.co/Xenova/RTMO-l)

- ONNX model: `onnx/model*.onnx`
- Unsupported operators: `TopK`, `NonMaxSuppression` — detection post-processing with data-dependent output shapes

## [Xenova/movenet-singlepose-thunder](https://huggingface.co/Xenova/movenet-singlepose-thunder)

- ONNX model: `onnx/model*.onnx`
- Shape inference fails in the keypoint post-processing: `Incompatible static dimensions for broadcasting: 17 vs 2` (17 keypoints vs. 2 coordinates); open

## [Xenova/bart-large-mnli](https://huggingface.co/Xenova/bart-large-mnli)

- ONNX model: `onnx/model*.onnx`
- Unsupported operators: `NonZero` (EOS-token pooling); data-dependent shape

## [D4ve-R/wavlm-base-plus-sv](https://huggingface.co/D4ve-R/wavlm-base-plus-sv)

- ONNX model: `onnx/model*.onnx`
- Graph conversion fails on the TDNN layers' N-D-index `Gather` (`gather axis 4 out of bounds for rank 3`); reported to rustnn. Needs a realistic raw-audio length (≥ 400 samples) to get that far

## [BricksDisplay/silero-vad-6.2](https://huggingface.co/BricksDisplay/silero-vad-6.2)

- ONNX model: `onnx/model*.onnx`
- Pinning `sr` inlines the sample-rate `If`, but the branches contain further `If`s gated on tensor shapes that are not folded; inputs use unnamed dims (`input_dim0`, `input_dim1`, `state_dim1`)

## [Xenova/4x_APISR_GRL_GAN_generator-onnx](https://huggingface.co/Xenova/4x_APISR_GRL_GAN_generator-onnx)

- ONNX model: `onnx/model*.onnx`
- Window-partition `Reshape` target is folded to rank 4 where a rank-6 view is expected (`Transpose permutation length 6 must match input rank 1`); open

## [AdamCodd/distilroberta-nsfw-prompt-stable-diffusion](https://huggingface.co/AdamCodd/distilroberta-nsfw-prompt-stable-diffusion)

- Gated repository (HTTP 401 without a token); not evaluated

## [Mozilla/distilvit](https://huggingface.co/Mozilla/distilvit)

- `onnx/decoder_model_merged*.onnx` — the cached-attention `If` branch's `Reshape` shape input (`.../attn/Concat_*_output_0`) is not constant-folded, so its target shape can't be resolved from the branch's `Transpose` output; open (encoder_model converts)

## [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0)

- ONNX model: `onnx/model*.onnx`
- Unsupported operators: the quantized export uses the com.microsoft fusion `DynamicQuantizeLSTM` (four LSTM layers)

## [Xenova/nllb-200-distilled-600M](https://huggingface.co/Xenova/nllb-200-distilled-600M)

- `onnx/decoder_model_merged*.onnx` — the shared output-embedding bias reshape mis-derives its target shape from `decoder_sequence_length` instead of the vocab size, producing an incompatible broadcast (`decoder_sequence_length` vs. 256206); open (encoder_model converts)
