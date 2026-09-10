# onnx2webnn

ONNX → WebNN lowering crate extracted from [webnn-graph](../webnn-graph). ONNX operators lower
directly to [rustnn](../rustnn) `MLGraphBuilder`; full-graph validation runs via ORT CPU
`build()` (`onnx-runtime` feature). There is no JSON IR and no on-disk graph export — success
means `builder.build()` returns `Ok(MLGraph)`.

Supported ONNX opset range: **1–26** (see `MIN_SUPPORTED_OPSET` / `MAX_SUPPORTED_OPSET` in
`src/onnx/convert.rs`).

## Build

```powershell
cargo build
# or
cargo build --release
```

`make build`, `make test`, `make fmt`, and `make check` are defined in the repo `Makefile`.

## Convert

```powershell
cargo run -- convert --input model.onnx --optimize --override-dim batch_size=1
```

Dynamic ONNX inputs (unresolved symbolic dims kept as WebNN dynamic metadata):

```powershell
cargo run -- convert --input model.onnx `
  --experimental-dynamic-inputs `
  --override-dim batch_size=1 `
  --override-dim sequence_length=1
```

If `model.dims.json` sits beside the ONNX file and no overrides were passed on the CLI, dimension
bindings are loaded from that sidecar (`freeDimensionOverrides` or a flat JSON object).

Merged decoders (optimum's `decoder_model_merged*.onnx`) branch at runtime on `use_cache_branch`.
WebNN has no runtime `If`, so pin the input and convert each branch separately:

```powershell
cargo run -- convert --input decoder_model_merged.onnx --optimize `
  --pin-input use_cache_branch=false `
  --override-dim batch_size=1 --override-dim decoder_sequence_length=4 `
  --override-dim past_decoder_sequence_length=0 ...
```

Pinned inputs become constants, the chosen `If` branch is inlined, and inputs the branch never
reads (e.g. the KV cache in the prefill branch) and zero-size dummy outputs are dropped.

| Flag | Purpose |
|------|---------|
| `--input` | Input `.onnx` path (required) |
| `--optimize` | Constant folding and shape propagation |
| `--override-dim NAME=VALUE` | Bind a symbolic dim (repeatable); unnamed zero dims are addressed as `<input>_dim<axis>` |
| `--override-dims-file` | JSON overrides (`freeDimensionOverrides` or flat object) |
| `--pin-input NAME=VALUE` | Freeze a graph input to `true`/`false`/an integer (repeatable) |
| `--allow-missing-external-data` | Zero-fill external tensors whose data file is absent (weight-stripped skeleton models) |
| `--experimental-dynamic-inputs` | Preserve unresolved symbolic dims as dynamic metadata |
| `--debug` | Verbose conversion logging (global) |

On success the CLI prints `✓ ORT graph build succeeded for …`.

## Model sweep

`tests/models/manifest.json` lists the transformers.js exports the converter is expected to handle,
with their dimension overrides and pinned inputs; `tests/model_skeletons.rs` converts each entry and
builds it in ORT. No weights are downloaded: the test reads each export from the Hugging Face Hub with
HTTP range requests, keeps the graph and small constants, and points every large initializer at a file
that does not exist, which the converter zero-fills. A 1.4 GB export becomes a ~0.2 MB skeleton for
~10 MB of traffic. Skeletons are kept in `target/model-skeletons` (or `O2W_SKELETON_CACHE`), about
40 MB for the whole manifest, and CI caches that directory keyed on the manifest and scanner source.

`O2W_MODELS` selects the source: `hub` (the default when `CI` is set), `dir=<path>` for full local
downloads laid out as `<org>--<repo>/onnx/<file>.onnx`, or `strip=<path>` to run local files through
the skeleton scanner. Unset outside CI, the sweep is skipped. `O2W_MODEL_FETCH_JOBS` (default 8) sets
how many skeletons are fetched at once, `O2W_MODEL_TEST_JOBS` (default 4) how many convert at once,
and `O2W_MODEL_TEST_SKIP_HEAVY` skips the entries that need more than 10 GB of RAM.

```powershell
$env:O2W_MODELS = "dir=..\transformers_js_experiments\models"; cargo test --release --test model_skeletons
```

Library API:

```rust
use onnx2webnn::{convert_onnx, ConvertOptions};

let graph = convert_onnx("model.onnx", ConvertOptions::default())?;
```

## Layout

| Path | Purpose |
|------|---------|
| `src/onnx/convert.rs` | ONNX load, optional folding, lowering, ORT `build()` |
| `src/onnx/builder.rs` | `OnnxBuilder` — operand map and `MLGraphBuilder` bridge |
| `src/onnx/builder_helpers.rs` | Shared lowering helpers |
| `src/onnx/shape_inference.rs` | Static shape/type propagation |
| `src/onnx/constant_folding.rs` | Constant folding driver (with `--optimize`) |
| `src/onnx/constant_folding/evaluators/` | Per-op fold evaluators |
| `src/onnx/ops/` | ONNX op handlers (activation, conv, pool, reshape, …) |
| `src/protos.rs` | ONNX protobuf types |
| `src/debug.rs` | Debug logging toggle |

## Dependencies

- **rustnn** (`../rustnn`, `onnx-runtime`) — `MLGraphBuilder`, shape inference, ORT `build()` validation
- **webnn-onnx-utils** — ONNX protos, op names, data types

## Related

- [webnn-graph](../webnn-graph) — DSL parser, validator, JS/HTML emit (source of the extracted lowering code)
