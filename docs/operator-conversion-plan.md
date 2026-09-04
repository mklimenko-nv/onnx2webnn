# ONNX → WebNN Operator Conversion Plan

Phased plan for **onnx2webnn** to convert every **ai.onnx** operator at its **latest schema
version** within the current ONNX standard. Baseline: **opset 26** (ONNX package 1.21.0,
`onnx.defs.onnx_opset_version()`).

Each stage below is a **conversion category**, not a calendar milestone. Work within a stage can
proceed in parallel once opset-26 schema audits and the per-op workflow (see [Implementation
workflow](#implementation-workflow)) are in place.

---

## Goals

1. Accept models declaring **ai.onnx opset 26** (raise `MIN/MAX_SUPPORTED_OPSET` from 11–18).
2. Convert every ONNX operator that has a WebNN target — directly, via decomposition, or via
   convert-time constant folding.
3. Reject or pre-fold operators with no WebNN equivalent, with explicit error messages.
4. Track progress against a regenerated opset inventory (`webnn_exporter_supported` column).

## Non-goals

- Supporting multiple ONNX opsets simultaneously (target opset 26 only after completion).
- Executing converted graphs at runtime (conversion validates via rustnn ORT `build()` only).
- Implementing ONNX control flow (`If`, `Loop`, `Scan`) as WebNN graphs.

---

## Coverage baseline (opset 26)

| Category | Count | Share |
|----------|------:|------:|
| Total `ai.onnx` operators at opset 26 | 198 | 100% |
| **Stage 1 — Direct mappable** | ~95 | ~48% |
| **Stage 2 — Complex mappable** | ~19 | ~10% |
| **Stage 3 — Impossible to map** | ~84 | ~42% |

| Implementation status (today) | Count |
|-------------------------------|------:|
| Handlers in `src/onnx/ops/*.rs` | 59 |
| Stage 1 backlog (direct, not yet exported) | ~36 |
| Stage 2 backlog (complex) | ~19 |

Regenerate counts after each handler lands:

```powershell
.\.venv\Scripts\python.exe scripts\generate_onnx_opsets.py --min 26 --max 26 -o docs\onnx-opsets
.\.venv\Scripts\python.exe -c "import csv; r=list(csv.DictReader(open('docs/onnx-opsets/opset-26.csv'))); print(sum(1 for x in r if x['webnn_exporter_supported']=='yes'))"
```

`generate_onnx_opsets.py` defaults to `../webnn-graph/docs/onnx-opsets/`; pass `-o docs\onnx-opsets` to keep
inventory CSVs in this repo.

**Related artifacts**

| Artifact | Path |
|----------|------|
| Operator inventory (regenerate for opset 26) | `docs/onnx-opsets/opset-26.csv` (or `../webnn-graph/docs/onnx-opsets/`) |
| Exporter manifest | `scripts/webnn_onnx_ops.py` |
| Op handlers (Rust) | `src/onnx/ops/*.rs`, registry in `src/onnx/ops/mod.rs` |
| Opset gate | `src/onnx/convert.rs` (`MIN/MAX_SUPPORTED_OPSET`) |
| ONNX ↔ WebNN name map | `webnn-onnx-utils/src/operation_names.rs` |
| Per-op conversion tests | `tests/onnx_ops/{category}/{op}.rs` (auto-generated) |
| Test generator | `scripts/generate_rust_op_conversion_tests.py` |
| Model audit CLI | `scripts/onnx_ops_to_csv.py --check-webnn` |
| Opset upgrade (ad hoc) | `scripts/upgrade_onnx_opset.py` |

**Python scripts (`scripts/`)**

| Script | Role |
|--------|------|
| `webnn_onnx_ops.py` | Supported-op manifest; keep in sync with `src/onnx/ops/*.rs` |
| `generate_rust_op_conversion_tests.py` | Regenerate `tests/onnx_ops/` and `tests/onnx_op_tests.rs` |
| `onnx_fixture_builders.py`, `onnx_test_builders.py`, `rust_model_emitter.py` | Libraries used by the test generator |
| `onnx_ops_to_csv.py` | Audit a model's operators and WebNN support |
| `upgrade_onnx_opset.py` | Upgrade a model to a target `ai.onnx` opset |
| `generate_onnx_opsets.py` | Per-opset operator inventory CSVs (`webnn_exporter_supported` column) |

Install deps from repo root: `pip install -r requirements.txt` (needs `onnx`, `numpy`).

---

## Three conversion stages

```
ONNX model (opset 26, latest schema per op)
        │
        ▼
┌───────────────────────┐
│ Opset gate            │  convert.rs → opset 26 only
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 1: Direct       │  1 ONNX node → 1 WebNN op (± trivial attrs)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 2: Complex      │  fold, decompose, or multi-node subgraph
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 3: Impossible   │  reject (or fold-only for metadata ops)
└───────────────────────┘
```

### Stage 1 — Direct mappable operations

ONNX operators that map to a **single** [WebNN `MLGraphBuilder`
method](https://www.w3.org/TR/webnn/#operators) with straightforward attribute and I/O translation.
No multi-node decomposition required; optional schema-version audit for ops whose active schema
changed at opset 22/25/26.

#### 1a. Implemented today (59)

| Group | Operators | Rust module |
|-------|-----------|-------------|
| Linear | MatMul, Gemm | `matmul.rs` |
| Convolution | Conv, ConvTranspose | `conv.rs` |
| Pooling | MaxPool, AveragePool, GlobalMaxPool, GlobalAveragePool | `pool.rs` |
| Elementwise binary | Add, Sub, Mul, Div, Pow, Min, Max | `elementwise.rs` |
| Comparison | Greater, Less, Equal, GreaterOrEqual, LessOrEqual | `comparison.rs` |
| Conditional | Where | `conditional.rs` |
| Normalization | LayerNormalization, Softmax | `normalization.rs` |
| Shape | Reshape, Transpose, Concat, Split, Unsqueeze, Squeeze, Tile, Expand, Flatten | `reshape.rs` |
| Conversion | Cast, Constant | `conversion.rs` |
| Utility | Shape, Gather, Slice, ConstantOfShape, Range, Trilu, OneHot | `utility.rs` |
| Reduction | ReduceMean, ReduceSum, ReduceMax, ReduceMin | `reduction.rs` |
| Activation / unary | Relu, Gelu, Tanh, Sigmoid, Sqrt, Exp, Log, Abs, Neg, Erf, Cos, Sin, Identity | `activation.rs` |
| Scatter | ScatterND | `scatter.rs` |
| Pad | Pad | `pad.rs` |

#### 1b. Direct backlog — not yet exported (~36)

Implement as single-op handlers. Suggested Rust home in parentheses.

| Group | ONNX → WebNN | Module |
|-------|--------------|--------|
| Normalization | BatchNormalization → `batchNormalization()` | `normalization.rs` |
| Normalization | InstanceNormalization → `instanceNormalization()` | `normalization.rs` |
| Gather | GatherND → `gatherND()` | `utility.rs` |
| Gather | GatherElements → `gatherElements()` | `utility.rs` |
| Scatter | ScatterElements → `scatterElements()` | `scatter.rs` |
| Resample | Resize → `resample2d()` | new `resize.rs` |
| Activation | Elu → `elu()`, LeakyRelu → `leakyRelu()`, PRelu → `prelu()` | `activation.rs` |
| Activation | HardSigmoid → `hardSigmoid()`, HardSwish → `hardSwish()` | `activation.rs` |
| Activation | Softplus → `softplus()`, Softsign → `softsign()` | `activation.rs` |
| Unary math | Ceil → `ceil()`, Floor → `floor()`, Reciprocal → `reciprocal()` | `activation.rs` |
| Unary math | Sign → `sign()`, Tan → `tan()`, Round → `roundEven()` | `activation.rs` |
| Clamp | Clip → `clamp()` | `activation.rs` |
| Logical | Not → `logicalNot()`, And → `logicalAnd()`, Or → `logicalOr()`, Xor → `logicalXor()` | `comparison.rs` |
| Predicate | IsNaN → `isNaN()`, IsInf → `isInfinite()` | `comparison.rs` |
| Reduction | ArgMin → `argMin()`, ArgMax → `argMax()` | `reduction.rs` |
| Reduction | ReduceL1 → `reduceL1()`, ReduceL2 → `reduceL2()` | `reduction.rs` |
| Reduction | ReduceLogSum → `reduceLogSum()`, ReduceLogSumExp → `reduceLogSumExp()` | `reduction.rs` |
| Reduction | ReduceProd → `reduceProduct()`, ReduceSumSquare → `reduceSumSquare()` | `reduction.rs` |
| Reduction | CumSum → `cumulativeSum()` | `reduction.rs` |
| Pool | LpPool → `l2Pool2d()`, GlobalLpPool → `globalAveragePool` + L2 (verify spec) | `pool.rs` |
| Sequence | ReverseSequence → `reverse()` | `utility.rs` |

**Stage 1 exit criteria:** ~95 operators with `webnn_exporter_supported=yes`; all direct-backlog
fixtures convert; Birds EfficientNet and a small transformer ONNX convert without unsupported-op
errors for Stage 1 ops.

**Suggested implementation order (by model impact):**

1. BatchNormalization, InstanceNormalization, GatherND, GatherElements, ScatterElements, Resize
2. Remaining activations and unary math (Elu, LeakyRelu, PRelu, Clip, Ceil, Floor, …)
3. Reduction family completion (ArgMin/Max, ReduceL1/L2/…, CumSum)
4. Logical ops (Not, And, Or, Xor, IsNaN, IsInf)
5. LpPool, ReverseSequence

---

### Stage 2 — Complex mappable operations

Operators **can** be lowered to WebNN, but conversion requires one or more of:

- **Convert-time constant folding** (no runtime graph node)
- **Multi-node decomposition** into Stage 1 ops
- **Non-trivial attribute / mode mapping** (document limitations and reject unsupported modes early)
- **Schema-version-specific inputs** (e.g. opset 25 `pads` tensor on Pad)

#### 2a. Convert-time fold only (no runtime WebNN op)

These appear in graphs but become constants or metadata during `--optimize` / conversion — not
`MLGraphBuilder` nodes at inference time.

| Operator | Strategy | Notes |
|----------|----------|-------|
| Constant | Inline tensor data into graph const pool | Already handled; audit opset 25 schema |
| ConstantOfShape | Fold when shape is static | Audit opset 25 `value` attribute |
| Range | Fold when start/limit/delta are constant | |
| Shape | Fold when input shape is known | |
| Size | Fold to scalar product of static dimensions | Reject if shape is dynamic |

#### 2b. Decompose into Stage 1 subgraphs

| Operator | Decomposition sketch | Priority |
|----------|---------------------|----------|
| CastLike | Type-infer `to` from second input → `Cast` | P1 |
| Mean | `ReduceMean` over all axes (or specified) | P2 |
| Sum | `ReduceSum` over all axes (or specified) | P2 |
| LogSoftmax | `Softmax` + `Log` | P2 |
| Hardmax | `Softmax` + one-hot style mask (or reject axis edge cases) | P3 |
| Swish | `Sigmoid` × input (`Mul`) | P2 |
| Celu | `Elu` variant with alpha (after Elu lands) | P3 |
| Selu | scale × ELU-like piecewise | P3 |
| Mish | `Softplus` + `Tanh` + `Mul` (after Softplus lands) | P3 |
| ThresholdedRelu | `Relu` + comparison + `Where` | P3 |
| GroupNormalization | Mean/var over group dims + `Sub`/`Div`/`Mul`/`Add` | P1 — common in vision |
| RMSNormalization | L2 norm over axes + scale (new in opset 23) | P1 — common in LLMs |
| SimplifiedLayerNormalization | Alias of RMSNormalization | Implemented — standard/exporter fusion |
| SkipSimplifiedLayerNormalization | Residual/bias add + RMSNormalization | Implemented — com.microsoft fusion |
| RotaryEmbedding | Gather caches + rotate pairs + concatenate | Implemented — standard and com.microsoft |
| MatMulNBits | uint4/uint8 reinterpret + dequantizeLinear + transpose + matmul | Implemented — bits=4/8; reject g_idx |
| MatMulInteger | centered float `matmul` + `int32` cast (mirrors ConvInteger) | Implemented — scalar or 1-D zero points |
| Einsum | reduceSum + transpose/reshape + batched `matmul` | Implemented — ≤2 inputs; reject ellipsis/diagonals |
| MatMulBnb4 | packed uint8 constant, nibbles unpacked with int32 div/sub, codebook `gather`, per-block absmax `mul`, `matmul` | Implemented; weight stays 4-bit in the graph, dense fallback when N*K is not a whole number of blocks |
| GroupQueryAttention | reshape/transpose + KV-cache concat + scaled matmul + static causal mask + softmax | Implemented — unpadded batch assumption; reject packed QKV/do_rotary/softcap/local window |
| MoE | dense expert evaluation (batched matmul) + iterative top-k mask + masked softmax blend | Implemented — fused interleaved swiglu/relu/gelu/sigmoid; reject fc3/sparse mixer; num_experts/k× FLOP overhead |
| QMoE | uint4/uint8 expert constants with blockwise `dequantizeLinear`, then the MoE lowering | Implemented; same restrictions as MoE, dense fallback when `in` is not a whole number of blocks |
| GatherBlockQuantized | `gather` packed rows, scales and zero points, unpack nibbles, `dequantizeLinear` on the gathered slice | Implemented; 2-D tables along axis 0 (uint8, INT4, UINT4), other layouts dequantize at conversion time |
| SplitToSequence + SequenceAt | per-element slices registered as pseudo-operands, SequenceAt aliases by constant index | Implemented — torch split() export pattern; sequence graph outputs rejected |
| If (constant condition) | branch inlined into the outer graph before conversion; a boolean graph input gate (optimum's `use_cache_branch`) becomes constant via `--pin-input` | Implemented — data-dependent conditions remain unsupported |
| LSTM/GRU (direction=bidirectional/reverse) | WebNN native direction support + [dirs,...] bias split and Y layout | Implemented — LSTM initial_h/initial_c wired; sequence_lens/peephole rejected |
| CumProd | `cumulativeSum` on log + `Exp`, or iterative multiply subgraph | P3 — new in opset 26 |

#### 2c. Advanced single-entry WebNN ops (high attribute complexity)

Treat as Stage 2 because spec coverage and testing cost exceed typical Stage 1 ops.

| Operator | WebNN target | Key risks |
|----------|--------------|-----------|
| GRU | `gru()` / `gruCell()` | Layout, direction, hidden size, optional inputs |
| LSTM | `lstm()` / `lstmCell()` | Same as GRU; bidirectional |
| RNN | Partial via GRU/LSTM or reject | No dedicated WebNN `rnn()` |
| QuantizeLinear | `quantizeLinear()` | Scale/zero-point types, axis |
| DequantizeLinear | `dequantizeLinear()` | Pair with quant models |
| Pad | `pad()` | Opset 25: `pads` as input tensor; verify `pad.rs` |
| Resize | `resample2d()` | Coordinate transformation modes, cubic exclusion |
| Gemm | `gemm()` | Already exported; audit transA/transB, alpha/beta at opset 13+ |
| Conv / Pool | `conv2d()` / pools | Audit opset 22 dilations, auto_pad, ceil_mode |

**Stage 2 exit criteria:** All fold/decompose paths documented in lowering notes; GroupNormalization
and RMSNormalization decompositions tested on at least one real model; GRU/LSTM/quant handlers
pass unit tests with documented limitations.

---

### Stage 3 — Impossible to map operations

No WebNN `MLGraphBuilder` equivalent exists. **Do not** add graph handlers. Strategy: **reject**
at conversion with a clear message, unless the op can be eliminated earlier (optimizer, training
vs inference export).

#### 3a. Control flow and graph structure

| Operators | Reason |
|-----------|--------|
| If, Loop, Scan | WebNN has no conditional or recurrent graph execution |

#### 3b. Sequence and optional types

| Operators | Reason |
|-----------|--------|
| SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert, SequenceLength, SequenceMap, SplitToSequence, ConcatFromSequence | WebNN tensors only; no sequence type |
| Optional, OptionalGetElement, OptionalHasElement | No optional type in WebNN |

#### 3c. Strings and NLP utilities

| Operators | Reason |
|-----------|--------|
| StringConcat, StringSplit, StringNormalizer, RegexFullMatch, TfIdfVectorizer | No string tensors in WebNN |

#### 3d. Random and stochastic

| Operators | Reason |
|-----------|--------|
| Bernoulli, Multinomial, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike | Non-deterministic; no WebNN RNG ops |

#### 3e. Training, loss, and dropout

| Operators | Reason |
|-----------|--------|
| Dropout, NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss | Training-only; inference graphs should fold or omit |

#### 3f. Detection, vision, and spatial extras

| Operators | Reason |
|-----------|--------|
| NonMaxSuppression, RoiAlign, DeformConv, GridSample, ImageDecoder, MaxRoiPool, MaxUnpool, AffineGrid, CenterCropPad, Col2Im, DepthToSpace, SpaceToDepth | No WebNN ops; post-processing or custom CUDA territory |

#### 3g. Signal processing and windows

| Operators | Reason |
|-----------|--------|
| DFT, STFT, MelWeightMatrix, BlackmanWindow, HammingWindow, HannWindow | No WebNN signal ops |

#### 3h. Quantized conv / matmul (integer)

| Operators | Reason |
|-----------|--------|
| QLinearConv, QLinearMatMul | Fused quantized ops not in WebNN; use explicit float + quantization decomposition. `MatMulInteger` is now implemented via centered float matmul |

`ConvInteger` is supported through centered float `conv2d` followed by an `int32` cast,
`DynamicQuantizeLinear` is decomposed into reductions, scale/zero-point arithmetic, and WebNN
`quantizeLinear`, and `MatMulNBits` (bits=4) follows ORT's WebNN EP path:
`dequantizeLinear` → reshape → transpose → `matmul` (+ optional bias).

#### 3i. Bitwise and type reinterpretation

| Operators | Reason |
|-----------|--------|
| BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, BitShift, **BitCast** (new opset 26) | No bitwise ops in WebNN |

#### 3j. Linear algebra and einsum

| Operators | Reason |
|-----------|--------|
| Det | No determinant in WebNN. `Einsum` is now implemented for one/two-input equations without ellipsis or repeated per-term labels (lowered to reduceSum/transpose/reshape/matmul) |

#### 3k. Indexing and sorting extras

| Operators | Reason |
|-----------|--------|
| TopK, Unique, Compress, NonZero, OneHot, EyeLike | No WebNN equivalents |
| **TensorScatter** (opset 24) | No WebNN op; use ScatterND/ScatterElements subset only |

#### 3l. Attention and transformer extras (recent ONNX)

| Operators | Reason |
|-----------|--------|
| **Attention** (opset 24) | Composite op; no single WebNN op — could be Stage 2 decomposition in future, but out of scope until spec stabilizes |
| **MoE** (com.microsoft) | Requires TopK expert routing and efficient sparse dispatch, neither available in WebNN |

#### 3m. Miscellaneous

| Operators | Reason |
|-----------|--------|
| Acos, Acosh, Asin, Asinh, Atan, Atanh, Cosh, Sinh | No inverse/hyperbolic unary in WebNN |
| LRN, LpNormalization, MeanVarianceNormalization | Legacy normalization; decompose or reject |
| Mod, Shrink | No direct WebNN op |
| Mod, Mish (as dedicated op) | Prefer Stage 2 decomposition via Swish/Mish subgraph if needed |

**Stage 3 handling**

| Strategy | When | Example |
|----------|------|---------|
| Reject | No WebNN op and not foldable | If, Loop, Attention |
| Pre-fold / strip | Training artifact in inference export | Dropout → Identity |
| Document workaround | Rare; manual graph rewrite | Replace TopK with external post-process |

---

## Opset 26 schema revisions

Operators whose **active schema version** changed at or after opset 21 need a fixture pass even when
already exported. Priority: schema v25 and v26.

| Operator | Active schema | Exporter today | Action |
|----------|--------------:|----------------|--------|
| Cast | 25 | yes | Audit `to` / saturate rules |
| Constant | 25 | yes | Verify sparse / value attributes |
| ConstantOfShape | 25 | yes | Verify optional `value` |
| Flatten | 25 | yes | Verify `axis` default |
| Identity | 25 | yes | Smoke test |
| Reshape | 25 | yes | Verify `allowzero` |
| Shape | 25 | yes | Constant-fold path |
| Squeeze | 25 | yes | Verify axes input |
| Transpose | 25 | yes | Verify optional `perm` |
| Unsqueeze | 25 | yes | Verify axes input |
| Pad | 25 | yes | Confirm `pads` tensor input |
| If, Loop, Scan | 25 | no | Reject |
| CastLike | 25 | no | Stage 2 decompose |
| QuantizeLinear, DequantizeLinear | 25 | no | Stage 2 |
| Conv, ConvTranspose, MaxPool, AveragePool, … | 22 | partial | Audit dilations, auto_pad |
| **BitCast** | 26 | no | Stage 3 reject |
| **CumProd** | 26 | no | Stage 2 decompose |

Regenerate per-op Rust conversion tests (fixture opset 26, converter gate opset 18 today):

```powershell
.\.venv\Scripts\python.exe scripts\generate_rust_op_conversion_tests.py --fixture-opset 26 --test-opset 18
cargo test --test onnx_op_tests
```

---

## Implementation workflow

Use this checklist for **every** operator added in Stages 1–2:

- [ ] **Audit real models** — `scripts/onnx_ops_to_csv.py model.onnx --check-webnn`
- [ ] **Read ONNX schema** — `onnx.defs.get_schema(op, 26)` or opset-26 sidecar fixture
- [ ] **Read WebNN spec** — method signature, options, `opSupportLimits`
- [ ] **Implement handler** — `supports()` + `convert()` in `src/onnx/ops/*.rs`
- [ ] **Shape inference** — update `src/onnx/shape_inference.rs` if needed
- [ ] **Register** — add to `OpRegistry` in `src/onnx/ops/mod.rs`
- [ ] **Sync manifest** — add to `scripts/webnn_onnx_ops.py`
- [ ] **Update name map** — `webnn-onnx-utils/src/operation_names.rs` if new mapping
- [ ] **Regenerate inventory** — `scripts/generate_onnx_opsets.py --min 26 --max 26 -o docs/onnx-opsets`
- [ ] **Regenerate tests** — `scripts/generate_rust_op_conversion_tests.py --fixture-opset 26 --test-opset 18`
- [ ] **Unit tests** — handler tests; `cargo test --test onnx_op_tests {op_snake}::opset26`
- [ ] **Convert smoke test** — `cargo run -- convert --input model.onnx --optimize` on a real model using the op

---

## Rollout phases

Cross-cutting milestones that span the three stages:

### Phase 0 — Foundation

| Task | Files |
|------|-------|
| Regenerate opset-26 inventory | `scripts/generate_onnx_opsets.py` → `docs/onnx-opsets/` |
| Set opset gate to 26 only | `src/onnx/convert.rs`, `scripts/webnn_onnx_ops.py` |
| Opset 25/26 schema audit for 59 implemented ops | handlers + fixtures |
| Sync `webnn_onnx_ops.py` with Rust registry | `scripts/webnn_onnx_ops.py` |

**Exit:** Converter accepts opset 26, rejects 18; 59 ops pass v26 fixture conversion.

### Phase 1 — Stage 1 high-impact gaps

BatchNormalization, InstanceNormalization, GatherND, GatherElements, ScatterElements, Resize, Pad
v25 verification.

**Exit:** EfficientNet-class vision models convert; inventory ≥ 66 supported ops.

### Phase 2 — Stage 1 activations, unary, logical, reductions

Complete Stage 1 backlog table in [§1b](#1b-direct-backlog--not-yet-exported-36).

**Exit:** ~95 supported ops; Stage 1 complete.

### Phase 3 — Stage 2 decomposition and advanced ops

GroupNormalization, RMSNormalization, CastLike, Mean/Sum/LogSoftmax, Swish; then GRU/LSTM/quant.

**Exit:** Documented decomposition paths; optional advanced coverage for seq2seq and int8.

### Phase 4 — Documentation and CI

- Maintain supported/reject lists in lowering docs
- CI: `scripts/onnx_ops_to_csv.py model.onnx --check-webnn` on checked-in ONNX models
- Optional: generate `webnn_onnx_ops.py` from Rust `supports()` lists

---

## Progress tracking

| Milestone | Ops exported | Opset gate | Notes |
|-----------|-------------:|------------|-------|
| **Today** | 59 | 11–18 | ~30% of opset 26 |
| Phase 0 done | 59 | 26 only | v25/v26 schema audit |
| Phase 1 done | ~66 | 26 only | Vision models |
| Phase 2 done | ~95 | 26 only | Stage 1 complete |
| Phase 3 done | ~95+ | 26 only | Stage 2 decompositions |
| Stage 3 | — | 26 only | 84 ops permanently rejected |

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Schema v26 differs from v18 for same op name | Mandatory opset-26 fixture per op; read `schema_since_version` |
| WebNN `resample2d` ≠ ONNX `Resize` | Document mode/coordinate mapping; reject unsupported modes |
| `webnn_onnx_ops.py` drifts from Rust | Regenerate CSV after every op; optional codegen from `supports()` |
| GroupNormalization / RMSNormalization in new models | Stage 2 decomposition before reject |
| Attention / RotaryEmbedding in LLM exports | Stage 3 reject until decomposition designed |
| Dynamic shapes | `--optimize` constant folding; static shapes at runtime |
| MLGraphBuilder migration | Align handlers with [mlgraphbuilder-migration-plan.md](./mlgraphbuilder-migration-plan.md) |

---

## References

- [W3C WebNN specification](https://www.w3.org/TR/webnn/)
- [ONNX Operator Schemas](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [webnn-onnx-utils operation_names.rs](../../webnn-onnx-utils/src/operation_names.rs)
- [webnn-graph opset 21 plan (historical)](../../webnn-graph/docs/onnx-opsets/opset21-implementation-plan.md)
- [onnx2webnn AGENTS.md](../AGENTS.md)
