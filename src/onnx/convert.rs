/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Main ONNX to WebNN conversion logic

use crate::onnx::builder::{map_rustnn_error, tensor_proto_to_bytes, OnnxBuilder};
use crate::protos::onnx::{
    tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeProtoValue,
    ModelProto, TensorProto_DataType,
};
use prost::Message;
use rustnn::graph::{Dimension, DynamicDimension};
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraph, MLGraphBuilder, MLOperand, MLPowerPreference,
};
use rustnn::DataType;
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use thiserror::Error;
use webnn_onnx_utils::{data_types as utils_data_types, identifiers};

/// ONNX model lowered and validated via rustnn ORT `build()`.
pub struct ValidatedGraph<'ctx> {
    pub context: MLContext<'ctx>,
    pub graph: MLGraph<'ctx>,
}

const MIN_SUPPORTED_OPSET: i64 = 1;
const MAX_SUPPORTED_OPSET: i64 = 26;

/// ONNX ops that lower to WebNN element-wise logical ops and must emit `uint8` outputs.
/// Do not inline-fold them as integer constants (e.g. i64), since `where()` requires uint8 conditions.
fn is_element_wise_logical_onnx_op(op_type: &str) -> bool {
    matches!(
        op_type,
        "Equal"
            | "Greater"
            | "Less"
            | "GreaterOrEqual"
            | "LessOrEqual"
            | "Not"
            | "And"
            | "Or"
            | "Xor"
    )
}

/// One unsupported ONNX node reported during conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedOpEntry {
    pub op: String,
    pub node: String,
}

fn format_unsupported_ops_list(ops: &[UnsupportedOpEntry]) -> String {
    ops.iter()
        .map(|entry| format!("{} (node: {})", entry.op, entry.node))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("failed to read ONNX file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse ONNX protobuf: {0}")]
    ProtobufError(String),

    #[error("unsupported ONNX opset version {version} for domain '{domain}'")]
    UnsupportedOpset { domain: String, version: i64 },

    #[error("unsupported operator(s): {}", format_unsupported_ops_list(.0))]
    UnsupportedOps(Vec<UnsupportedOpEntry>),

    #[error("missing required attribute: {attr} in {op}")]
    MissingAttribute { attr: String, op: String },

    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("type conversion error: {0}")]
    TypeConversion(#[from] webnn_onnx_utils::error::ConversionError),

    #[error("shape inference failed for node: {0}")]
    ShapeInference(String),
}

impl OnnxError {
    /// Report a single unsupported operator/node pair.
    pub fn unsupported_op(op: impl Into<String>, node: impl Into<String>) -> Self {
        Self::UnsupportedOps(vec![UnsupportedOpEntry {
            op: op.into(),
            node: node.into(),
        }])
    }

    /// True when conversion failed because one or more operators are unsupported.
    pub fn is_unsupported_op(&self) -> bool {
        matches!(self, Self::UnsupportedOps(_))
    }

    /// Unsupported operator entries, when this error is [`Self::UnsupportedOps`].
    pub fn unsupported_ops(&self) -> Option<&[UnsupportedOpEntry]> {
        match self {
            Self::UnsupportedOps(ops) => Some(ops),
            _ => None,
        }
    }
}

/// Sanitize ONNX identifiers for WebNN DSL compatibility
/// Replaces problematic characters that would confuse the parser, and prefixes
/// digit-leading names (e.g. anonymous ONNX outputs like "495") with `_` so they
/// remain parseable in the .webnn text format.
pub fn sanitize_identifier(name: &str) -> String {
    let base = identifiers::sanitize_for_webnn(name);
    match base.chars().next() {
        Some(c) if c.is_ascii_digit() => format!("_{}", base),
        _ => base,
    }
}

/// Convert ONNX data type code to WebNN DataType using shared utilities
pub(crate) fn map_onnx_data_type(onnx_type: i32) -> Result<DataType, OnnxError> {
    if onnx_type == TensorProto_DataType::Bool as i32 {
        return Ok(DataType::Uint8);
    }
    // WebNN has no float64; double tensors are lowered to float32.
    if onnx_type == TensorProto_DataType::Double as i32 {
        return Ok(DataType::Float32);
    }
    // Packed 4-bit tensors (ONNX 1.16+): UINT4 = 21, INT4 = 22.
    if onnx_type == 21 {
        return Ok(DataType::Uint4);
    }
    if onnx_type == 22 {
        return Ok(DataType::Int4);
    }

    let utils_dtype = utils_data_types::onnx_to_webnn(onnx_type)?;
    Ok(match utils_dtype {
        utils_data_types::DataType::Float32 => DataType::Float32,
        utils_data_types::DataType::Float16 => DataType::Float16,
        utils_data_types::DataType::Int32 => DataType::Int32,
        utils_data_types::DataType::Uint32 => DataType::Uint32,
        utils_data_types::DataType::Int64 => DataType::Int64,
        utils_data_types::DataType::Uint64 => DataType::Uint64,
        utils_data_types::DataType::Int8 => DataType::Int8,
        utils_data_types::DataType::Uint8 => DataType::Uint8,
    })
}

/// Conversion options for ONNX -> MLGraphBuilder lowering + ORT validation.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    /// Override dynamic dimension values (e.g., batch_size=1, sequence_length=128)
    pub free_dim_overrides: HashMap<String, u32>,
    /// Enable constant folding and shape propagation optimizations
    pub optimize: bool,
    /// Experimental: preserve unresolved dynamic input dimensions in graph metadata
    pub experimental_dynamic_inputs: bool,
    /// Graph inputs frozen to a constant (e.g. `use_cache_branch=false`),
    /// turning runtime `If` gates into constant ones that can be inlined.
    pub pinned_inputs: HashMap<String, i64>,
    /// Zero-fill external tensors whose data file is missing. Lets a
    /// weight-stripped "skeleton" model exercise the full conversion and ORT
    /// graph build (values never matter for graph structure).
    pub zero_fill_missing_external_data: bool,
}

/// Parse a `--pin-input NAME=VALUE` argument (`true`/`false` or an integer).
pub fn parse_pinned_input(spec: &str) -> Result<(String, i64), OnnxError> {
    let (name, value) = spec.split_once('=').ok_or_else(|| {
        OnnxError::InvalidShape(format!(
            "Invalid pin-input format: '{spec}'. Expected NAME=VALUE"
        ))
    })?;
    let value = match value.trim() {
        "true" => 1,
        "false" => 0,
        v => v.parse::<i64>().map_err(|_| {
            OnnxError::InvalidShape(format!(
                "Invalid pin-input value '{v}' for '{name}': expected true/false or an integer"
            ))
        })?,
    };
    Ok((name.trim().to_string(), value))
}

/// Replace the named graph inputs with constant initializers of the declared
/// type and (static) shape, so constant folding and `If` inlining treat them
/// as constants.
pub fn pin_graph_inputs(
    model: &mut ModelProto,
    pinned: &HashMap<String, i64>,
) -> Result<(), OnnxError> {
    if pinned.is_empty() {
        return Ok(());
    }
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| OnnxError::InvalidShape("model has no graph".to_string()))?;
    for (name, &value) in pinned {
        let idx = graph
            .input
            .iter()
            .position(|vi| vi.name == *name)
            .ok_or_else(|| {
                OnnxError::InvalidShape(format!("pin-input: '{name}' is not a graph input"))
            })?;
        let vi = graph.input.remove(idx);
        let Some(TypeProtoValue::TensorType(tt)) =
            vi.r#type.as_ref().and_then(|t| t.value.as_ref())
        else {
            return Err(OnnxError::InvalidShape(format!(
                "pin-input: '{name}' is not a tensor input"
            )));
        };
        let dims: Vec<i64> = match tt.shape.as_ref() {
            Some(shape) => shape
                .dim
                .iter()
                .map(|d| match d.value.as_ref() {
                    Some(DimensionValue::DimValue(v)) => Ok(*v),
                    _ => Err(OnnxError::InvalidShape(format!(
                        "pin-input: '{name}' has a dynamic dimension; only static shapes can be pinned"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };
        let numel: usize = dims.iter().product::<i64>().max(1) as usize;
        let elem_type = tt.elem_type;
        let raw_data: Vec<u8> = if elem_type == TensorProto_DataType::Bool as i32 {
            vec![u8::from(value != 0); numel]
        } else if elem_type == TensorProto_DataType::Int64 as i32 {
            value.to_le_bytes().repeat(numel)
        } else if elem_type == TensorProto_DataType::Int32 as i32 {
            (value as i32).to_le_bytes().repeat(numel)
        } else if elem_type == TensorProto_DataType::Float as i32 {
            (value as f32).to_le_bytes().repeat(numel)
        } else {
            return Err(OnnxError::InvalidShape(format!(
                "pin-input: unsupported element type {elem_type} for '{name}' (bool, int32, int64, float)"
            )));
        };
        graph.initializer.push(crate::protos::onnx::TensorProto {
            name: name.clone(),
            data_type: elem_type,
            dims,
            raw_data,
            ..Default::default()
        });
    }
    Ok(())
}

/// Drop graph outputs that are zero-size constants (optimum's merged decoders
/// return dummy `[0, H, 1, D]` encoder KV outputs from the cache branch).
/// WebNN cannot represent zero-size tensors and they carry no data.
fn prune_empty_graph_outputs(model: &mut ModelProto) {
    let Some(graph) = model.graph.as_mut() else {
        return;
    };
    let mut empty_consts: HashSet<String> = graph
        .initializer
        .iter()
        .filter(|t| t.dims.contains(&0))
        .map(|t| t.name.clone())
        .collect();
    for node in graph.node.iter().filter(|n| n.op_type == "Constant") {
        let is_empty = node.attribute.iter().any(|a| {
            a.name.as_str() == "value" && a.t.as_ref().is_some_and(|t| t.dims.contains(&0))
        });
        if is_empty {
            empty_consts.extend(node.output.iter().cloned());
        }
    }
    if empty_consts.is_empty() {
        return;
    }
    let before = graph.output.len();
    graph.output.retain(|o| !empty_consts.contains(&o.name));
    if graph.output.len() == before {
        return;
    }
    if graph.output.is_empty() {
        // Keep the model well-formed; conversion reports the empty outputs.
        return;
    }
    let consumed: HashSet<String> = graph
        .node
        .iter()
        .flat_map(|n| n.input.iter().cloned())
        .chain(graph.output.iter().map(|o| o.name.clone()))
        .collect();
    graph
        .node
        .retain(|n| n.op_type != "Constant" || n.output.iter().any(|o| consumed.contains(o)));
    graph
        .initializer
        .retain(|t| !t.dims.contains(&0) || consumed.contains(&t.name));
    crate::debug_println!(
        "[CONVERT] Dropped {} zero-size graph output(s)",
        before - graph.output.len()
    );
}

/// Drop nodes whose outputs nothing consumes, transitively - e.g. the
/// shape-derived `Shape -> Gather -> Equal -> Cast` condition chain left behind
/// after a constant `If` is inlined. Dead ops are not just waste: backends may
/// reject them (CoreML fails to compile a scalar `equal` the graph never uses).
/// ONNX nodes are pure, so removal is side-effect free.
fn prune_dead_nodes(graph: &mut crate::protos::onnx::GraphProto) {
    /// Names a subgraph reads from enclosing scopes: anything referenced by its
    /// nodes/outputs that the subgraph does not define itself. Locals of
    /// intermediate scopes are over-approximated as captures, which only
    /// over-keeps - safe for pruning.
    fn collect_subgraph_captures(
        g: &crate::protos::onnx::GraphProto,
        needed: &mut HashSet<String>,
    ) {
        let mut local: HashSet<&str> = g.input.iter().map(|vi| vi.name.as_str()).collect();
        local.extend(g.initializer.iter().map(|t| t.name.as_str()));
        for n in &g.node {
            local.extend(n.output.iter().map(|s| s.as_str()));
        }
        for n in &g.node {
            for i in &n.input {
                if !i.is_empty() && !local.contains(i.as_str()) {
                    needed.insert(i.clone());
                }
            }
            for a in &n.attribute {
                if let Some(sg) = a.g.as_ref() {
                    collect_subgraph_captures(sg, needed);
                }
                for sg in &a.graphs {
                    collect_subgraph_captures(sg, needed);
                }
            }
        }
        for o in &g.output {
            if !local.contains(o.name.as_str()) {
                needed.insert(o.name.clone());
            }
        }
    }

    let mut needed: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();
    let mut kept = vec![false; graph.node.len()];
    // Nodes are topologically sorted per the ONNX spec, so one reverse pass
    // reaches every transitive producer.
    for (idx, node) in graph.node.iter().enumerate().rev() {
        if node.output.iter().any(|o| needed.contains(o)) {
            kept[idx] = true;
            needed.extend(node.input.iter().filter(|i| !i.is_empty()).cloned());
            // Control-flow bodies (If/Loop/Scan) read outer values that never
            // appear in node.input; keep their producers and initializers.
            for a in &node.attribute {
                if let Some(sg) = a.g.as_ref() {
                    collect_subgraph_captures(sg, &mut needed);
                }
                for sg in &a.graphs {
                    collect_subgraph_captures(sg, &mut needed);
                }
            }
        }
    }
    if kept.iter().all(|&k| k) {
        return;
    }
    let dropped = kept.iter().filter(|&&k| !k).count();
    let mut it = kept.iter();
    graph.node.retain(|_| *it.next().unwrap());
    // Drop initializers only dead nodes referenced; keep any that double as
    // graph inputs (older opsets) so declared feeds stay well-formed.
    let input_names: HashSet<&str> = graph.input.iter().map(|vi| vi.name.as_str()).collect();
    graph
        .initializer
        .retain(|t| needed.contains(&t.name) || input_names.contains(t.name.as_str()));
    crate::debug_println!("[if-inline] pruned {dropped} dead node(s)");
}

/// Drop graph inputs no node consumes (e.g. the KV cache of an inlined
/// no-cache branch). Inputs wired straight to a graph output are kept.
fn prune_unused_graph_inputs(model: &mut ModelProto) {
    let Some(graph) = model.graph.as_mut() else {
        return;
    };
    let used: HashSet<String> = graph
        .node
        .iter()
        .flat_map(|n| n.input.iter().cloned())
        .chain(graph.output.iter().map(|o| o.name.clone()))
        .collect();
    graph.input.retain(|vi| used.contains(&vi.name));
}

struct TensorInfo {
    _data_type: DataType,
    _shape: Vec<i64>,
}

/// Main converter structure
pub struct OnnxConverter {
    model: ModelProto,
    _value_info: HashMap<String, TensorInfo>,
}

impl OnnxConverter {
    /// Create a new converter from an ONNX model
    pub fn new(model: ModelProto) -> Result<Self, OnnxError> {
        Ok(Self {
            model,
            _value_info: HashMap::new(),
        })
    }

    /// Extract metadata from ONNX model
    pub fn extract_metadata(&self) -> Result<(), OnnxError> {
        if self.model.graph.is_none() {
            return Err(OnnxError::ProtobufError(
                "Missing graph in model".to_string(),
            ));
        }

        let graph = self.model.graph.as_ref().unwrap();
        let graph_name = if graph.name.is_empty() {
            "graph"
        } else {
            graph.name.as_str()
        };

        // Print basic info
        println!("Model name: {graph_name}");
        println!("Inputs: {}", graph.input.as_slice().len());
        println!("Outputs: {}", graph.output.as_slice().len());
        println!("Nodes: {}", graph.node.as_slice().len());
        println!("Initializers: {}", graph.initializer.as_slice().len());

        Ok(())
    }

    /// Lower ONNX into an [`OnnxBuilder`] (MLGraphBuilder + operand map).
    pub fn convert_with_builder(
        &self,
        b: &mut OnnxBuilder<'_, '_, '_>,
        options: &ConvertOptions,
    ) -> Result<(), OnnxError> {
        if self.model.graph.is_none() {
            return Err(OnnxError::ProtobufError(
                "Missing graph in model".to_string(),
            ));
        }

        // Validate opset imports
        for import in self.model.opset_import.as_slice() {
            let domain = import.domain.as_str();
            let version = import.version;
            let domain_name = if domain.is_empty() {
                "ai.onnx".to_string()
            } else {
                domain.to_string()
            };

            if (domain.is_empty() || domain == "ai.onnx")
                && !(MIN_SUPPORTED_OPSET..=MAX_SUPPORTED_OPSET).contains(&version)
            {
                return Err(OnnxError::UnsupportedOpset {
                    domain: domain_name,
                    version,
                });
            }
        }

        let onnx_graph = self.model.graph.as_ref().unwrap();

        // Fail fast on unsupported operators before any graph setup. Input/initializer
        // registration below can error on tensor kinds an unsupported op happens to use
        // (e.g. bool/string initializers), which would otherwise mask the real cause with a
        // confusing shape/builder error instead of a clean `UnsupportedOps`.
        //
        // **Domain behavior (today):** the pre-scan keys handlers by `op_type` only; per-node
        // `domain` (when present on `NodeProto`) is not consulted. That matches
        // [`OpRegistry::convert_node`] and [`OpRegistry::is_supported`]. Opset gating above
        // applies only to the standard `ai.onnx` domain (empty or `"ai.onnx"` import); other
        // `opset_import` entries are not version-checked yet.
        //
        // **Custom / vendor domains later:** to support non-official ops (e.g.
        // `com.microsoft.FusedConv`), extend handlers to register on `(domain, op_type)` and
        // update this loop to use the same key. Until then, a custom-domain node whose `op_type`
        // collides with an `ai.onnx` name may pass the pre-scan and be lowered incorrectly -
        // domain-aware dispatch is required before enabling those graphs.
        {
            let registry = crate::onnx::ops::OpRegistry::new();
            let unsupported = registry.collect_unsupported_nodes(onnx_graph.node.as_slice());
            if !unsupported.is_empty() {
                return Err(OnnxError::UnsupportedOps(unsupported));
            }
        }

        let mut value_name_map: HashMap<String, String> = HashMap::new();
        let mut effective_overrides = options.free_dim_overrides.clone();
        let mut inference_overrides = effective_overrides.clone();
        let mut value_types: HashMap<String, DataType> = HashMap::new();

        // Merge overrides from model metadata if present
        for meta in self.model.metadata_props.as_slice() {
            if meta
                .key
                .as_str()
                .eq_ignore_ascii_case("freedimensionoverrides")
            {
                if let Ok(json) = serde_json::from_str::<JsonValue>(meta.value.as_str()) {
                    let obj = json
                        .get("freeDimensionOverrides")
                        .unwrap_or(&json)
                        .as_object()
                        .cloned();
                    if let Some(map) = obj {
                        for (name, value) in map {
                            if let Some(v) = value.as_u64() {
                                effective_overrides.entry(name.clone()).or_insert(v as u32);
                            }
                        }
                    }
                }
            }
        }

        // Process inputs (exclude initializers)
        let initializer_names: HashSet<String> = onnx_graph
            .initializer
            .as_slice()
            .iter()
            .map(|init| init.name.as_str().to_string())
            .collect();

        let default_dynamic_max_size: u32 = 65_535;
        let default_inference_dim_values: HashMap<&str, u32> =
            HashMap::from([("batch_size", 1), ("batch", 1), ("n", 1), ("b", 1)]);
        let dynamic_max_for_dim = |name: &str| -> u32 {
            let lower = name.to_ascii_lowercase();
            if lower.contains("past")
                || lower.contains("seq")
                || lower.contains("length")
                || lower == "s"
                || lower == "t"
            {
                4096
            } else if lower.contains("batch") || lower == "b" || lower == "n" {
                8
            } else {
                default_dynamic_max_size
            }
        };
        let resolve_dim_override =
            |dim_param: &str, overrides: &HashMap<String, u32>| -> Option<u32> {
                if let Some(v) = overrides.get(dim_param) {
                    return Some(*v);
                }

                let lower = dim_param.to_ascii_lowercase();
                overrides.get(&lower).copied()
            };
        let dimension_for_param =
            |dim_param: &str, overrides: &HashMap<String, u32>| -> Dimension {
                if let Some(v) = resolve_dim_override(dim_param, overrides) {
                    Dimension::Static(v)
                } else {
                    Dimension::Dynamic(DynamicDimension {
                        name: dim_param.to_string(),
                        max_size: dynamic_max_for_dim(dim_param),
                    })
                }
            };

        let resolve_dim_for_inference =
            |dim_param: &str, overrides: &mut HashMap<String, u32>| -> Option<u32> {
                if let Some(v) = resolve_dim_override(dim_param, overrides) {
                    return Some(v);
                }
                let lower = dim_param.to_ascii_lowercase();
                if let Some(v) = default_inference_dim_values.get(lower.as_str()) {
                    overrides.insert(dim_param.to_string(), *v);
                    return Some(*v);
                }
                None
            };

        for input in onnx_graph.input.as_slice() {
            let raw_name = input.name.as_str().to_string();
            let name = sanitize_identifier(&raw_name);

            // Skip if this is an initializer (constant)
            if initializer_names.contains(&raw_name) {
                continue;
            }

            // Get type info
            if let Some(type_proto) = &input.r#type {
                if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                    let data_type = if tensor_type.elem_type != 0 {
                        let onnx_type = tensor_type.elem_type;
                        map_onnx_data_type(onnx_type)?
                    } else {
                        DataType::Float32 // Default
                    };

                    let shape = if let Some(shape_proto) = &tensor_type.shape {
                        let mut resolved: Vec<Dimension> = Vec::new();
                        for (idx, dim) in shape_proto.dim.iter().enumerate() {
                            if let Some(dim_value) = &dim.value {
                                match dim_value {
                                    DimensionValue::DimValue(v) => {
                                        if *v > 0 {
                                            resolved.push(Dimension::Static(*v as u32));
                                        } else if let Some(v) =
                                            effective_overrides.get(&format!("{}_dim{}", name, idx))
                                        {
                                            resolved.push(Dimension::Static(*v));
                                        } else if options.experimental_dynamic_inputs {
                                            resolved.push(Dimension::Dynamic(DynamicDimension {
                                                name: format!("{}_dim{}", name, idx),
                                                max_size: default_dynamic_max_size,
                                            }));
                                        } else {
                                            let dim_hint = format!("{}_dim{}", name, idx);
                                            return Err(OnnxError::InvalidShape(format!(
                                                "Input '{}' has non-positive dim value ({}) at index {}. \
Provide --override-dim {}=<value> or enable --experimental-dynamic-inputs.",
                                                raw_name,
                                                v,
                                                idx,
                                                dim_hint
                                            )));
                                        }
                                    }
                                    DimensionValue::DimParam(dim_param) => {
                                        if let Some(v) =
                                            resolve_dim_override(dim_param, &effective_overrides)
                                        {
                                            resolved.push(Dimension::Static(v));
                                        } else if options.experimental_dynamic_inputs {
                                            let max_size = dynamic_max_for_dim(dim_param);
                                            resolved.push(Dimension::Dynamic(DynamicDimension {
                                                name: dim_param.to_string(),
                                                max_size,
                                            }));
                                        } else if let Some(v) = resolve_dim_for_inference(
                                            dim_param,
                                            &mut inference_overrides,
                                        ) {
                                            effective_overrides
                                                .entry(dim_param.clone())
                                                .or_insert(v);
                                            resolved.push(Dimension::Static(v));
                                        } else {
                                            return Err(OnnxError::InvalidShape(format!(
                                                "Input '{}' has unresolved dynamic dimension '{}'. \
Provide --override-dim {}=<value> or enable --experimental-dynamic-inputs.",
                                                raw_name, dim_param, dim_param
                                            )));
                                        }
                                    }
                                }
                            } else if options.experimental_dynamic_inputs {
                                resolved.push(Dimension::Dynamic(DynamicDimension {
                                    name: format!("{}_dim{}", name, idx),
                                    max_size: default_dynamic_max_size,
                                }));
                            } else {
                                let dim_hint = format!("{}_dim{}", name, idx);
                                return Err(OnnxError::InvalidShape(format!(
                                    "Input '{}' has unknown dimension at index {}. \
Provide --override-dim {}=<value> or enable --experimental-dynamic-inputs.",
                                    raw_name, idx, dim_hint
                                )));
                            }
                        }
                        resolved
                    } else {
                        return Err(OnnxError::InvalidShape(format!(
                            "Input '{}' is missing shape information",
                            raw_name
                        )));
                    };

                    if shape.is_empty() {
                        continue;
                    }

                    b.register_input(&raw_name, data_type, &shape)?;

                    value_name_map.insert(raw_name.clone(), name.clone());
                    value_name_map.insert(name.clone(), name.clone());
                    value_types.insert(raw_name.clone(), data_type);
                    value_types.insert(name.clone(), data_type);
                }
            }
        }

        // Process initializers (constants/weights)
        for initializer in onnx_graph.initializer.as_slice() {
            let name = sanitize_identifier(initializer.name.as_str());
            let raw_data = initializer.raw_data.as_slice();

            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !raw_data.is_empty()
                || !initializer.float_data.as_slice().is_empty()
                || !initializer.int32_data.as_slice().is_empty()
                || !initializer.int64_data.as_slice().is_empty()
                || !initializer.double_data.as_slice().is_empty();

            // Zero-element tensors are optional-input placeholders (often produced when
            // empty Constant nodes are folded into initializers). Track them so Cast/Resize
            // can treat them as absent without materializing invalid WebNN 0-sized dims.
            if crate::onnx::builder::tensor_element_count(initializer) == 0 {
                b.mark_empty_optional(initializer.name.as_str());
                continue;
            }

            if !has_data {
                crate::debug_println!("Warning: Skipping initializer '{}' with no data", name);
                continue;
            }

            let onnx_type = initializer.data_type;
            let data_type = map_onnx_data_type(onnx_type)?;
            let shape: Vec<u32> = initializer
                .dims
                .as_slice()
                .iter()
                .map(|d| *d as u32)
                .collect();

            let bytes = tensor_proto_to_bytes(initializer)?;
            b.register_constant_from_bytes(initializer.name.as_str(), data_type, &shape, bytes)?;

            value_name_map.insert(initializer.name.as_str().to_string(), name.clone());
            value_name_map.insert(name.clone(), name.clone());
            value_types.insert(initializer.name.as_str().to_string(), data_type);
            value_types.insert(name, data_type);
        }

        // Process nodes using OpRegistry
        let registry = crate::onnx::ops::OpRegistry::new();

        // Build initializers map for resolving constant shapes
        let mut initializers_map = std::collections::HashMap::new();
        for initializer in onnx_graph.initializer.as_slice() {
            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !initializer.raw_data.as_slice().is_empty()
                || !initializer.float_data.as_slice().is_empty()
                || !initializer.int32_data.as_slice().is_empty()
                || !initializer.int64_data.as_slice().is_empty()
                || !initializer.double_data.as_slice().is_empty();

            if !has_data {
                continue;
            }
            initializers_map.insert(initializer.name.as_str().to_string(), initializer);
        }

        // Build value_shapes map from value_info and inputs for shape inference
        let mut value_shapes = std::collections::HashMap::new();
        let mut value_shape_dims = std::collections::HashMap::new();

        // Add input shapes (already validated)
        for (raw_name, mapped_name) in value_name_map.clone() {
            if initializer_names.contains(&raw_name) {
                continue;
            }
            if let Some(input) = onnx_graph
                .input
                .as_slice()
                .iter()
                .find(|i| i.name.as_str() == raw_name)
            {
                if let Some(type_proto) = &input.r#type {
                    if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                        if let Some(shape_proto) = &tensor_type.shape {
                            let mut shape: Vec<i64> = Vec::new();
                            let mut unknown = false;
                            for dim in &shape_proto.dim {
                                if let Some(dim_value) = &dim.value {
                                    match dim_value {
                                        DimensionValue::DimValue(v) => {
                                            if *v > 0 {
                                                shape.push(*v);
                                            } else if options.experimental_dynamic_inputs {
                                                shape.push(default_dynamic_max_size as i64);
                                            } else {
                                                unknown = true;
                                                break;
                                            }
                                        }
                                        DimensionValue::DimParam(dim_param) => {
                                            if options.experimental_dynamic_inputs {
                                                shape.push(
                                                    resolve_dim_override(
                                                        dim_param,
                                                        &inference_overrides,
                                                    )
                                                    .unwrap_or_else(|| {
                                                        dynamic_max_for_dim(dim_param)
                                                    })
                                                        as i64,
                                                );
                                            } else if let Some(v) = resolve_dim_for_inference(
                                                dim_param,
                                                &mut inference_overrides,
                                            ) {
                                                shape.push(v as i64);
                                            } else {
                                                unknown = true;
                                                break;
                                            }
                                        }
                                    }
                                } else if options.experimental_dynamic_inputs {
                                    shape.push(default_dynamic_max_size as i64);
                                } else {
                                    unknown = true;
                                    break;
                                }
                            }
                            if !unknown && !shape.is_empty() {
                                value_shapes.insert(raw_name.clone(), shape.clone());
                                value_shapes.insert(mapped_name.clone(), shape);
                            }
                            let mut dims = Vec::new();
                            for dim in &shape_proto.dim {
                                if let Some(dim_value) = &dim.value {
                                    match dim_value {
                                        DimensionValue::DimValue(v) => {
                                            if *v > 0 {
                                                dims.push(rustnn::graph::Dimension::Static(
                                                    *v as u32,
                                                ));
                                            }
                                        }
                                        DimensionValue::DimParam(dim_param) => {
                                            dims.push(dimension_for_param(
                                                dim_param,
                                                &inference_overrides,
                                            ));
                                        }
                                    }
                                }
                            }
                            if !dims.is_empty() {
                                value_shape_dims.insert(raw_name.clone(), dims.clone());
                                value_shape_dims.insert(mapped_name.clone(), dims);
                            }
                        }
                    }
                }
            }
        }

        // Add initializer shapes
        for initializer in onnx_graph.initializer.as_slice() {
            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !initializer.raw_data.as_slice().is_empty()
                || !initializer.float_data.as_slice().is_empty()
                || !initializer.int32_data.as_slice().is_empty()
                || !initializer.int64_data.as_slice().is_empty()
                || !initializer.double_data.as_slice().is_empty();

            if !has_data {
                continue;
            }
            let shape: Vec<i64> = initializer.dims.as_slice().to_vec();
            value_shapes.insert(initializer.name.as_str().to_string(), shape);
            let dims: Vec<rustnn::graph::Dimension> = initializer
                .dims
                .iter()
                .copied()
                .filter(|d| *d > 0)
                .map(|d| rustnn::graph::Dimension::Static(d as u32))
                .collect();
            if !dims.is_empty() {
                value_shape_dims.insert(initializer.name.as_str().to_string(), dims);
            }
        }

        // Add value_info shapes (intermediate tensors from shape inference)
        // Try to resolve dynamic dimensions using overrides
        for value_info in onnx_graph.value_info.as_slice() {
            if let Some(type_proto) = &value_info.r#type {
                if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                    if let Some(shape_proto) = &tensor_type.shape {
                        let mut shape: Vec<i64> = Vec::new();
                        let mut unknown = false;

                        for dim in &shape_proto.dim {
                            if let Some(dim_value) = &dim.value {
                                match dim_value {
                                    DimensionValue::DimValue(v) => {
                                        if *v > 0 {
                                            shape.push(*v);
                                        } else if options.experimental_dynamic_inputs {
                                            shape.push(default_dynamic_max_size as i64);
                                        } else {
                                            unknown = true;
                                            break;
                                        }
                                    }
                                    DimensionValue::DimParam(dim_param) => {
                                        if options.experimental_dynamic_inputs {
                                            shape.push(
                                                resolve_dim_override(
                                                    dim_param,
                                                    &inference_overrides,
                                                )
                                                .unwrap_or_else(|| dynamic_max_for_dim(dim_param))
                                                    as i64,
                                            );
                                        } else if let Some(v) = resolve_dim_for_inference(
                                            dim_param,
                                            &mut inference_overrides,
                                        ) {
                                            shape.push(v as i64);
                                        } else {
                                            unknown = true;
                                            break;
                                        }
                                    }
                                }
                            } else if options.experimental_dynamic_inputs {
                                shape.push(default_dynamic_max_size as i64);
                            } else {
                                unknown = true;
                                break;
                            }
                        }

                        if !unknown && !shape.is_empty() && shape.iter().all(|&d| d > 0) {
                            value_shapes.insert(value_info.name.as_str().to_string(), shape);
                        }
                        let mut dims = Vec::new();
                        for dim in &shape_proto.dim {
                            if let Some(dim_value) = &dim.value {
                                match dim_value {
                                    DimensionValue::DimValue(v) => {
                                        if *v > 0 {
                                            dims.push(rustnn::graph::Dimension::Static(*v as u32));
                                        }
                                    }
                                    DimensionValue::DimParam(dim_param) => {
                                        dims.push(dimension_for_param(
                                            dim_param,
                                            &inference_overrides,
                                        ));
                                    }
                                }
                            }
                        }
                        // Without the dynamic-inputs feature, rustnn rejects graphs
                        // containing Dynamic dimensions - keep only fully static
                        // metadata so unresolved composite dim_params (e.g.
                        // "batch_size * sequence_length") never reach the builder.
                        let keep_dims = options.experimental_dynamic_inputs
                            || dims
                                .iter()
                                .all(|d| matches!(d, rustnn::graph::Dimension::Static(_)));
                        if !dims.is_empty() && keep_dims {
                            value_shape_dims.insert(value_info.name.as_str().to_string(), dims);
                        }
                    }
                }
            }
        }

        // Seed const values with integer initializers and Constant nodes
        let mut const_values: HashMap<String, Vec<i64>> = HashMap::new();
        for (name, initializer) in &initializers_map {
            if initializer.data_type == TensorProto_DataType::Int64 as i32
                || initializer.data_type == TensorProto_DataType::Int32 as i32
            {
                let raw = initializer.raw_data.as_slice();
                let values = if !raw.is_empty() {
                    if initializer.data_type == TensorProto_DataType::Int32 as i32 {
                        raw.chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                            .collect()
                    } else {
                        raw.chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect()
                    }
                } else if !initializer.int64_data.as_slice().is_empty() {
                    initializer.int64_data.as_slice().to_vec()
                } else if !initializer.int32_data.as_slice().is_empty() {
                    initializer
                        .int32_data
                        .as_slice()
                        .iter()
                        .map(|&v| v as i64)
                        .collect()
                } else {
                    Vec::new()
                };

                if !values.is_empty() {
                    const_values.insert(name.clone(), values);
                }
            }
        }

        for node in onnx_graph.node.as_slice() {
            if node.op_type.as_str() == "Constant" {
                if let Some(attr) = node
                    .attribute
                    .as_slice()
                    .iter()
                    .find(|a| a.name.as_str() == "value" && a.t.is_some())
                {
                    let tensor = attr.t.as_ref().unwrap();
                    if tensor.data_type == TensorProto_DataType::Int64 as i32
                        || tensor.data_type == TensorProto_DataType::Int32 as i32
                    {
                        let raw = tensor.raw_data.as_slice();
                        let values = if !raw.is_empty() {
                            if tensor.data_type == TensorProto_DataType::Int32 as i32 {
                                raw.chunks_exact(4)
                                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                                    .collect()
                            } else {
                                raw.chunks_exact(8)
                                    .map(|c| {
                                        i64::from_le_bytes([
                                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                        ])
                                    })
                                    .collect()
                            }
                        } else if !tensor.int64_data.as_slice().is_empty() {
                            tensor.int64_data.as_slice().to_vec()
                        } else if !tensor.int32_data.as_slice().is_empty() {
                            tensor
                                .int32_data
                                .as_slice()
                                .iter()
                                .map(|&v| v as i64)
                                .collect()
                        } else {
                            Vec::new()
                        };

                        if let Some(out) = node.output.as_slice().first() {
                            if !values.is_empty() {
                                const_values.insert(out.to_string(), values);
                                value_types.insert(out.to_string(), DataType::Int64);
                            }
                        }
                    }
                }
            }
        }

        // Run the static shape/type inference scaffold to seed shapes/types/constants
        // before lowering. Errors surface early if dynamic dims remain.
        if options.experimental_dynamic_inputs {
            for input in onnx_graph.input.as_slice() {
                if initializer_names.contains(&input.name) {
                    continue;
                }
                if let Some(type_proto) = &input.r#type {
                    if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                        if let Some(shape_proto) = &tensor_type.shape {
                            for dim in &shape_proto.dim {
                                if let Some(DimensionValue::DimParam(dim_param)) = &dim.value {
                                    inference_overrides
                                        .entry(dim_param.clone())
                                        .or_insert_with(|| dynamic_max_for_dim(dim_param));
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut dynamic_inference_attempts: HashSet<String> = HashSet::new();
        loop {
            match crate::onnx::shape_inference::infer_static_shapes(
                &self.model,
                &inference_overrides,
            ) {
                Ok(inferred) => {
                    // Initial seeding: use or_insert since these are the first values
                    // (no prior shapes to override)
                    for (k, v) in inferred.value_shapes {
                        value_shapes.entry(k).or_insert(v);
                    }
                    for (k, v) in inferred.value_types {
                        value_types.entry(k).or_insert(v);
                    }
                    for (k, v) in inferred.const_values {
                        // Use insert() instead of or_insert() to allow shape inference to correct
                        // earlier wrong values (e.g., Where operation heuristics)
                        if k.contains("rotary") && k.contains("Where") {
                            if let Some(old_val) = const_values.get(&k) {
                                crate::debug_println!(
                                    "[CONVERT] Overwriting {} from {:?} to {:?}",
                                    k,
                                    old_val,
                                    v
                                );
                            } else {
                                crate::debug_println!("[CONVERT] Inserting new {} = {:?}", k, v);
                            }
                        }
                        const_values.insert(k, v);
                    }
                    break;
                }
                Err(crate::onnx::shape_inference::ShapeInferenceError::DynamicDim {
                    input,
                    dim,
                }) => {
                    if options.experimental_dynamic_inputs
                        && !dynamic_inference_attempts.contains(dim.as_str())
                    {
                        let fallback = resolve_dim_override(&dim, &inference_overrides)
                            .unwrap_or_else(|| dynamic_max_for_dim(&dim));
                        inference_overrides.insert(dim.clone(), fallback);
                        dynamic_inference_attempts.insert(dim.clone());
                        crate::debug_println!(
                            "[CONVERT] Retrying static shape inference with inferred override {}={} \
                             (required by input '{}')",
                            dim,
                            fallback,
                            input
                        );
                        continue;
                    }
                    crate::debug_println!(
                        "[CONVERT] Skipping static shape inference due to unresolved dynamic dim '{}' on input '{}'",
                        dim,
                        input
                    );
                    break;
                }
                Err(e) => return Err(OnnxError::ShapeInference(e.to_string())),
            }
        }

        crate::onnx::shape_inference::propagate_shapes_and_fold_constants(
            onnx_graph,
            &initializers_map,
            &mut value_shapes,
            &mut value_types,
            &mut const_values,
            &mut value_shape_dims,
            &crate::onnx::shape_inference::PropagateOptions {
                optimize: options.optimize,
                experimental_dynamic_inputs: options.experimental_dynamic_inputs,
            },
        );

        // DEBUG: Check value before node conversion
        if let Some(val) = const_values.get("/model/rotary_emb/Where_output_0") {
            crate::debug_println!("[NODE CONV] /model/rotary_emb/Where_output_0 = {:?}", val);
        }
        // O2W_PROBE=<substr>: dump post-propagation shape/const state for
        // matching node outputs (diagnostics only).
        if let Ok(probe) = std::env::var("O2W_PROBE") {
            for onnx_node in onnx_graph.node.as_slice() {
                for out in onnx_node.output.as_slice() {
                    if out.contains(&probe) {
                        eprintln!(
                            "[probe] {} ({}) shape={:?} type={:?} const={:?}",
                            out,
                            onnx_node.op_type,
                            value_shapes.get(out.as_str()),
                            value_types.get(out.as_str()),
                            const_values.get(out.as_str()).map(|v| v
                                .iter()
                                .take(8)
                                .copied()
                                .collect::<Vec<_>>()),
                        );
                    }
                }
            }
        }
        for onnx_node in onnx_graph.node.as_slice() {
            // If all outputs are compile-time constants, emit them directly and skip conversion
            let outputs = onnx_node.output.as_slice();
            let has_dynamic_output_metadata = outputs.iter().any(|o| {
                crate::onnx::shape_inference::value_shape_dims_for(o.as_str(), &value_shape_dims)
                    .map(|dims| dims.iter().any(|d| matches!(d, Dimension::Dynamic(_))))
                    .unwrap_or(false)
            });
            if !outputs.is_empty()
                && !has_dynamic_output_metadata
                && onnx_node.op_type.as_str() != "Cast"
                && onnx_node.op_type.as_str() != "ConstantOfShape"
                && !is_element_wise_logical_onnx_op(onnx_node.op_type.as_str())
                && outputs
                    .iter()
                    .all(|o| const_values.contains_key(o.as_str()))
            {
                // Check if outputs are true scalars (rank 0), not just single-element tensors
                let all_scalar = outputs.iter().all(|o| {
                    value_shapes
                        .get(o.as_str())
                        .map(|s| s.is_empty()) // True scalar has empty shape
                        .unwrap_or_else(|| {
                            // Fallback: check if data length is 1
                            const_values
                                .get(o.as_str())
                                .map(|v| v.len() == 1)
                                .unwrap_or(false)
                        })
                });

                // Handle scalar constants by emitting them inline
                if all_scalar {
                    for out in outputs {
                        if let Some(values) = const_values.get(out) {
                            let const_name = sanitize_identifier(out);
                            // Use the intended shape from value_shapes, not just empty for single-element
                            let shape = value_shapes
                                .get(out.as_str())
                                .map(|s| s.iter().map(|&d| d as u32).collect())
                                .unwrap_or_else(Vec::new);

                            let bytes = values[0].to_le_bytes().to_vec();
                            b.register_constant_from_bytes(
                                &const_name,
                                DataType::Int64,
                                &shape,
                                bytes,
                            )?;

                            value_name_map.insert(out.to_string(), const_name.clone());
                            value_name_map.insert(const_name.clone(), const_name.clone());
                            value_types.insert(out.to_string(), DataType::Int64);
                            value_types.insert(const_name, DataType::Int64);
                        }
                    }
                }
                // For non-scalar constants (like Range output), emit inline consts so downstream
                // nodes have a defined producer.
                for out in outputs {
                    if let Some(values) = const_values.get(out) {
                        // Zero-element constants (e.g. folded empty axes lists)
                        // cannot exist as WebNN operands; consumers treat them
                        // as absent optionals.
                        if values.is_empty() {
                            b.mark_empty_optional(out);
                            continue;
                        }
                        let const_name = sanitize_identifier(out);
                        let mut shape = value_shapes
                            .get(out.as_str())
                            .cloned()
                            .unwrap_or_else(|| vec![values.len() as i64]);
                        let declared_numel = shape
                            .iter()
                            .try_fold(1usize, |acc, d| usize::try_from(*d).ok().map(|v| acc * v));
                        if declared_numel != Some(values.len()) {
                            // Some folded constants are broadcast candidates where value_shapes
                            // carries the post-broadcast shape but const_values stores the compact payload.
                            // Keep shape/data internally consistent by using the compact shape.
                            shape = vec![values.len() as i64];
                            // Repair value_shapes so downstream shape lookups (e.g. Einsum)
                            // match the materialized operand instead of the inflated shape.
                            value_shapes.insert(out.to_string(), shape.clone());
                            value_shapes.insert(sanitize_identifier(out), shape.clone());
                        }
                        let dtype = value_types
                            .get(out.as_str())
                            .cloned()
                            .unwrap_or(DataType::Int64);

                        // Serialize the i64 payload at the width of the tracked
                        // dtype (folded bool masks are Uint8, Cast chains may
                        // be Int32); unsupported widths fall back to Int64.
                        let (dtype, bytes): (DataType, Vec<u8>) = match dtype {
                            DataType::Uint8 | DataType::Int8 => {
                                (dtype, values.iter().map(|&v| v as u8).collect())
                            }
                            DataType::Int32 => (
                                dtype,
                                values
                                    .iter()
                                    .flat_map(|&v| (v as i32).to_le_bytes())
                                    .collect(),
                            ),
                            DataType::Float32 => (
                                dtype,
                                values
                                    .iter()
                                    .flat_map(|&v| (v as f32).to_le_bytes())
                                    .collect(),
                            ),
                            DataType::Float16 => (
                                dtype,
                                values
                                    .iter()
                                    .flat_map(|&v| half::f16::from_f64(v as f64).to_le_bytes())
                                    .collect(),
                            ),
                            _ => (
                                DataType::Int64,
                                values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
                            ),
                        };

                        let shape_u32: Vec<u32> = shape.iter().map(|d| *d as u32).collect();
                        b.register_constant_from_bytes(&const_name, dtype, &shape_u32, bytes)?;

                        value_name_map.insert(out.to_string(), const_name.clone());
                        value_name_map.insert(const_name.clone(), const_name.clone());
                        value_types.insert(out.to_string(), dtype);
                        value_types.insert(const_name, dtype);
                    }
                }
                continue;
            }

            let context = crate::onnx::ops::ConversionContext {
                initializers: &initializers_map,
                value_shapes: &value_shapes,
                value_shape_dims: &value_shape_dims,
                const_values: &const_values,
                value_ids: &value_name_map,
                value_types: &value_types,
            };

            let converted = registry.convert_node(onnx_node, &context, b)?;

            for (onnx_out, dtype) in converted.output_types {
                let webnn_id = sanitize_identifier(&onnx_out);
                value_name_map.insert(onnx_out.clone(), webnn_id.clone());
                value_types.insert(webnn_id, dtype);
            }

            // Track output shapes after conversion to prevent shape inflation
            // Use .insert() to force correct shapes (not .or_insert() which preserves old shapes)
            if let Some(inferred_shape) = crate::onnx::shape_inference::infer_node_output_shape(
                onnx_node,
                &value_shapes,
                &initializers_map,
                &const_values,
            ) {
                for output_name in onnx_node.output.as_slice() {
                    // Insert shape for both raw and sanitized names
                    value_shapes.insert(output_name.to_string(), inferred_shape.clone());
                    value_shapes.insert(sanitize_identifier(output_name), inferred_shape.clone());
                }
            }
        }

        Ok(())
    }
}

/// Convert an ONNX file and validate via rustnn ORT `MLGraphBuilder::build()`.
/// Resolve `data_location = EXTERNAL` initializers by reading the referenced
/// files (relative to the model) into `raw_data`. Hub exports split large
/// weights across several chunk files (`model.onnx_data`, `model.onnx_data_1`,
/// ...); each tensor names its own `location`, so per-tensor reads handle the
/// chunking naturally. Files are read once and cached across tensors.
/// Zero-fill every external tensor (graph and subgraphs) that still lacks
/// data: the in-memory path of weight-stripped skeleton models.
fn zero_fill_external_tensors(
    graph: &mut crate::protos::onnx::GraphProto,
) -> Result<(), OnnxError> {
    const EXTERNAL: i32 = 1; // TensorProto_DataLocation::External
    for tensor in graph.initializer.iter_mut() {
        if tensor.data_location != EXTERNAL {
            continue;
        }
        let length = tensor
            .external_data
            .iter()
            .find(|e| e.key.as_str() == "length")
            .and_then(|e| e.value.parse::<usize>().ok());
        let len = tensor_byte_len(tensor).or(length).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "external tensor '{}' has no length and an unknown element size",
                tensor.name
            ))
        })?;
        tensor.raw_data = vec![0u8; len];
        tensor.data_location = 0;
        tensor.external_data.clear();
    }
    for node in graph.node.iter_mut() {
        for attr in node.attribute.iter_mut() {
            if let Some(sub) = attr.g.as_mut() {
                zero_fill_external_tensors(sub)?;
            }
            for sub in attr.graphs.iter_mut() {
                zero_fill_external_tensors(sub)?;
            }
        }
    }
    Ok(())
}

/// Drop every initializer payload (graph and subgraphs), keeping names, dims
/// and types. Called once lowering has copied all constants into the WebNN
/// graph, so the proto's weight volume is released before the backend compile.
fn strip_initializer_payloads(graph: &mut crate::protos::onnx::GraphProto) {
    for tensor in graph.initializer.iter_mut() {
        tensor.raw_data = Vec::new();
        tensor.float_data = Vec::new();
        tensor.int32_data = Vec::new();
        tensor.int64_data = Vec::new();
        tensor.double_data = Vec::new();
        tensor.uint64_data = Vec::new();
        tensor.string_data = Vec::new();
    }
    for node in graph.node.iter_mut() {
        for attr in node.attribute.iter_mut() {
            if let Some(sub) = attr.g.as_mut() {
                strip_initializer_payloads(sub);
            }
            for sub in attr.graphs.iter_mut() {
                strip_initializer_payloads(sub);
            }
        }
    }
}

/// Byte size of a tensor from its dims and element type.
fn tensor_byte_len(tensor: &crate::protos::onnx::TensorProto) -> Option<usize> {
    let elem = match tensor.data_type {
        x if x == TensorProto_DataType::Float as i32
            || x == TensorProto_DataType::Int32 as i32
            || x == TensorProto_DataType::Uint32 as i32 =>
        {
            4
        }
        x if x == TensorProto_DataType::Float16 as i32
            || x == TensorProto_DataType::Bfloat16 as i32
            || x == TensorProto_DataType::Int16 as i32
            || x == TensorProto_DataType::Uint16 as i32 =>
        {
            2
        }
        x if x == TensorProto_DataType::Int64 as i32
            || x == TensorProto_DataType::Uint64 as i32
            || x == TensorProto_DataType::Double as i32 =>
        {
            8
        }
        x if x == TensorProto_DataType::Int8 as i32
            || x == TensorProto_DataType::Uint8 as i32
            || x == TensorProto_DataType::Bool as i32 =>
        {
            1
        }
        // UINT4 / INT4: two elements per byte, handled below.
        21 | 22 => 0,
        _ => return None,
    };
    let numel: usize = tensor.dims.iter().try_fold(1usize, |acc, &d| {
        usize::try_from(d).ok().and_then(|d| acc.checked_mul(d))
    })?;
    if elem == 0 {
        return Some(numel.div_ceil(2));
    }
    numel.checked_mul(elem)
}

fn load_external_tensor_data(
    model: &mut ModelProto,
    onnx_path: &Path,
    zero_fill_missing: bool,
) -> Result<(), OnnxError> {
    const EXTERNAL: i32 = 1; // TensorProto_DataLocation::External
    let graph = match model.graph.as_mut() {
        Some(g) => g,
        None => return Ok(()),
    };
    if !graph
        .initializer
        .iter()
        .any(|t| t.data_location == EXTERNAL)
    {
        return Ok(());
    }
    let base_dir = onnx_path.parent().unwrap_or_else(|| Path::new("."));
    // Keep open handles, not file contents: weight files are multi-GB and every
    // tensor is copied into `raw_data` anyway.
    let mut files: std::collections::HashMap<String, fs::File> = std::collections::HashMap::new();

    for tensor in graph.initializer.iter_mut() {
        if tensor.data_location != EXTERNAL {
            continue;
        }
        let mut location = None;
        let mut offset = 0usize;
        let mut length = None;
        for entry in tensor.external_data.as_slice() {
            match entry.key.as_str() {
                "location" => location = Some(entry.value.clone()),
                "offset" => {
                    offset = entry.value.trim().parse::<usize>().map_err(|_| {
                        OnnxError::InvalidShape(format!(
                            "invalid external data offset '{}' for tensor '{}'",
                            entry.value, tensor.name
                        ))
                    })?;
                }
                "length" => {
                    length = Some(entry.value.trim().parse::<usize>().map_err(|_| {
                        OnnxError::InvalidShape(format!(
                            "invalid external data length '{}' for tensor '{}'",
                            entry.value, tensor.name
                        ))
                    })?);
                }
                _ => {}
            }
        }
        let location = location.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "external tensor '{}' has no data location",
                tensor.name
            ))
        })?;
        // The spec requires relative paths inside the model directory.
        if Path::new(&location).is_absolute() || location.contains("..") {
            return Err(OnnxError::InvalidShape(format!(
                "external tensor '{}' references a non-relative location '{location}'",
                tensor.name
            )));
        }
        let path = base_dir.join(&location);
        if zero_fill_missing && !path.exists() {
            // dims x element size is authoritative; `length` is a fallback
            // for element types we cannot size.
            let len = tensor_byte_len(tensor).or(length).ok_or_else(|| {
                OnnxError::InvalidShape(format!(
                    "external tensor '{}' has no length and an unknown element size",
                    tensor.name
                ))
            })?;
            tensor.raw_data = vec![0u8; len];
            tensor.data_location = 0;
            tensor.external_data.clear();
            continue;
        }
        if !files.contains_key(&location) {
            let file = fs::File::open(&path).map_err(|e| {
                OnnxError::InvalidShape(format!(
                    "failed to read external data '{}' for tensor '{}': {e}",
                    path.display(),
                    tensor.name
                ))
            })?;
            files.insert(location.clone(), file);
        }
        let file = files.get_mut(&location).expect("inserted above");
        let file_len = file
            .metadata()
            .map_err(|e| OnnxError::InvalidShape(format!("external data '{location}': {e}")))?
            .len() as usize;
        let end = match length {
            Some(len) => offset.checked_add(len),
            None => Some(file_len),
        }
        .filter(|&end| end <= file_len)
        .ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "external tensor '{}' range {offset}+{:?} exceeds '{location}' ({file_len} bytes)",
                tensor.name, length
            ))
        })?;
        use std::io::{Read, Seek, SeekFrom};
        let mut bytes = vec![0u8; end - offset];
        file.seek(SeekFrom::Start(offset as u64))
            .and_then(|_| file.read_exact(&mut bytes))
            .map_err(|e| {
                OnnxError::InvalidShape(format!(
                    "failed to read external data '{}' for tensor '{}': {e}",
                    path.display(),
                    tensor.name
                ))
            })?;
        tensor.raw_data = bytes;
        tensor.data_location = 0;
        tensor.external_data.clear();
    }
    Ok(())
}

pub fn convert_onnx<P: AsRef<Path>>(
    onnx_path: P,
    mut options: ConvertOptions,
) -> Result<ValidatedGraph<'static>, OnnxError> {
    // Read ONNX file
    let onnx_path_ref = onnx_path.as_ref();
    let onnx_bytes = fs::read(onnx_path_ref)?;

    // Parse protobuf
    let mut model: ModelProto =
        ModelProto::decode(&onnx_bytes[..]).map_err(|e| OnnxError::ProtobufError(e.to_string()))?;
    // Inline weights now live in `model`; release the file buffer.
    drop(onnx_bytes);

    load_external_tensor_data(
        &mut model,
        onnx_path_ref,
        options.zero_fill_missing_external_data,
    )?;

    // Merge overrides from sidecar dims file if provided implicitly and not already set
    if options.free_dim_overrides.is_empty() {
        let mut sidecar = onnx_path_ref.to_path_buf();
        sidecar.set_extension("dims.json");
        if sidecar.exists() {
            let content = fs::read_to_string(&sidecar)?;
            if let Ok(json) = serde_json::from_str::<JsonValue>(&content) {
                if let Some(obj) = json
                    .get("freeDimensionOverrides")
                    .unwrap_or(&json)
                    .as_object()
                {
                    for (name, value) in obj {
                        if let Some(v) = value.as_u64() {
                            options
                                .free_dim_overrides
                                .entry(name.clone())
                                .or_insert(v as u32);
                        }
                    }
                }
            }
        }
    }

    convert_model(model, &options)
}

/// Inline `If` nodes whose condition folds to a constant (e.g. pyannote's
/// shape-derived `Equal(Gather(Shape(x)), k)` gate). The chosen branch's
/// nodes and initializers are spliced into the outer graph with internal
/// names prefixed; outer-scope captures keep their names. Runtime-dependent
/// conditions are left in place (and later rejected as unsupported).
fn inline_constant_ifs(model: &mut ModelProto, options: &ConvertOptions) {
    use crate::protos::onnx::GraphProto;

    for _ in 0..4 {
        let graph = match model.graph.as_ref() {
            Some(g) if g.node.iter().any(|n| n.op_type == "If") => g,
            _ => return,
        };

        // Minimal seeding so shape-derived conditions fold.
        let mut value_shapes: HashMap<String, Vec<i64>> = HashMap::new();
        let mut value_types: HashMap<String, DataType> = HashMap::new();
        let mut const_values: HashMap<String, Vec<i64>> = HashMap::new();
        let mut value_shape_dims = HashMap::new();
        for vi in graph.input.as_slice() {
            let Some(TypeProtoValue::TensorType(tt)) =
                vi.r#type.as_ref().and_then(|t| t.value.as_ref())
            else {
                continue;
            };
            let Some(shape) = tt.shape.as_ref() else {
                continue;
            };
            let dims: Option<Vec<i64>> = shape
                .dim
                .iter()
                .enumerate()
                .map(|(idx, d)| match d.value.as_ref() {
                    Some(DimensionValue::DimValue(v)) if *v > 0 => Some(*v),
                    Some(DimensionValue::DimValue(_)) => options
                        .free_dim_overrides
                        .get(&format!("{}_dim{}", sanitize_identifier(&vi.name), idx))
                        .map(|&v| v as i64),
                    Some(DimensionValue::DimParam(p)) => {
                        options.free_dim_overrides.get(p).map(|&v| v as i64)
                    }
                    _ => None,
                })
                .collect();
            if let Some(dims) = dims {
                value_shapes.insert(vi.name.clone(), dims);
            }
        }
        let initializers_map: HashMap<String, &crate::protos::onnx::TensorProto> = graph
            .initializer
            .iter()
            .map(|t| (t.name.clone(), t))
            .collect();
        for (name, t) in &initializers_map {
            value_shapes.insert(name.clone(), t.dims.clone());
            let vals = crate::onnx::shape_inference::read_int_tensor(t);
            if !vals.is_empty() {
                const_values.insert(name.clone(), vals);
            }
        }
        for node in graph.node.as_slice() {
            if node.op_type == "Constant" {
                if let (Some(out), Some(t)) = (
                    node.output.first(),
                    node.attribute
                        .iter()
                        .find(|a| a.name == "value")
                        .and_then(|a| a.t.as_ref()),
                ) {
                    let vals = crate::onnx::shape_inference::read_int_tensor(t);
                    if !vals.is_empty() {
                        value_shapes.insert(out.clone(), t.dims.clone());
                        const_values.insert(out.clone(), vals);
                    }
                }
            }
        }
        crate::onnx::shape_inference::propagate_shapes_and_fold_constants(
            graph,
            &initializers_map,
            &mut value_shapes,
            &mut value_types,
            &mut const_values,
            &mut value_shape_dims,
            &crate::onnx::shape_inference::PropagateOptions {
                optimize: true,
                experimental_dynamic_inputs: false,
            },
        );

        // Splice each If with a folded condition.
        let mut new_nodes: Vec<crate::protos::onnx::NodeProto> = Vec::new();
        let mut new_initializers: Vec<crate::protos::onnx::TensorProto> = Vec::new();
        let mut changed = false;
        for node in graph.node.as_slice() {
            if node.op_type != "If" {
                new_nodes.push(node.clone());
                continue;
            }
            let cond = node
                .input
                .first()
                .and_then(|c| const_values.get(c.as_str()))
                .and_then(|v| v.first().copied());
            let Some(cond) = cond else {
                new_nodes.push(node.clone());
                continue;
            };
            let branch_attr = if cond != 0 {
                "then_branch"
            } else {
                "else_branch"
            };
            let Some(branch): Option<&GraphProto> = node
                .attribute
                .iter()
                .find(|a| a.name == branch_attr)
                .and_then(|a| a.g.as_ref())
            else {
                new_nodes.push(node.clone());
                continue;
            };

            crate::debug_println!(
                "[if-inline] {} taking {branch_attr} (cond={cond})",
                node.name
            );
            let prefix = if node.name.is_empty() {
                format!("{}_if", node.output.first().cloned().unwrap_or_default())
            } else {
                node.name.clone()
            };
            // Names produced inside the branch get prefixed; everything else
            // is an outer-scope capture and keeps its name.
            let mut rename: HashMap<String, String> = HashMap::new();
            for t in branch.initializer.as_slice() {
                rename.insert(t.name.clone(), format!("{prefix}::{}", t.name));
            }
            for n in branch.node.as_slice() {
                for out in n.output.as_slice() {
                    if !out.is_empty() {
                        rename.insert(out.clone(), format!("{prefix}::{out}"));
                    }
                }
            }
            // Branch graph outputs feed the If node's outputs directly.
            for (branch_out, if_out) in branch.output.iter().zip(node.output.as_slice()) {
                rename.insert(branch_out.name.clone(), if_out.clone());
            }

            for t in branch.initializer.as_slice() {
                let mut t = t.clone();
                if let Some(new_name) = rename.get(&t.name) {
                    t.name = new_name.clone();
                }
                new_initializers.push(t);
            }
            let mut produced_outputs: HashSet<String> = HashSet::new();
            for n in branch.node.as_slice() {
                let mut n = n.clone();
                if !n.name.is_empty() {
                    n.name = format!("{prefix}::{}", n.name);
                }
                for i in n.input.iter_mut() {
                    if let Some(new_name) = rename.get(i) {
                        *i = new_name.clone();
                    }
                }
                for o in n.output.iter_mut() {
                    if let Some(new_name) = rename.get(o) {
                        *o = new_name.clone();
                    }
                    produced_outputs.insert(o.clone());
                }
                new_nodes.push(n);
            }
            // A branch output that is an outer capture or initializer needs an
            // explicit passthrough to the If output name.
            for (branch_out, if_out) in branch.output.iter().zip(node.output.as_slice()) {
                if !produced_outputs.contains(if_out.as_str()) {
                    let src = rename
                        .get(&branch_out.name)
                        .filter(|n| *n != if_out)
                        .cloned()
                        .unwrap_or_else(|| branch_out.name.clone());
                    new_nodes.push(crate::protos::onnx::NodeProto {
                        op_type: "Identity".to_string(),
                        name: format!("{prefix}::passthrough_{if_out}"),
                        input: vec![src],
                        output: vec![if_out.clone()],
                        ..Default::default()
                    });
                }
            }
            changed = true;
        }

        if !changed {
            return;
        }
        if let Some(g) = model.graph.as_mut() {
            g.node = new_nodes;
            g.initializer.extend(new_initializers);
            // Drop the now-dead condition chain the splice orphaned.
            prune_dead_nodes(g);
        }
    }
}

/// Lower an in-memory ONNX [`ModelProto`] to [`MLGraphBuilder`] and validate with ORT `build()`.
pub fn convert_model_proto(
    model: ModelProto,
    options: &ConvertOptions,
) -> Result<ValidatedGraph<'static>, OnnxError> {
    convert_model(model, options)
}

/// Lower ONNX to [`MLGraphBuilder`] and validate with ORT `build()`.
pub(crate) fn convert_model(
    mut model: ModelProto,
    options: &ConvertOptions,
) -> Result<ValidatedGraph<'static>, OnnxError> {
    if options.zero_fill_missing_external_data {
        if let Some(graph) = model.graph.as_mut() {
            zero_fill_external_tensors(graph)?;
        }
    }
    pin_graph_inputs(&mut model, &options.pinned_inputs)?;
    if options.optimize {
        crate::debug_println!("Running constant folding...");
        let evaluators = crate::onnx::constant_folding::evaluators::get_evaluators();
        let nodes_folded =
            crate::onnx::constant_folding::fold_constants_in_model(&mut model, &evaluators)?;
        crate::debug_println!("Constant folding: {} nodes folded", nodes_folded);
    }
    inline_constant_ifs(&mut model, options);
    if !options.pinned_inputs.is_empty() {
        prune_empty_graph_outputs(&mut model);
        prune_unused_graph_inputs(&mut model);
    }
    // Move, don't clone: `model` carries every weight tensor.
    let mut converter = OnnxConverter::new(model)?;
    converter.extract_metadata()?;

    let mut context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, false))
        .map_err(|e| OnnxError::ShapeInference(format!("MLContext::create failed: {e}")))?;

    let mut ml_builder = MLGraphBuilder::new(&mut context).map_err(map_rustnn_error)?;
    let mut onnx_builder = OnnxBuilder::new(&mut ml_builder);

    converter.convert_with_builder(&mut onnx_builder, options)?;

    // Every constant now lives in the WebNN graph; the proto's copy of the
    // weights would otherwise stay resident through the whole backend compile
    // in `finish_build` (for real models that is the full weight volume). Only
    // the graph's output names are read from the proto below.
    if let Some(graph) = converter.model.graph.as_mut() {
        strip_initializer_payloads(graph);
    }

    let onnx_graph = converter
        .model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::ProtobufError("Missing graph in model".to_string()))?;

    let mut outputs: HashMap<String, MLOperand> = HashMap::new();
    for output in onnx_graph.output.as_slice() {
        // Sequences only exist as lowered elements; one escaping to a graph
        // output has no WebNN representation.
        if onnx_builder
            .sequence_element_count(output.name.as_str())
            .is_some()
        {
            return Err(OnnxError::unsupported_op(
                "SplitToSequence(sequence graph output)",
                output.name.clone(),
            ));
        }
        let op = onnx_builder.output_operand(output.name.as_str())?;
        let output_key = onnx_builder.build_output_key(output.name.as_str());
        outputs.insert(output_key, op);
    }
    let output_refs: HashMap<&str, MLOperand> =
        outputs.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    let graph = onnx_builder.finish_build(output_refs)?;

    Ok(ValidatedGraph { context, graph })
}

#[cfg(test)]
mod external_data_tests {
    use super::*;
    use crate::onnx::test_models::prelude::*;

    fn model_with_missing_external_weight() -> ModelProto {
        let mut weight = f32_init("w", &[2, 3], &[]);
        weight.data_location = 1; // EXTERNAL
        weight
            .external_data
            .push(crate::protos::onnx::StringStringEntryProto {
                key: "location".to_string(),
                value: "does_not_exist.bin".to_string(),
            });
        model(
            17,
            graph(
                "ext",
                vec![f32_input("x", &[2, 3])],
                vec![f32_output("y", &[2, 3])],
                vec![node("Add", "add", &["x", "w"], &["y"], &[])],
                vec![weight],
            ),
        )
    }

    #[test]
    fn tensor_byte_len_uses_dims_and_element_size() {
        assert_eq!(tensor_byte_len(&f32_init("a", &[2, 3], &[])), Some(24));
        assert_eq!(tensor_byte_len(&i64_init("b", &[4], &[])), Some(32));
        assert_eq!(tensor_byte_len(&u8_init("c", &[], &[])), Some(1));
        assert_eq!(tensor_byte_len(&f16_init("d", &[5], &[])), Some(10));
    }

    #[test]
    fn missing_external_data_is_zero_filled_only_when_allowed() {
        let dir = std::env::temp_dir();
        let onnx_path = dir.join("skeleton_model.onnx");

        let mut model = model_with_missing_external_weight();
        let err = load_external_tensor_data(&mut model, &onnx_path, false).unwrap_err();
        assert!(err.to_string().contains("does_not_exist.bin"), "{err}");

        let mut model = model_with_missing_external_weight();
        load_external_tensor_data(&mut model, &onnx_path, true).unwrap();
        let w = &model.graph.as_ref().unwrap().initializer[0];
        assert_eq!(w.data_location, 0);
        assert!(w.external_data.is_empty());
        assert_eq!(w.raw_data, vec![0u8; 24]);

        // The zero-filled skeleton still converts and builds in ORT.
        let options = ConvertOptions {
            zero_fill_missing_external_data: true,
            ..ConvertOptions::default()
        };
        convert_model(model, &options).expect("skeleton conversion should succeed");
    }
}

#[cfg(test)]
mod pin_input_tests {
    use super::*;
    use crate::onnx::test_models::prelude::*;

    fn if_model_with_bool_input() -> ModelProto {
        model(
            17,
            graph(
                "pin_graph",
                vec![f32_input("x", &[2]), bool_input("flag", &[1])],
                vec![f32_output("y", &[2])],
                vec![node("Add", "add", &["x", "x"], &["y"], &[])],
                vec![],
            ),
        )
    }

    #[test]
    fn parse_pinned_input_accepts_bools_and_integers() {
        assert_eq!(
            parse_pinned_input("use_cache_branch=false").unwrap(),
            ("use_cache_branch".to_string(), 0)
        );
        assert_eq!(
            parse_pinned_input(" flag = true ").unwrap(),
            ("flag".to_string(), 1)
        );
        assert_eq!(parse_pinned_input("n=3").unwrap(), ("n".to_string(), 3));
        assert!(parse_pinned_input("flag").is_err());
        assert!(parse_pinned_input("flag=maybe").is_err());
    }

    #[test]
    fn pin_graph_inputs_turns_input_into_initializer() {
        let mut m = if_model_with_bool_input();
        let pinned = HashMap::from([("flag".to_string(), 1i64)]);
        pin_graph_inputs(&mut m, &pinned).unwrap();
        let initializers_after_pin = {
            let g = m.graph.as_ref().unwrap();
            assert!(g.input.iter().all(|vi| vi.name != "flag"));
            let init = g.initializer.iter().find(|t| t.name == "flag").unwrap();
            assert_eq!(init.data_type, TensorProto_DataType::Bool as i32);
            assert_eq!(init.dims, vec![1]);
            assert_eq!(init.raw_data, vec![1u8]);
            g.initializer.len()
        };

        // The input is gone now, so pinning it again is an error.
        assert!(pin_graph_inputs(&mut m, &pinned).is_err());
        assert_eq!(
            m.graph.as_ref().unwrap().initializer.len(),
            initializers_after_pin
        );

        let unknown = HashMap::from([("nope".to_string(), 1i64)]);
        assert!(pin_graph_inputs(&mut m, &unknown).is_err());
    }

    #[test]
    fn pin_graph_inputs_rejects_dynamic_dims() {
        let mut m = model(
            17,
            graph(
                "pin_dyn",
                vec![tensor_input(
                    "ids",
                    TensorProto_DataType::Int64 as i32,
                    &[-1],
                )],
                vec![f32_output("y", &[2])],
                vec![],
                vec![],
            ),
        );
        // Mark the dim symbolic.
        if let Some(TypeProtoValue::TensorType(tt)) = m.graph.as_mut().unwrap().input[0]
            .r#type
            .as_mut()
            .and_then(|t| t.value.as_mut())
        {
            tt.shape.as_mut().unwrap().dim[0].value =
                Some(DimensionValue::DimParam("n".to_string()));
        }
        let pinned = HashMap::from([("ids".to_string(), 7i64)]);
        assert!(pin_graph_inputs(&mut m, &pinned).is_err());
    }

    #[test]
    fn prune_dead_nodes_drops_dead_chain_and_keeps_input_initializers() {
        use crate::onnx::test_models::prelude::*;
        let mut m = model(
            17,
            graph(
                "prune_dead",
                vec![f32_input("x", &[2, 3]), i64_input("kept_in", &[1])],
                vec![f32_output("y", &[2, 3])],
                vec![
                    node("Add", "live", &["x", "x"], &["y"], &[]),
                    // Dead chain: nothing consumes `cond`.
                    node("Shape", "shape", &["x"], &["xs"], &[]),
                    node(
                        "Gather",
                        "gather",
                        &["xs", "idx"],
                        &["d0"],
                        &[attr_int("axis", 0)],
                    ),
                    node("Equal", "eq", &["d0", "two"], &["cond"], &[]),
                ],
                vec![
                    i64_init("idx", &[], &[0]),
                    i64_init("two", &[], &[2]),
                    // Doubles as a graph input; must survive even though only
                    // dead nodes referenced it.
                    i64_init("kept_in", &[1], &[5]),
                ],
            ),
        );
        prune_dead_nodes(m.graph.as_mut().unwrap());
        let g = m.graph.as_ref().unwrap();
        let names: Vec<&str> = g.node.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names, vec!["live"]);
        assert!(g
            .initializer
            .iter()
            .all(|t| t.name != "idx" && t.name != "two"));
        assert!(g.initializer.iter().any(|t| t.name == "kept_in"));
    }

    #[test]
    fn prune_dead_nodes_keeps_producers_captured_by_if_subgraphs() {
        use crate::onnx::test_models::prelude::*;
        use crate::protos::onnx::AttributeProto;

        // `captured` is produced at top level but consumed ONLY implicitly
        // inside the If's branch bodies - it must survive pruning.
        let branch = |tag: &str| {
            graph(
                &format!("{tag}_g"),
                vec![],
                vec![f32_output("branch_out", &[2, 3])],
                vec![node(
                    "Identity",
                    &format!("{tag}_id"),
                    &["captured"],
                    &["branch_out"],
                    &[],
                )],
                vec![],
            )
        };
        let mut if_node = node("If", "test_if", &["cond"], &["y"], &[]);
        for (name, g) in [
            ("then_branch", branch("then")),
            ("else_branch", branch("else")),
        ] {
            if_node.attribute.push(AttributeProto {
                name: name.to_string(),
                r#type: 5, // GRAPH
                g: Some(g),
                ..Default::default()
            });
        }
        let mut m = model(
            17,
            graph(
                "prune_captures",
                vec![f32_input("x", &[2, 3]), bool_input("cond", &[])],
                vec![f32_output("y", &[2, 3])],
                vec![
                    node("Add", "capture_producer", &["x", "x"], &["captured"], &[]),
                    if_node,
                ],
                vec![],
            ),
        );
        prune_dead_nodes(m.graph.as_mut().unwrap());
        let g = m.graph.as_ref().unwrap();
        let names: Vec<&str> = g.node.iter().map(|n| n.name.as_str()).collect();
        assert!(
            names.contains(&"capture_producer"),
            "producer consumed only inside If branches was pruned: {names:?}"
        );
        assert!(names.contains(&"test_if"));
    }

    #[test]
    fn prune_empty_graph_outputs_drops_zero_size_constants() {
        let mut m = model(
            17,
            graph(
                "prune_out",
                vec![f32_input("x", &[2])],
                vec![f32_output("y", &[2]), f32_output("dummy", &[0, 4])],
                vec![
                    node("Add", "add", &["x", "x"], &["y"], &[]),
                    node(
                        "Constant",
                        "dummy_const",
                        &[],
                        &["dummy"],
                        &[attr_tensor("value", f32_init("dummy_val", &[0, 4], &[]))],
                    ),
                ],
                vec![],
            ),
        );
        prune_empty_graph_outputs(&mut m);
        let g = m.graph.as_ref().unwrap();
        assert_eq!(
            g.output.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
            vec!["y"]
        );
        assert!(g.node.iter().all(|n| n.op_type != "Constant"));
    }

    #[test]
    fn prune_unused_graph_inputs_keeps_consumed_and_passthrough_inputs() {
        let mut m = model(
            17,
            graph(
                "prune_in",
                vec![
                    f32_input("x", &[2]),
                    f32_input("unused", &[2]),
                    f32_input("passthrough", &[2]),
                ],
                vec![f32_output("y", &[2]), f32_output("passthrough", &[2])],
                vec![node("Add", "add", &["x", "x"], &["y"], &[])],
                vec![],
            ),
        );
        prune_unused_graph_inputs(&mut m);
        let names: Vec<&str> = m
            .graph
            .as_ref()
            .unwrap()
            .input
            .iter()
            .map(|vi| vi.name.as_str())
            .collect();
        assert_eq!(names, vec!["x", "passthrough"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_options_default() {
        let options = ConvertOptions::default();
        assert!(!options.optimize);
        assert!(options.free_dim_overrides.is_empty());
    }

    #[test]
    fn test_sanitize_identifier_replaces_colons() {
        assert_eq!(sanitize_identifier("foo::bar"), "foo__bar");
        assert_eq!(sanitize_identifier("foo:bar"), "foo_bar");
    }

    #[test]
    fn test_sanitize_identifier_replaces_dots() {
        assert_eq!(sanitize_identifier("encoder.block.0"), "encoder_block_0");
        assert_eq!(
            sanitize_identifier("model.layer.weight"),
            "model_layer_weight"
        );
        assert_eq!(sanitize_identifier("a.b.c"), "a_b_c");
    }

    #[test]
    fn test_sanitize_identifier_replaces_combined() {
        // Test combinations of :: : and .
        assert_eq!(
            sanitize_identifier("module::class:method.field"),
            "module__class_method_field"
        );
        assert_eq!(
            sanitize_identifier("encoder.attention::output:dense"),
            "encoder_attention__output_dense"
        );
    }

    #[test]
    fn test_sanitize_identifier_no_change() {
        // Identifiers that don't need sanitization
        assert_eq!(sanitize_identifier("simple_name"), "simple_name");
        assert_eq!(sanitize_identifier("CamelCase"), "CamelCase");
        assert_eq!(sanitize_identifier("name123"), "name123");
    }

    #[test]
    fn test_inline_bytes_encoding_for_i64_values() {
        // Test the inline bytes encoding logic used for non-scalar constants
        // This simulates what happens when Range or similar ops produce constant arrays
        let values: Vec<i64> = vec![0, 1, 2, 3, 4];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Verify byte length
        assert_eq!(bytes.len(), 40); // 5 values * 8 bytes each

        // Verify first value (0)
        let first_bytes: [u8; 8] = bytes[0..8].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(first_bytes), 0);

        // Verify last value (4)
        let last_bytes: [u8; 8] = bytes[32..40].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(last_bytes), 4);
    }

    #[test]
    fn test_inline_bytes_encoding_single_value() {
        // Test single value encoding
        let values: Vec<i64> = vec![42];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(bytes.len(), 8);
        let decoded: [u8; 8] = bytes.try_into().unwrap();
        assert_eq!(i64::from_le_bytes(decoded), 42);
    }

    #[test]
    fn test_inline_bytes_encoding_negative_values() {
        // Test with negative values (important for Range with negative delta)
        let values: Vec<i64> = vec![5, 4, 3, 2, 1, 0, -1, -2];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(bytes.len(), 64); // 8 values * 8 bytes each

        // Verify a negative value
        let neg_bytes: [u8; 8] = bytes[56..64].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(neg_bytes), -2);
    }

    #[test]
    fn test_inline_bytes_encoding_large_values() {
        // Test with large i64 values
        let values: Vec<i64> = vec![i64::MAX, i64::MIN, 0];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(bytes.len(), 24);

        // Verify MAX value
        let max_bytes: [u8; 8] = bytes[0..8].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(max_bytes), i64::MAX);

        // Verify MIN value
        let min_bytes: [u8; 8] = bytes[8..16].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(min_bytes), i64::MIN);
    }

    #[test]
    fn test_collects_all_unsupported_ops() {
        use crate::protos::onnx::{GraphProto, ModelProto, NodeProto, OperatorSetIdProto};

        let model = ModelProto {
            opset_import: vec![OperatorSetIdProto {
                version: 17,
                ..Default::default()
            }],
            graph: Some(GraphProto {
                node: vec![
                    NodeProto {
                        op_type: "If".to_string(),
                        name: "if_node".to_string(),
                        ..Default::default()
                    },
                    NodeProto {
                        op_type: "Loop".to_string(),
                        name: "loop_node".to_string(),
                        ..Default::default()
                    },
                    NodeProto {
                        op_type: "Add".to_string(),
                        name: "add_node".to_string(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };

        let err = match convert_model_proto(model, &ConvertOptions::default()) {
            Err(err) => err,
            Ok(_) => panic!("expected unsupported ops error"),
        };
        assert!(err.is_unsupported_op());
        let ops = err.unsupported_ops().expect("unsupported ops payload");
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].op, "If");
        assert_eq!(ops[0].node, "if_node");
        assert_eq!(ops[1].op, "Loop");
        assert_eq!(ops[1].node, "loop_node");
    }

    #[test]
    fn test_convert_preserves_dynamic_input_dim_without_override() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{GraphProto, ModelProto, TensorShapeProto, ValueInfoProto};

        let dim_batch = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "batch_size".to_string(),
            )),
            denotation: String::new(),
        };
        let dim_seq = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
            denotation: String::new(),
        };
        let shape = TensorShapeProto {
            dim: vec![dim_batch, dim_seq],
        };

        let tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Int64.into(),
            shape: Some(shape),
        };
        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let input_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let output_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![input_vi],
                output: vec![output_vi],
                ..Default::default()
            }),
            ..Default::default()
        };

        convert_model(
            model,
            &ConvertOptions {
                experimental_dynamic_inputs: true,
                ..ConvertOptions::default()
            },
        )
        .expect("ORT build should succeed for experimental dynamic inputs");
    }

    #[test]
    fn test_convert_rejects_dynamic_input_dim_without_flag() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{GraphProto, ModelProto, TensorShapeProto, ValueInfoProto};

        let dim_batch = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "unknown_dim".to_string(),
            )),
            denotation: String::new(),
        };
        let dim_seq = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
            denotation: String::new(),
        };
        let shape = TensorShapeProto {
            dim: vec![dim_batch, dim_seq],
        };

        let tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Int64.into(),
            shape: Some(shape),
        };
        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let input_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let output_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![input_vi],
                output: vec![output_vi],
                ..Default::default()
            }),
            ..Default::default()
        };

        let msg = match convert_model(model, &ConvertOptions::default()) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("should require overrides or flag"),
        };
        assert!(msg.contains("override-dim"));
        assert!(msg.contains("experimental-dynamic-inputs"));
    }
}
