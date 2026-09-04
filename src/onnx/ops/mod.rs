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

// Operator handler trait and registry

use crate::onnx::convert::{OnnxError, UnsupportedOpEntry};
use crate::protos::onnx::{NodeProto, TensorProto};
use std::collections::HashMap;
use std::sync::OnceLock;

pub mod activation;
pub mod attention;
pub mod comparison;
pub mod conditional;
pub mod conv;
pub mod conversion;
pub mod einsum;
pub mod elementwise;
pub mod matmul;
pub mod misc;
pub mod moe;
pub mod normalization;
pub mod pad;
pub mod pool;
pub mod reduction;
pub mod reshape;
pub mod resize;
pub mod rnn;
pub mod scatter;
pub mod utility;

use activation::ActivationHandler;
use attention::AttentionHandler;
use comparison::ComparisonHandler;
use conditional::ConditionalHandler;
use conv::ConvHandler;
use conversion::ConversionHandler;
use einsum::EinsumHandler;
use elementwise::ElementwiseHandler;
use matmul::MatMulHandler;
use misc::MiscHandler;
use moe::MoeHandler;
use normalization::NormalizationHandler;
use pad::PadHandler;
use pool::PoolHandler;
use reduction::ReductionHandler;
use reshape::ReshapeHandler;
use resize::ResizeHandler;
use rnn::RnnHandler;
use scatter::ScatterHandler;
use utility::UtilityHandler;

/// Context for operator conversion
pub struct ConversionContext<'a> {
    /// Map of initializer names to TensorProto (for resolving constant shapes)
    pub initializers: &'a HashMap<String, &'a TensorProto>,
    /// Map of value names to their shapes (for shape inference)
    pub value_shapes: &'a HashMap<String, Vec<i64>>,
    /// Map of value names to shape dimensions preserving ONNX dim_param where available.
    pub value_shape_dims: &'a HashMap<String, Vec<rustnn::graph::Dimension>>,
    /// Map of value names to constant integer contents (for const folding)
    pub const_values: &'a HashMap<String, Vec<i64>>,
    /// Map of ONNX value names to WebNN value identifiers
    pub value_ids: &'a HashMap<String, String>,
    /// Map of value names to data types
    pub value_types: &'a HashMap<String, rustnn::DataType>,
}

impl<'a> ConversionContext<'a> {
    pub fn resolve_input(&self, name: &str) -> String {
        if let Some(mapped) = self.value_ids.get(name) {
            return mapped.clone();
        }

        let sanitized = crate::onnx::convert::sanitize_identifier(name);
        if let Some(mapped) = self.value_ids.get(&sanitized) {
            return mapped.clone();
        }

        sanitized
    }

    pub fn resolve_shape(&self, name: &str) -> Option<&Vec<i64>> {
        let sanitized = crate::onnx::convert::sanitize_identifier(name);
        let trimmed = name.trim_start_matches('/');
        self.value_shapes
            .get(name)
            .or_else(|| self.value_shapes.get(&sanitized))
            .or_else(|| self.value_shapes.get(trimmed))
    }

    pub fn input_rank(&self, name: &str) -> Option<usize> {
        self.resolve_shape(name).map(|s| s.len())
    }
}

pub fn normalize_axis(axis: i64, rank: usize) -> Result<i64, OnnxError> {
    let rank_i64 = rank as i64;
    let normalized = if axis < 0 { axis + rank_i64 } else { axis };
    if normalized < 0 || normalized >= rank_i64 {
        return Err(OnnxError::InvalidShape(format!(
            "axis {} is out of bounds for rank {}",
            axis, rank
        )));
    }
    Ok(normalized)
}

pub fn normalize_axes(axes: &[i64], rank: usize) -> Result<Vec<i64>, OnnxError> {
    axes.iter().map(|&a| normalize_axis(a, rank)).collect()
}

pub fn normalize_axis_best_effort(axis: i64, rank: usize) -> i64 {
    normalize_axis(axis, rank).unwrap_or(axis)
}

pub fn normalize_axes_best_effort(axes: &[i64], rank: usize) -> Vec<i64> {
    axes.iter()
        .map(|&a| normalize_axis_best_effort(a, rank))
        .collect()
}

pub fn empty_value_shape_dims() -> &'static HashMap<String, Vec<rustnn::graph::Dimension>> {
    static EMPTY: OnceLock<HashMap<String, Vec<rustnn::graph::Dimension>>> = OnceLock::new();
    EMPTY.get_or_init(HashMap::new)
}

/// Results of converting a single ONNX node
#[derive(Default, Debug)]
pub struct ConversionResult {
    /// ONNX output name -> data type (for downstream shape inference)
    pub output_types: HashMap<String, rustnn::DataType>,
}

/// Trait for handling ONNX operator conversion
pub trait OpHandler {
    /// Check if this handler supports the given operator type
    fn supports(&self, op_type: &str) -> bool;

    /// Lower an ONNX node onto [`OnnxBuilder`].
    fn convert<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
        b: &mut crate::onnx::builder::OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError>;
}

/// Registry for operator handlers
pub struct OpRegistry {
    handlers: Vec<Box<dyn OpHandler>>,
}

impl OpRegistry {
    /// Create a new operator registry with all handlers
    pub fn new() -> Self {
        let handlers: Vec<Box<dyn OpHandler>> = vec![
            Box::new(MatMulHandler),
            Box::new(AttentionHandler),
            Box::new(MoeHandler),
            Box::new(EinsumHandler),
            Box::new(ConvHandler),
            Box::new(PoolHandler),
            Box::new(ElementwiseHandler),
            Box::new(MiscHandler),
            Box::new(ComparisonHandler),
            Box::new(ConditionalHandler),
            Box::new(NormalizationHandler),
            Box::new(ReshapeHandler),
            Box::new(PadHandler),
            Box::new(ResizeHandler),
            Box::new(ConversionHandler),
            Box::new(UtilityHandler),
            Box::new(ReductionHandler),
            Box::new(ActivationHandler),
            Box::new(ScatterHandler),
            Box::new(RnnHandler),
        ];

        OpRegistry { handlers }
    }

    /// Returns true if any registered handler claims support for `op_type`.
    ///
    /// This mirrors the dispatch in [`convert_node`], but performs no conversion. It lets the
    /// converter fail fast with a clean [`OnnxError::UnsupportedOps`] before graph setup (input and
    /// initializer registration) can panic on tensor kinds an unsupported op happens to use
    /// (e.g. bool/string initializers).
    ///
    /// **Domain:** matches on `op_type` only today (standard `ai.onnx` operators). When
    /// custom-domain handlers are added, this should take `(domain, op_type)` so the pre-scan in
    /// `convert_with_builder` stays aligned with [`convert_node`].
    pub fn is_supported(&self, op_type: &str) -> bool {
        self.handlers.iter().any(|h| h.supports(op_type))
    }

    /// Collect every graph node whose `op_type` has no registered handler.
    pub fn collect_unsupported_nodes(&self, nodes: &[NodeProto]) -> Vec<UnsupportedOpEntry> {
        nodes
            .iter()
            .filter_map(|node| {
                let op_type = node.op_type.as_str();
                if self.is_supported(op_type) {
                    None
                } else {
                    let node_name = if node.name.is_empty() {
                        "<unnamed>".to_string()
                    } else {
                        node.name.clone()
                    };
                    Some(UnsupportedOpEntry {
                        op: op_type.to_string(),
                        node: node_name,
                    })
                }
            })
            .collect()
    }

    /// Convert an ONNX node using the appropriate handler and apply to the MLGraphBuilder.
    pub fn convert_node<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
        builder: &mut crate::onnx::builder::OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();

        // Zero-size values (an empty Slice, a Reshape of it, ...) have no WebNN
        // representation. When shape inference proves every output empty, mark
        // them like empty optional inputs so consumers such as Concat drop them.
        let outputs: Vec<&String> = node.output.iter().filter(|o| !o.is_empty()).collect();
        if !outputs.is_empty()
            && outputs.iter().all(|o| {
                context
                    .value_shapes
                    .get(o.as_str())
                    .is_some_and(|shape| shape.contains(&0))
            })
        {
            for out in outputs {
                builder.mark_empty_optional(out);
                builder.mark_empty_optional(&crate::onnx::convert::sanitize_identifier(out));
            }
            return Ok(ConversionResult::default());
        }

        for handler in &self.handlers {
            if handler.supports(op_type) {
                let result = handler.convert(node, context, builder)?;
                return Ok(result);
            }
        }

        // No handler found
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "<unnamed>".to_string()
        };

        Err(OnnxError::unsupported_op(op_type, node_name))
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a handler against a minimal CPU ORT builder with float32 inputs registered.
#[cfg(test)]
pub fn convert_with_test_builder(
    handler: &dyn OpHandler,
    node: &NodeProto,
) -> Result<ConversionResult, OnnxError> {
    let initializers = HashMap::new();
    let value_shapes = HashMap::new();
    let const_values = HashMap::new();
    let value_ids = HashMap::new();
    let value_types = HashMap::new();
    let context = ConversionContext {
        initializers: &initializers,
        value_shapes: &value_shapes,
        value_shape_dims: empty_value_shape_dims(),
        const_values: &const_values,
        value_ids: &value_ids,
        value_types: &value_types,
    };
    convert_handler_with_context(handler, node, &context)
}

#[cfg(test)]
fn i64_shape_to_dims(shape: &[i64]) -> Vec<rustnn::graph::Dimension> {
    shape
        .iter()
        .map(|&d| {
            if d >= 0 {
                rustnn::graph::Dimension::Static(d as u32)
            } else {
                rustnn::graph::Dimension::Dynamic(rustnn::graph::DynamicDimension {
                    name: format!("dim_{d}"),
                    max_size: 1,
                })
            }
        })
        .collect()
}

#[cfg(test)]
fn dummy_constant_bytes(dtype: rustnn::DataType, numel: usize) -> Vec<u8> {
    let elem_size = match dtype {
        rustnn::DataType::Float32 => 4,
        rustnn::DataType::Float16 => 2,
        rustnn::DataType::Int32 | rustnn::DataType::Uint32 => 4,
        rustnn::DataType::Int64 | rustnn::DataType::Uint64 => 8,
        rustnn::DataType::Int8 | rustnn::DataType::Uint8 => 1,
        rustnn::DataType::Int4 | rustnn::DataType::Uint4 => 1,
    };
    vec![0u8; numel.saturating_mul(elem_size).max(elem_size)]
}

#[cfg(test)]
fn register_test_operand(
    builder: &mut crate::onnx::builder::OnnxBuilder<'_, '_, '_>,
    context: &ConversionContext,
    name: &str,
) -> Result<(), OnnxError> {
    use crate::onnx::convert::{map_onnx_data_type, sanitize_identifier};
    use rustnn::DataType;

    let sanitized = sanitize_identifier(name);
    if builder.resolve_operand(name).is_ok() {
        return Ok(());
    }

    if let Some(tensor) = context
        .initializers
        .get(name)
        .or_else(|| context.initializers.get(&sanitized))
    {
        let dtype = map_onnx_data_type(tensor.data_type)?;
        let mut shape: Vec<u32> = tensor
            .dims
            .iter()
            .map(|&d| u32::try_from(d.max(0)).unwrap_or(1))
            .collect();
        if shape.is_empty() {
            if !tensor.int64_data.is_empty() {
                shape = vec![tensor.int64_data.len() as u32];
            } else if !tensor.int32_data.is_empty() {
                shape = vec![tensor.int32_data.len() as u32];
            } else if !tensor.float_data.is_empty() {
                shape = vec![tensor.float_data.len() as u32];
            } else {
                shape = vec![1];
            }
        }
        let numel = shape.iter().map(|&d| d as usize).product::<usize>().max(1);
        let bytes = if let Ok(b) = crate::onnx::builder::tensor_proto_to_bytes(tensor) {
            b
        } else {
            dummy_constant_bytes(dtype, numel)
        };
        builder.register_constant_from_bytes(name, dtype, &shape, bytes)?;
        return Ok(());
    }

    let shape = context
        .resolve_shape(name)
        .map(|s| i64_shape_to_dims(s))
        .unwrap_or_else(|| {
            vec![
                rustnn::graph::Dimension::Static(2),
                rustnn::graph::Dimension::Static(2),
            ]
        });
    let dtype = context
        .value_types
        .get(name)
        .or_else(|| context.value_types.get(&sanitized))
        .cloned()
        .unwrap_or(DataType::Float32);
    builder.register_input(name, dtype, &shape)
}

#[cfg(test)]
pub fn convert_handler_with_context(
    handler: &dyn OpHandler,
    node: &NodeProto,
    context: &ConversionContext,
) -> Result<ConversionResult, OnnxError> {
    use crate::onnx::builder::{map_rustnn_error, OnnxBuilder};
    use rustnn::mlcontext::{MLContext, MLContextOptions, MLGraphBuilder, MLPowerPreference};

    let mut ml_context =
        MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, false))
            .map_err(|e| OnnxError::ShapeInference(format!("MLContext::create failed: {e}")))?;
    let mut ml_builder = MLGraphBuilder::new(&mut ml_context).map_err(map_rustnn_error)?;
    let mut builder = OnnxBuilder::new(&mut ml_builder);
    for input in node.input.iter() {
        if input.is_empty() {
            continue;
        }
        register_test_operand(&mut builder, context, input)?;
    }
    handler.convert(node, context, &mut builder)
}
