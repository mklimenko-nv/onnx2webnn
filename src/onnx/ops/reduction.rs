/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Reduction operators: Reduce*, ArgMin, ArgMax, CumSum

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{
    normalize_axes_best_effort, normalize_axis_best_effort, ConversionContext, ConversionResult,
    OpHandler,
};
use crate::protos::onnx::{NodeProto, TensorProto, TensorProto_DataType};
use rustnn::operator_options::{MLArgMinMaxOptions, MLCumulativeSumOptions, MLReduceOptions};
use std::collections::HashMap;

pub struct ReductionHandler;

impl OpHandler for ReductionHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "ReduceMean"
                | "ReduceSum"
                | "ReduceMax"
                | "ReduceMin"
                | "ReduceL1"
                | "ReduceL2"
                | "ReduceLogSum"
                | "ReduceLogSumExp"
                | "ReduceProd"
                | "ReduceSumSquare"
                | "ArgMin"
                | "ArgMax"
                | "CumSum"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.clone()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "ReduceMean" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_mean_with_options(i, o)
            }),
            "ReduceSum" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_sum_with_options(i, o)
            }),
            "ReduceMax" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_max_with_options(i, o)
            }),
            "ReduceMin" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_min_with_options(i, o)
            }),
            "ReduceL1" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_l1_with_options(i, o)
            }),
            "ReduceL2" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_l2_with_options(i, o)
            }),
            "ReduceLogSum" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_log_sum_with_options(i, o)
            }),
            "ReduceLogSumExp" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_log_sum_exp_with_options(i, o)
            }),
            "ReduceProd" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_product_with_options(i, o)
            }),
            "ReduceSumSquare" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_sum_square_with_options(i, o)
            }),
            "ArgMin" => self.convert_arg_min_max(node, &node_name, context, b, |g, i, axis, o| {
                g.arg_min_with_options(i, axis, o)
            }),
            "ArgMax" => self.convert_arg_min_max(node, &node_name, context, b, |g, i, axis, o| {
                g.arg_max_with_options(i, axis, o)
            }),
            "CumSum" => self.convert_cum_sum(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

impl ReductionHandler {
    fn convert_reduce(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
        emit: impl FnOnce(
            &mut rustnn::mlgraphbuilder::MLGraphBuilder<'_, '_>,
            rustnn::mlcontext::MLOperand,
            MLReduceOptions,
        )
            -> Result<rustnn::mlcontext::MLOperand, rustnn::error::GraphBuilderError>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects at least 1 input",
                node.op_type
            )));
        }

        let mut axes: Option<Vec<i64>> = None;
        let mut keepdims = 1i64;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "axes" if !attr.ints.is_empty() => axes = Some(attr.ints.clone()),
                "keepdims" => keepdims = attr.i,
                _ => {}
            }
        }
        // Opset 18+ moved reduce axes from an attribute to a constant input.
        if axes.is_none() {
            if let Some(axes_name) = inputs.get(1).filter(|n| !n.is_empty()) {
                axes = context
                    .const_values
                    .get(axes_name.as_str())
                    .cloned()
                    .filter(|v| !v.is_empty());
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;

        let axes_u32 = axes
            .map(|axes_values| {
                let normalized = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
                    normalize_axes_best_effort(&axes_values, rank)
                } else {
                    axes_values
                };
                if normalized.iter().any(|&a| a < 0) {
                    return Err(OnnxError::InvalidShape(format!(
                        "{} '{node_name}' has negative reduce axes {normalized:?} but the \
                         input rank is unknown",
                        node.op_type
                    )));
                }
                Ok(normalized.into_iter().map(|a| a as u32).collect::<Vec<_>>())
            })
            .transpose()?;

        let opts = MLReduceOptions {
            label: output_name.clone(),
            axes: axes_u32,
            keep_dimensions: keepdims != 0,
        };
        let out = emit(b.builder, input, opts).map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_arg_min_max(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
        emit: impl FnOnce(
            &mut rustnn::mlgraphbuilder::MLGraphBuilder<'_, '_>,
            rustnn::mlcontext::MLOperand,
            u32,
            MLArgMinMaxOptions,
        )
            -> Result<rustnn::mlcontext::MLOperand, rustnn::error::GraphBuilderError>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects at least 1 input",
                node.op_type
            )));
        }

        let mut axis = 0i64;
        let mut keepdims = 1i64;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "axis" => axis = attr.i,
                "keepdims" if attr.i != 0 => keepdims = attr.i,
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank) as u32
        } else {
            axis as u32
        };

        let opts = MLArgMinMaxOptions {
            label: output_name.clone(),
            keep_dimensions: keepdims != 0,
            ..Default::default()
        };
        let out = emit(b.builder, input, axis, opts).map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_cum_sum(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "CumSum expects at least 1 input".to_string(),
            ));
        }

        let mut exclusive = false;
        let mut reversed = false;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "exclusive" if attr.i != 0 => exclusive = true,
                "reverse" if attr.i != 0 => reversed = true,
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let axis = if inputs.len() > 1 && !inputs[1].is_empty() {
            read_scalar_i64(&inputs[1], context.initializers, context.const_values)?
        } else {
            return Err(OnnxError::InvalidShape(
                "CumSum requires a constant axis input".to_string(),
            ));
        };
        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank) as u32
        } else {
            axis as u32
        };

        let opts = MLCumulativeSumOptions {
            label: output_name.clone(),
            exclusive,
            reversed,
        };
        let out = b
            .builder
            .cumulative_sum_with_options(input, axis, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }
}

fn read_scalar_i64(
    name: &str,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Result<i64, OnnxError> {
    if let Some(vals) = const_values.get(name) {
        return vals.first().copied().ok_or_else(|| {
            OnnxError::InvalidShape(format!("constant tensor '{name}' has no scalar value"))
        });
    }
    if let Some(t) = initializers.get(name) {
        let vals = read_int64_tensor_proto(t).ok_or_else(|| {
            OnnxError::InvalidShape(format!("initializer '{name}' has no integer data"))
        })?;
        return vals.first().copied().ok_or_else(|| {
            OnnxError::InvalidShape(format!("initializer '{name}' has no scalar value"))
        });
    }
    Err(OnnxError::InvalidShape(format!(
        "expected constant scalar tensor '{name}'"
    )))
}

fn read_int64_tensor_proto(t: &TensorProto) -> Option<Vec<i64>> {
    if !t.raw_data.is_empty() {
        if t.data_type == TensorProto_DataType::Int32 as i32 {
            return Some(
                t.raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                    .collect(),
            );
        }
        return Some(
            t.raw_data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
        );
    }
    if !t.int64_data.is_empty() {
        return Some(t.int64_data.clone());
    }
    if !t.int32_data.is_empty() {
        return Some(t.int32_data.iter().map(|&v| v as i64).collect());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, NodeProto};

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        });
    }

    fn add_ints_attribute(node: &mut NodeProto, name: &str, values: Vec<i64>) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            ints: values,
            ..Default::default()
        });
    }

    #[test]
    fn test_reduction_handler_supports() {
        let handler = ReductionHandler;
        assert!(handler.supports("ReduceMean"));
        assert!(handler.supports("ReduceSum"));
    }

    #[test]
    fn test_convert_reduce_mean() {
        let handler = ReductionHandler;
        let mut node = create_test_node("ReduceMean", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![1, 2]);
        add_int_attribute(&mut node, "keepdims", 1);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 2, 3, 4]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = crate::onnx::ops::ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };
        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
    }

    #[test]
    fn test_convert_reduce_sum() {
        let handler = ReductionHandler;
        let node = create_test_node("ReduceSum", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
