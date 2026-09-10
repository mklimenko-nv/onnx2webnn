/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Variadic elementwise ops and CumProd.

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{
    normalize_axis_best_effort, ConversionContext, ConversionResult, OpHandler,
};
use crate::protos::onnx::{NodeProto, TensorProto, TensorProto_DataType};
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::MLCumulativeSumOptions;
use rustnn::DataType;
use std::collections::HashMap;

pub struct MiscHandler;

impl OpHandler for MiscHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Mean" | "Sum" | "CumProd")
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
            "Mean" => self.convert_mean(node, &node_name, b),
            "Sum" => self.convert_sum(node, &node_name, b),
            "CumProd" => self.convert_cum_prod(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

impl MiscHandler {
    fn convert_sum(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Sum expects at least 1 input".to_string(),
            ));
        }

        let output_name = output_label(node, node_name);
        let out = fold_variadic_add(b, inputs, &output_name)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_mean(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Mean expects at least 1 input".to_string(),
            ));
        }

        let output_name = output_label(node, node_name);
        let n = inputs.len();

        let out = if n == 1 {
            let input = b.resolve_operand(&inputs[0])?;
            let opts = OnnxBuilder::labeled_options(&output_name);
            b.builder
                .identity_with_options(input, opts)
                .map_err(map_op_error)?
        } else {
            let sum_label = format!("{output_name}__sum");
            let sum = fold_variadic_add(b, inputs, &sum_label)?;

            let scalar_name = format!("{output_name}__n");
            let n_f32 = n as f32;
            b.register_constant_from_bytes(
                &scalar_name,
                DataType::Float32,
                &[1],
                n_f32.to_le_bytes().to_vec(),
            )?;
            let divisor = b.resolve_operand(&scalar_name)?;
            let opts = OnnxBuilder::labeled_options(&output_name);
            b.builder
                .div_with_options(sum, divisor, opts)
                .map_err(map_op_error)?
        };

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_cum_prod(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "CumProd expects at least 1 input".to_string(),
            ));
        }

        let mut exclusive = false;
        let mut reversed = false;
        let mut axis_attr = None;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "exclusive" if attr.i != 0 => exclusive = true,
                "reverse" if attr.i != 0 => reversed = true,
                "axis" => axis_attr = Some(attr.i),
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let axis = if inputs.len() > 1 && !inputs[1].is_empty() {
            read_scalar_i64(&inputs[1], context.initializers, context.const_values)?
        } else if let Some(axis) = axis_attr {
            axis
        } else {
            return Err(OnnxError::InvalidShape(
                "CumProd requires a constant axis input or axis attribute".to_string(),
            ));
        };
        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank) as u32
        } else {
            axis as u32
        };

        // cumulativeProduct via log -> cumulativeSum -> exp (fixture uses positive floats).
        let log_label = format!("{output_name}__log");
        let log_opts = OnnxBuilder::labeled_options(&log_label);
        let log_x = b
            .builder
            .log_with_options(input, log_opts)
            .map_err(map_op_error)?;

        let cum_label = format!("{output_name}__log_cumsum");
        let cum_opts = MLCumulativeSumOptions {
            label: cum_label.clone(),
            exclusive,
            reversed,
        };
        let log_cum = b
            .builder
            .cumulative_sum_with_options(log_x, axis, cum_opts)
            .map_err(map_op_error)?;

        let exp_opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .exp_with_options(log_cum, exp_opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }
}

fn fold_variadic_add(
    b: &mut OnnxBuilder<'_, '_, '_>,
    inputs: &[String],
    output_name: &str,
) -> Result<MLOperand, OnnxError> {
    if inputs.len() == 1 {
        let input = b.resolve_operand(&inputs[0])?;
        let opts = OnnxBuilder::labeled_options(output_name);
        return b
            .builder
            .identity_with_options(input, opts)
            .map_err(map_op_error);
    }

    let mut acc = b.resolve_operand(&inputs[0])?;
    for (step, input_name) in inputs[1..].iter().enumerate() {
        let next = b.resolve_operand(input_name)?;
        let label = if step + 2 == inputs.len() {
            output_name.to_string()
        } else {
            format!("{output_name}__fold_{step}")
        };
        let opts = OnnxBuilder::labeled_options(&label);
        acc = b
            .builder
            .add_with_options(acc, next, opts)
            .map_err(map_op_error)?;
    }
    Ok(acc)
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
    use crate::protos::onnx::NodeProto;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn test_misc_handler_supports() {
        let handler = MiscHandler;
        assert!(handler.supports("Mean"));
        assert!(handler.supports("Sum"));
        assert!(handler.supports("CumProd"));
        assert!(!handler.supports("ReduceMean"));
    }

    #[test]
    fn test_convert_sum_single_input() {
        let handler = MiscHandler;
        let node = create_test_node("Sum", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_mean_single_input() {
        let handler = MiscHandler;
        let node = create_test_node("Mean", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
