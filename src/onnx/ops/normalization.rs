/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Normalization operators: BatchNormalization, InstanceNormalization, LayerNormalization,
// GroupNormalization, RMSNormalization, SimplifiedLayerNormalization,
// SkipSimplifiedLayerNormalization, Softmax, LogSoftmax, Hardmax

use crate::onnx::builder::{map_ast_data_type, map_op_error, operand_index, OnnxBuilder};
use crate::onnx::builder_helpers::{
    expand_with_shape, i64_slice_to_mldim, output_label, record_node_output, reshape_with_shape,
};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{
    normalize_axis_best_effort, ConversionContext, ConversionResult, OpHandler,
};
use crate::protos::onnx::NodeProto;
use half::f16;
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLInstanceNormalizationOptions,
    MLLayerNormalizationOptions, MLReduceOptions,
};
use rustnn::DataType;

pub struct NormalizationHandler;

impl OpHandler for NormalizationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "BatchNormalization"
                | "InstanceNormalization"
                | "LayerNormalization"
                | "GroupNormalization"
                | "RMSNormalization"
                | "SimplifiedLayerNormalization"
                | "SkipSimplifiedLayerNormalization"
                | "Softmax"
                | "LogSoftmax"
                | "Hardmax"
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
            "BatchNormalization" => self.convert_batch_normalization(node, &node_name, context, b),
            "InstanceNormalization" => {
                self.convert_instance_normalization(node, &node_name, context, b)
            }
            "LayerNormalization" => self.convert_layer_norm(node, &node_name, context, b),
            "GroupNormalization" => self.convert_group_normalization(node, &node_name, context, b),
            "RMSNormalization" => self.convert_rms_normalization(node, &node_name, context, b),
            "SimplifiedLayerNormalization" => {
                self.convert_rms_normalization(node, &node_name, context, b)
            }
            "SkipSimplifiedLayerNormalization" => {
                self.convert_skip_simplified_layer_normalization(node, &node_name, context, b)
            }
            "Softmax" => self.convert_softmax(node, &node_name, context, b),
            "LogSoftmax" => self.convert_log_softmax(node, &node_name, context, b),
            "Hardmax" => self.convert_hardmax(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

impl NormalizationHandler {
    fn convert_batch_normalization(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 5 {
            return Err(OnnxError::InvalidShape(format!(
                "BatchNormalization expects 5 inputs, got {}",
                inputs.len()
            )));
        }

        let mut epsilon = 1e-5f64;
        let mut axis = 1i64;
        let mut training_mode = 0i64;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "epsilon" if attr.f != 0.0 => epsilon = attr.f as f64,
                "axis" => axis = attr.i,
                "training_mode" => training_mode = attr.i,
                _ => {}
            }
        }
        if training_mode != 0 {
            return Err(OnnxError::InvalidShape(
                "BatchNormalization training_mode != 0 is unsupported".to_string(),
            ));
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let mean = b.resolve_operand(&inputs[3])?;
        let variance = b.resolve_operand(&inputs[4])?;

        let scale = if !inputs[1].is_empty() {
            Some(operand_index(b.resolve_operand(&inputs[1])?))
        } else {
            None
        };
        let bias = if !inputs[2].is_empty() {
            Some(operand_index(b.resolve_operand(&inputs[2])?))
        } else {
            None
        };

        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank) as u32
        } else {
            axis as u32
        };

        let opts = MLBatchNormalizationOptions {
            label: output_name.clone(),
            scale,
            bias,
            axis,
            epsilon,
        };
        let out = b
            .builder
            .batch_normalization_with_options(input, mean, variance, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_instance_normalization(
        &self,
        node: &NodeProto,
        node_name: &str,
        _context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "InstanceNormalization expects at least 1 input".to_string(),
            ));
        }

        let mut epsilon = 1e-5f64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "epsilon" && attr.f != 0.0 {
                epsilon = attr.f as f64;
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let scale = inputs
            .get(1)
            .filter(|name| !name.is_empty())
            .map(|name| b.resolve_operand(name))
            .transpose()?
            .map(operand_index);
        let bias = inputs
            .get(2)
            .filter(|name| !name.is_empty())
            .map(|name| b.resolve_operand(name))
            .transpose()?
            .map(operand_index);

        let opts = MLInstanceNormalizationOptions {
            label: output_name.clone(),
            scale,
            bias,
            epsilon,
            layout: String::new(),
        };
        let out = b
            .builder
            .instance_normalization_with_options(input, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_layer_norm(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "LayerNormalization expects at least 1 input".to_string(),
            ));
        }

        let mut epsilon = 1e-5f64;
        let mut axis = -1i64;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "epsilon" if attr.f != 0.0 => epsilon = attr.f as f64,
                "axis" if attr.i != 0 => axis = attr.i,
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;

        let axes = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            Some(vec![normalize_axis_best_effort(axis, rank) as u32])
        } else if axis != -1 {
            Some(vec![axis as u32])
        } else {
            None
        };

        let scale = inputs
            .get(1)
            .map(|n| b.resolve_operand(n))
            .transpose()?
            .map(operand_index);
        let bias = inputs
            .get(2)
            .map(|n| b.resolve_operand(n))
            .transpose()?
            .map(operand_index);

        let opts = MLLayerNormalizationOptions {
            label: output_name.clone(),
            scale,
            bias,
            axes,
            epsilon,
        };
        let out = b
            .builder
            .layer_normalization_with_options(input, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_group_normalization(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "GroupNormalization expects 3 inputs (X, scale, bias), got {}",
                inputs.len()
            )));
        }

        let mut epsilon = 1e-5f64;
        let mut num_groups: Option<i64> = None;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "epsilon" if attr.f != 0.0 => epsilon = attr.f as f64,
                "num_groups" => num_groups = Some(attr.i),
                _ => {}
            }
        }
        let num_groups = num_groups.ok_or_else(|| {
            OnnxError::InvalidShape("GroupNormalization requires num_groups attribute".to_string())
        })?;
        if num_groups <= 0 {
            return Err(OnnxError::InvalidShape(format!(
                "GroupNormalization num_groups must be positive, got {num_groups}"
            )));
        }

        let input_shape = context
            .resolve_shape(inputs[0].as_str())
            .ok_or_else(|| {
                OnnxError::InvalidShape(
                    "GroupNormalization requires a known input shape for decomposition".to_string(),
                )
            })?
            .clone();
        if input_shape.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "GroupNormalization expects rank >= 2, got rank {}",
                input_shape.len()
            )));
        }
        let channels = input_shape[1];
        if channels % num_groups != 0 {
            return Err(OnnxError::InvalidShape(format!(
                "GroupNormalization channels {channels} not divisible by num_groups {num_groups}"
            )));
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let scale = b.resolve_operand(&inputs[1])?;
        let bias = b.resolve_operand(&inputs[2])?;

        let tail_product: i64 = input_shape[1..].iter().product();
        let group_inner = tail_product / num_groups;
        let grouped_shape = vec![input_shape[0], num_groups, group_inner];
        let grouped_mldim = i64_slice_to_mldim(&grouped_shape)?;

        let reshape_label = format!("{output_name}__gn_reshape");
        let grouped = reshape_with_shape(b, input, &reshape_label, grouped_mldim)?;

        let reduce_axes = vec![2u32];
        let mean_label = format!("{output_name}__gn_mean");
        let mean = b
            .builder
            .reduce_mean_with_options(
                grouped,
                MLReduceOptions {
                    label: mean_label.clone(),
                    axes: Some(reduce_axes.clone()),
                    keep_dimensions: true,
                },
            )
            .map_err(map_op_error)?;

        let centered_label = format!("{output_name}__gn_centered");
        let centered = b
            .builder
            .sub_with_options(grouped, mean, OnnxBuilder::labeled_options(&centered_label))
            .map_err(map_op_error)?;

        let sq_label = format!("{output_name}__gn_sq");
        let squared = b
            .builder
            .mul_with_options(centered, centered, OnnxBuilder::labeled_options(&sq_label))
            .map_err(map_op_error)?;

        let var_label = format!("{output_name}__gn_var");
        let variance = b
            .builder
            .reduce_mean_with_options(
                squared,
                MLReduceOptions {
                    label: var_label.clone(),
                    axes: Some(reduce_axes),
                    keep_dimensions: true,
                },
            )
            .map_err(map_op_error)?;

        let input_dtype = resolve_value_type(context, &inputs[0]).unwrap_or(DataType::Float32);
        let eps_op = register_scalar_like(
            b,
            &format!("{output_name}__gn_eps"),
            epsilon as f32,
            input_dtype,
        )?;
        let var_eps_label = format!("{output_name}__gn_var_eps");
        let var_eps = b
            .builder
            .add_with_options(
                variance,
                eps_op,
                OnnxBuilder::labeled_options(&var_eps_label),
            )
            .map_err(map_op_error)?;

        let std_label = format!("{output_name}__gn_std");
        let std = b
            .builder
            .sqrt_with_options(var_eps, OnnxBuilder::labeled_options(&std_label))
            .map_err(map_op_error)?;

        let normed_group_label = format!("{output_name}__gn_normed");
        let normed_group = b
            .builder
            .div_with_options(
                centered,
                std,
                OnnxBuilder::labeled_options(&normed_group_label),
            )
            .map_err(map_op_error)?;

        let restore_label = format!("{output_name}__gn_restore");
        let restored = reshape_with_shape(
            b,
            normed_group,
            &restore_label,
            i64_slice_to_mldim(&input_shape)?,
        )?;

        let scaled_label = format!("{output_name}__gn_scaled");
        let scaled = b
            .builder
            .mul_with_options(restored, scale, OnnxBuilder::labeled_options(&scaled_label))
            .map_err(map_op_error)?;

        let out = b
            .builder
            .add_with_options(scaled, bias, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_rms_normalization(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "RMSNormalization expects 2 inputs (X, scale), got {}",
                inputs.len()
            )));
        }

        let mut epsilon = 1e-5f64;
        let mut axis = -1i64;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "epsilon" if attr.f != 0.0 => epsilon = attr.f as f64,
                "axis" if attr.i != 0 => axis = attr.i,
                _ => {}
            }
        }

        let input = b.resolve_operand(&inputs[0])?;
        let scale = b.resolve_operand(&inputs[1])?;
        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank) as u32
        } else {
            axis as u32
        };

        let output_name = output_label(node, node_name);
        let out = rms_normalize_operand(
            b,
            input,
            scale,
            axis,
            epsilon,
            resolve_value_type(context, &inputs[0]).unwrap_or(DataType::Float32),
            &output_name,
        )?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_skip_simplified_layer_normalization(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if !(3..=4).contains(&inputs.len()) {
            return Err(OnnxError::InvalidShape(format!(
                "SkipSimplifiedLayerNormalization expects 3 or 4 inputs \
                 (input, skip, gamma[, bias]), got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let skip = b.resolve_operand(&inputs[1])?;
        let gamma = b.resolve_operand(&inputs[2])?;
        let mut sum = b
            .builder
            .add_with_options(
                input,
                skip,
                OnnxBuilder::labeled_options(&format!("{output_name}__sum")),
            )
            .map_err(map_op_error)?;
        if let Some(bias_name) = inputs.get(3).filter(|name| !name.is_empty()) {
            let bias = b.resolve_operand(bias_name)?;
            sum = b
                .builder
                .add_with_options(
                    sum,
                    bias,
                    OnnxBuilder::labeled_options(&format!("{output_name}__sum_bias")),
                )
                .map_err(map_op_error)?;
        }

        let epsilon = node
            .attribute
            .iter()
            .find(|attr| attr.name == "epsilon")
            .map(|attr| attr.f as f64)
            .unwrap_or(1e-5);
        let rank = context.input_rank(&inputs[0]).ok_or_else(|| {
            OnnxError::InvalidShape(
                "SkipSimplifiedLayerNormalization requires a known input rank".to_string(),
            )
        })?;
        let axis = u32::try_from(rank.saturating_sub(1)).map_err(|_| {
            OnnxError::InvalidShape(
                "SkipSimplifiedLayerNormalization input rank is out of range".to_string(),
            )
        })?;
        let input_dtype = resolve_value_type(context, &inputs[0]).unwrap_or(DataType::Float32);
        let out = rms_normalize_operand(b, sum, gamma, axis, epsilon, input_dtype, &output_name)?;

        let mut result = ConversionResult::default();
        if let Some(name) = node.output.first().filter(|name| !name.is_empty()) {
            record_node_output(b, name, &output_name, out);
            result.output_types.insert(name.clone(), input_dtype);
        }
        if let Some(name) = node.output.get(3).filter(|name| !name.is_empty()) {
            record_node_output(b, name, &format!("{output_name}__residual_sum"), sum);
            result.output_types.insert(name.clone(), input_dtype);
        }
        Ok(result)
    }

    fn convert_softmax(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Softmax expects 1 input, got {}",
                inputs.len()
            )));
        }

        let mut axis = -1i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" && attr.i != 0 {
                axis = attr.i;
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank) as u32
        } else {
            axis as u32
        };
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .softmax_with_options(input, axis, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_log_softmax(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "LogSoftmax expects 1 input, got {}",
                inputs.len()
            )));
        }

        let mut axis = -1i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" && attr.i != 0 {
                axis = attr.i;
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank) as u32
        } else {
            axis as u32
        };

        let softmax_label = format!("{output_name}__softmax");
        let softmax = b
            .builder
            .softmax_with_options(input, axis, OnnxBuilder::labeled_options(&softmax_label))
            .map_err(map_op_error)?;

        let out = b
            .builder
            .log_with_options(softmax, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_hardmax(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Hardmax expects 1 input, got {}",
                inputs.len()
            )));
        }

        let mut axis = -1i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" {
                axis = attr.i;
            }
        }

        let input_shape = context
            .resolve_shape(inputs[0].as_str())
            .ok_or_else(|| {
                OnnxError::InvalidShape(
                    "Hardmax requires a known input shape for decomposition".to_string(),
                )
            })?
            .clone();
        let rank = input_shape.len();
        if rank == 0 {
            return Err(OnnxError::InvalidShape(
                "Hardmax expects non-scalar input".to_string(),
            ));
        }

        let axis = normalize_axis_best_effort(axis, rank) as usize;

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let input_dtype = resolve_value_type(context, &inputs[0]).unwrap_or(DataType::Float32);
        let argmax_input = if input_dtype == DataType::Float16 {
            b.builder
                .cast_with_options(
                    input,
                    map_ast_data_type(DataType::Float32)?,
                    OnnxBuilder::labeled_options(&format!("{output_name}__argmax_input")),
                )
                .map_err(map_op_error)?
        } else {
            input
        };

        let argmax_label = format!("{output_name}__argmax");
        let argmax = b
            .builder
            .arg_max_with_options(
                argmax_input,
                axis as u32,
                MLArgMinMaxOptions {
                    label: argmax_label.clone(),
                    keep_dimensions: true,
                    ..Default::default()
                },
            )
            .map_err(map_op_error)?;

        let argmax_i64_label = format!("{output_name}__argmax_i64");
        let argmax_i64 = b
            .builder
            .cast_with_options(
                argmax,
                map_ast_data_type(DataType::Int64)?,
                OnnxBuilder::labeled_options(&argmax_i64_label),
            )
            .map_err(map_op_error)?;

        let positions =
            axis_position_indices(b, &input_shape, axis, &format!("{output_name}__positions"))?;

        let mask_label = format!("{output_name}__mask");
        let mask = b
            .builder
            .equal_with_options(
                positions,
                argmax_i64,
                OnnxBuilder::labeled_options(&mask_label),
            )
            .map_err(map_op_error)?;

        let zero = register_scalar_like(b, &format!("{output_name}__zero"), 0.0, input_dtype)?;
        let one = register_scalar_like(b, &format!("{output_name}__one"), 1.0, input_dtype)?;
        let out = b
            .builder
            .where_with_options(mask, one, zero, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }
}

fn rms_normalize_operand(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    scale: MLOperand,
    axis: u32,
    epsilon: f64,
    input_dtype: DataType,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    let squared = b
        .builder
        .mul_with_options(
            input,
            input,
            OnnxBuilder::labeled_options(&format!("{label}__rms_sq")),
        )
        .map_err(map_op_error)?;
    let mean_sq = b
        .builder
        .reduce_mean_with_options(
            squared,
            MLReduceOptions {
                label: format!("{label}__rms_mean"),
                axes: Some(vec![axis]),
                keep_dimensions: true,
            },
        )
        .map_err(map_op_error)?;
    let eps = register_scalar_like(b, &format!("{label}__rms_eps"), epsilon as f32, input_dtype)?;
    let mean_eps = b
        .builder
        .add_with_options(
            mean_sq,
            eps,
            OnnxBuilder::labeled_options(&format!("{label}__rms_mean_eps")),
        )
        .map_err(map_op_error)?;
    let rms = b
        .builder
        .sqrt_with_options(
            mean_eps,
            OnnxBuilder::labeled_options(&format!("{label}__rms_denom")),
        )
        .map_err(map_op_error)?;
    let normed = b
        .builder
        .div_with_options(
            input,
            rms,
            OnnxBuilder::labeled_options(&format!("{label}__rms_normed")),
        )
        .map_err(map_op_error)?;
    b.builder
        .mul_with_options(normed, scale, OnnxBuilder::labeled_options(label))
        .map_err(map_op_error)
}

fn register_f32_scalar(
    b: &mut OnnxBuilder<'_, '_, '_>,
    name: &str,
    value: f32,
) -> Result<MLOperand, OnnxError> {
    b.register_constant_from_bytes(name, DataType::Float32, &[1], value.to_le_bytes().to_vec())?;
    b.resolve_operand(name)
}

fn register_scalar_like(
    b: &mut OnnxBuilder<'_, '_, '_>,
    name: &str,
    value: f32,
    data_type: DataType,
) -> Result<MLOperand, OnnxError> {
    match data_type {
        DataType::Float16 => {
            let bits = f16::from_f32(value).to_bits();
            b.register_constant_from_bytes(
                name,
                DataType::Float16,
                &[1],
                bits.to_le_bytes().to_vec(),
            )?;
            b.resolve_operand(name)
        }
        _ => register_f32_scalar(b, name, value),
    }
}

fn resolve_value_type(context: &ConversionContext, name: &str) -> Option<DataType> {
    let sanitized = crate::onnx::convert::sanitize_identifier(name);
    context
        .value_types
        .get(name)
        .or_else(|| context.value_types.get(&sanitized))
        .copied()
}

fn axis_position_indices(
    b: &mut OnnxBuilder<'_, '_, '_>,
    shape: &[i64],
    axis: usize,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    let axis_dim = shape[axis];
    if axis_dim <= 0 {
        return Err(OnnxError::InvalidShape(format!(
            "cannot build position indices for non-positive axis dimension {axis_dim}"
        )));
    }
    let values: Vec<i64> = (0..axis_dim).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    b.register_constant_from_bytes(label, DataType::Int64, &[axis_dim as u32], bytes)?;
    let one_d = b.resolve_operand(label)?;

    let mut broadcast_shape = vec![1i64; shape.len()];
    broadcast_shape[axis] = axis_dim;
    let reshaped = reshape_with_shape(
        b,
        one_d,
        &format!("{label}__bc"),
        i64_slice_to_mldim(&broadcast_shape)?,
    )?;

    if broadcast_shape == shape {
        return Ok(reshaped);
    }
    expand_with_shape(
        b,
        reshaped,
        &format!("{label}__exp"),
        i64_slice_to_mldim(shape)?,
    )
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

    fn add_float_attribute(node: &mut NodeProto, name: &str, value: f32) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            f: value,
            ..Default::default()
        });
    }

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        });
    }

    #[test]
    fn test_normalization_handler_supports() {
        let handler = NormalizationHandler;
        assert!(handler.supports("LayerNormalization"));
        assert!(handler.supports("Softmax"));
        assert!(handler.supports("GroupNormalization"));
        assert!(handler.supports("RMSNormalization"));
        assert!(handler.supports("SimplifiedLayerNormalization"));
        assert!(handler.supports("SkipSimplifiedLayerNormalization"));
        assert!(handler.supports("LogSoftmax"));
        assert!(handler.supports("Hardmax"));
    }

    #[test]
    fn test_convert_layer_norm() {
        let handler = NormalizationHandler;
        let mut node =
            create_test_node("LayerNormalization", vec!["x", "scale", "bias"], vec!["y"]);
        add_float_attribute(&mut node, "epsilon", 1e-5);
        add_int_attribute(&mut node, "axis", -1);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_softmax() {
        let handler = NormalizationHandler;
        let mut node = create_test_node("Softmax", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "axis", 1);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_log_softmax() {
        let handler = NormalizationHandler;
        let mut node = create_test_node("LogSoftmax", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "axis", -1);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_group_normalization() {
        use std::collections::HashMap;

        let handler = NormalizationHandler;
        let mut node =
            create_test_node("GroupNormalization", vec!["x", "scale", "bias"], vec!["y"]);
        add_float_attribute(&mut node, "epsilon", 1e-5);
        add_int_attribute(&mut node, "num_groups", 1);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 2]);
        value_shapes.insert("scale".to_string(), vec![1, 2]);
        value_shapes.insert("bias".to_string(), vec![1, 2]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
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
    fn test_convert_rms_normalization() {
        use std::collections::HashMap;

        let handler = NormalizationHandler;
        let mut node = create_test_node("RMSNormalization", vec!["x", "scale"], vec!["y"]);
        add_float_attribute(&mut node, "epsilon", 1e-5);
        add_int_attribute(&mut node, "axis", -1);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 2]);
        value_shapes.insert("scale".to_string(), vec![1, 2]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
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
    fn test_convert_simplified_and_skip_simplified_layer_normalization() {
        use std::collections::HashMap;

        let handler = NormalizationHandler;
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 2, 4]);
        value_shapes.insert("skip".to_string(), vec![1, 2, 4]);
        value_shapes.insert("gamma".to_string(), vec![4]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let context = crate::onnx::ops::ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let mut simplified = create_test_node(
            "SimplifiedLayerNormalization",
            vec!["x", "gamma"],
            vec!["y"],
        );
        add_float_attribute(&mut simplified, "epsilon", 1e-5);
        add_int_attribute(&mut simplified, "axis", -1);
        crate::onnx::ops::convert_handler_with_context(&handler, &simplified, &context).unwrap();

        let mut skip = create_test_node(
            "SkipSimplifiedLayerNormalization",
            vec!["x", "skip", "gamma"],
            vec!["y", "", "", "sum"],
        );
        add_float_attribute(&mut skip, "epsilon", 1e-5);
        let result =
            crate::onnx::ops::convert_handler_with_context(&handler, &skip, &context).unwrap();
        assert_eq!(result.output_types.get("y"), Some(&DataType::Float32));
        assert_eq!(result.output_types.get("sum"), Some(&DataType::Float32));
    }

    #[test]
    fn test_convert_hardmax() {
        use std::collections::HashMap;

        let handler = NormalizationHandler;
        let mut node = create_test_node("Hardmax", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "axis", -1);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 2]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
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
}
