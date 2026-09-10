/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Recurrent and positional operators: GRU, LSTM, RotaryEmbedding

use crate::onnx::builder::{map_ast_data_type, map_op_error, operand_index, OnnxBuilder};
use crate::onnx::builder_helpers::{
    i64_slice_to_mldim, map_op_result, output_label, record_node_output, reshape_with_shape,
    slice_with_params,
};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::{
    MLDimension, MLGatherOptions, MLGruOptions, MLLstmOptions, MLSliceOptions, MLSqueezeOptions,
    MLTransposeOptions, MLUnsqueezeOptions,
};
use rustnn::DataType;

pub struct RnnHandler;

impl OpHandler for RnnHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "GRU" | "LSTM" | "RotaryEmbedding")
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        // Sanitized: it seeds operand labels, which become MIL value names in the
        // CoreML backend (raw ONNX names like `/lstm/LSTM` contain `/`, which the
        // MIL parser rejects).
        let node_name = if !node.name.is_empty() {
            sanitize_identifier(&node.name)
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "GRU" => self.convert_gru(node, &node_name, context, b),
            "LSTM" => self.convert_lstm(node, &node_name, context, b),
            "RotaryEmbedding" => self.convert_rotary_embedding(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

impl RnnHandler {
    fn convert_rotary_embedding(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if !(3..=4).contains(&inputs.len()) {
            return Err(OnnxError::InvalidShape(format!(
                "RotaryEmbedding expects 3 or 4 inputs, got {}",
                inputs.len()
            )));
        }

        let is_ms_domain = node.domain == "com.microsoft";
        let (x_name, position_ids_name, cos_name, sin_name) = if is_ms_domain {
            if inputs.len() != 4 {
                return Err(OnnxError::InvalidShape(
                    "com.microsoft.RotaryEmbedding expects \
                     (input, position_ids, cos_cache, sin_cache)"
                        .to_string(),
                ));
            }
            (
                inputs[0].as_str(),
                Some(inputs[1].as_str()),
                inputs[2].as_str(),
                inputs[3].as_str(),
            )
        } else {
            (
                inputs[0].as_str(),
                inputs
                    .get(3)
                    .filter(|name| !name.is_empty())
                    .map(String::as_str),
                inputs[1].as_str(),
                inputs[2].as_str(),
            )
        };

        let input_shape = context.resolve_shape(x_name).ok_or_else(|| {
            OnnxError::InvalidShape("RotaryEmbedding requires a known input shape".to_string())
        })?;
        if input_shape.len() != 3 && input_shape.len() != 4 {
            return Err(OnnxError::InvalidShape(format!(
                "RotaryEmbedding input rank must be 3 or 4, got {}",
                input_shape.len()
            )));
        }
        if input_shape.iter().any(|&dim| dim <= 0) {
            return Err(OnnxError::InvalidShape(format!(
                "RotaryEmbedding requires concrete positive dimensions, got {input_shape:?}"
            )));
        }

        let mut interleaved = false;
        let mut num_heads_attr = 0i64;
        let mut rotary_dim_attr = 0i64;
        let mut scale = 1.0f32;
        for attr in &node.attribute {
            match attr.name.as_str() {
                "interleaved" => interleaved = attr.i != 0,
                "num_heads" => num_heads_attr = attr.i,
                "rotary_embedding_dim" => rotary_dim_attr = attr.i,
                "scale" => scale = attr.f,
                _ => {}
            }
        }
        if (scale - 1.0).abs() > f32::EPSILON {
            return Err(OnnxError::unsupported_op(
                format!("RotaryEmbedding(scale={scale})"),
                node_name.to_string(),
            ));
        }

        let cos_shape = context.resolve_shape(cos_name).ok_or_else(|| {
            OnnxError::InvalidShape("RotaryEmbedding requires a known cos_cache shape".to_string())
        })?;
        let cache_half = *cos_shape.last().filter(|&&dim| dim > 0).ok_or_else(|| {
            OnnxError::InvalidShape("RotaryEmbedding cos_cache has invalid shape".to_string())
        })?;
        let rotary_dim = if rotary_dim_attr > 0 {
            rotary_dim_attr
        } else {
            cache_half.checked_mul(2).ok_or_else(|| {
                OnnxError::InvalidShape("RotaryEmbedding dimension overflow".to_string())
            })?
        };
        if rotary_dim % 2 != 0 || rotary_dim / 2 != cache_half {
            return Err(OnnxError::InvalidShape(format!(
                "RotaryEmbedding dimension {rotary_dim} does not match cache width {cache_half}"
            )));
        }

        let (batch, sequence, num_heads, head_size, restore_4d) = if input_shape.len() == 4 {
            (
                input_shape[0],
                input_shape[2],
                input_shape[1],
                input_shape[3],
                true,
            )
        } else {
            let hidden = input_shape[2];
            let head_size = if num_heads_attr > 0 {
                if hidden % num_heads_attr != 0 {
                    return Err(OnnxError::InvalidShape(format!(
                        "RotaryEmbedding hidden size {hidden} is not divisible by num_heads \
                         {num_heads_attr}"
                    )));
                }
                hidden / num_heads_attr
            } else {
                rotary_dim
            };
            if head_size <= 0 || hidden % head_size != 0 {
                return Err(OnnxError::InvalidShape(format!(
                    "RotaryEmbedding cannot derive heads from hidden size {hidden} and \
                     head size {head_size}"
                )));
            }
            (
                input_shape[0],
                input_shape[1],
                hidden / head_size,
                head_size,
                false,
            )
        };
        if rotary_dim > head_size {
            return Err(OnnxError::InvalidShape(format!(
                "RotaryEmbedding dimension {rotary_dim} exceeds head size {head_size}"
            )));
        }

        let label = output_label(node, node_name);
        let mut x = b.resolve_operand(x_name)?;
        if restore_4d {
            x = b
                .builder
                .transpose_with_options(
                    x,
                    MLTransposeOptions {
                        label: format!("{label}__to_bsnh"),
                        permutation: vec![0, 2, 1, 3],
                    },
                )
                .map_err(map_op_error)?;
        }
        x = reshape_with_shape(
            b,
            x,
            &format!("{label}__heads"),
            i64_slice_to_mldim(&[batch, sequence, num_heads, head_size])?,
        )?;

        let mut cos = b.resolve_operand(cos_name)?;
        let mut sin = b.resolve_operand(sin_name)?;
        if let Some(position_ids_name) = position_ids_name {
            let position_ids = b.resolve_operand(position_ids_name)?;
            cos = b
                .builder
                .gather_with_options(
                    cos,
                    position_ids,
                    MLGatherOptions {
                        label: format!("{label}__cos_gather"),
                        axis: 0,
                    },
                )
                .map_err(map_op_error)?;
            sin = b
                .builder
                .gather_with_options(
                    sin,
                    position_ids,
                    MLGatherOptions {
                        label: format!("{label}__sin_gather"),
                        axis: 0,
                    },
                )
                .map_err(map_op_error)?;
        }
        let cache_shape = i64_slice_to_mldim(&[batch, sequence, 1, cache_half])?;
        cos = reshape_with_shape(b, cos, &format!("{label}__cos"), cache_shape.clone())?;
        sin = reshape_with_shape(b, sin, &format!("{label}__sin"), cache_shape)?;

        let full_shape = [batch, sequence, num_heads, head_size];
        let rotate_shape = [batch, sequence, num_heads, rotary_dim];
        let x_rotate = slice_operand(
            b,
            x,
            &[0, 0, 0, 0],
            &rotate_shape,
            &[1, 1, 1, 1],
            &format!("{label}__rotate"),
        )?;
        let half = rotary_dim / 2;
        let half_shape = [batch, sequence, num_heads, half];
        let (x1, x2) = if interleaved {
            // WebNN slice `sizes` describe the input extent before applying strides.
            let even_extent = [batch, sequence, num_heads, rotary_dim];
            let odd_extent = [batch, sequence, num_heads, rotary_dim - 1];
            (
                slice_operand(
                    b,
                    x_rotate,
                    &[0, 0, 0, 0],
                    &even_extent,
                    &[1, 1, 1, 2],
                    &format!("{label}__even"),
                )?,
                slice_operand(
                    b,
                    x_rotate,
                    &[0, 0, 0, 1],
                    &odd_extent,
                    &[1, 1, 1, 2],
                    &format!("{label}__odd"),
                )?,
            )
        } else {
            (
                slice_operand(
                    b,
                    x_rotate,
                    &[0, 0, 0, 0],
                    &half_shape,
                    &[1, 1, 1, 1],
                    &format!("{label}__first_half"),
                )?,
                slice_operand(
                    b,
                    x_rotate,
                    &[0, 0, 0, half],
                    &half_shape,
                    &[1, 1, 1, 1],
                    &format!("{label}__second_half"),
                )?,
            )
        };

        let cos_x1 = b.builder.mul(x1, cos).map_err(map_op_error)?;
        let sin_x2 = b.builder.mul(x2, sin).map_err(map_op_error)?;
        let real = b.builder.sub(cos_x1, sin_x2).map_err(map_op_error)?;
        let sin_x1 = b.builder.mul(x1, sin).map_err(map_op_error)?;
        let cos_x2 = b.builder.mul(x2, cos).map_err(map_op_error)?;
        let imag = b.builder.add(sin_x1, cos_x2).map_err(map_op_error)?;

        let rotated = if interleaved {
            let component_shape = i64_slice_to_mldim(&[batch, sequence, num_heads, half, 1])?;
            let real = reshape_with_shape(
                b,
                real,
                &format!("{label}__real_component"),
                component_shape.clone(),
            )?;
            let imag = reshape_with_shape(
                b,
                imag,
                &format!("{label}__imag_component"),
                component_shape,
            )?;
            let joined = b.builder.concat(&[real, imag], 4).map_err(map_op_error)?;
            reshape_with_shape(
                b,
                joined,
                &format!("{label}__interleaved"),
                i64_slice_to_mldim(&rotate_shape)?,
            )?
        } else {
            b.builder.concat(&[real, imag], 3).map_err(map_op_error)?
        };

        let mut out = if rotary_dim < head_size {
            let tail_shape = [batch, sequence, num_heads, head_size - rotary_dim];
            let tail = slice_operand(
                b,
                x,
                &[0, 0, 0, rotary_dim],
                &tail_shape,
                &[1, 1, 1, 1],
                &format!("{label}__tail"),
            )?;
            b.builder
                .concat(&[rotated, tail], 3)
                .map_err(map_op_error)?
        } else {
            rotated
        };
        debug_assert_eq!(full_shape[3], head_size);
        if restore_4d {
            out = b
                .builder
                .transpose_with_options(
                    out,
                    MLTransposeOptions {
                        label: format!("{label}__restore_bnsd"),
                        permutation: vec![0, 2, 1, 3],
                    },
                )
                .map_err(map_op_error)?;
        } else {
            out = reshape_with_shape(b, out, &label, i64_slice_to_mldim(input_shape)?)?;
        }

        if let Some(name) = node.output.first().filter(|name| !name.is_empty()) {
            record_node_output(b, name, &label, out);
        }
        let mut result = ConversionResult::default();
        if let Some(name) = node.output.first().filter(|name| !name.is_empty()) {
            result
                .output_types
                .insert(name.clone(), rnn_input_dtype(context, x_name));
        }
        Ok(result)
    }

    fn convert_gru(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "GRU expects at least 3 inputs (X, W, R), got {}",
                inputs.len()
            )));
        }

        let direction = validate_rnn_attrs(node, node_name, "GRU")?;
        let bidirectional = direction == "both";
        reject_optional_rnn_inputs(inputs, 4, node_name, "GRU")?;

        let hidden_size = require_hidden_size(node, "GRU")?;
        let gate_bias_len = 3u32 * hidden_size;
        let input_dtype = rnn_input_dtype(context, &inputs[0]);
        let compute_f32 = input_dtype == DataType::Float16;

        let x = maybe_cast_for_rnn(
            b,
            b.resolve_operand(&inputs[0])?,
            compute_f32,
            DataType::Float32,
            &format!("{node_name}_x_f32"),
        )?;
        let w = maybe_cast_for_rnn(
            b,
            b.resolve_operand(&inputs[1])?,
            compute_f32,
            DataType::Float32,
            &format!("{node_name}_w_f32"),
        )?;
        let r = maybe_cast_for_rnn(
            b,
            b.resolve_operand(&inputs[2])?,
            compute_f32,
            DataType::Float32,
            &format!("{node_name}_r_f32"),
        )?;
        let steps = resolve_steps(context, &inputs[0]);

        let num_directions = if bidirectional { 2 } else { 1 };
        let (bias, recurrent_bias) = split_combined_bias(
            b,
            node_name,
            inputs.get(3).map(String::as_str),
            gate_bias_len,
            compute_f32,
            num_directions,
        )?;

        let outputs = node.output.as_slice();
        let wants_sequence = outputs.first().is_some_and(|name| !name.is_empty());
        let wants_hidden = outputs.get(1).is_some_and(|name| !name.is_empty());

        let mut linear_before_reset = 0i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "linear_before_reset" {
                linear_before_reset = attr.i;
            }
        }

        let label = output_label(node, node_name);
        let options = MLGruOptions {
            label: label.clone(),
            bias,
            recurrent_bias,
            return_sequence: wants_sequence,
            direction,
            reset_after: linear_before_reset != 0,
            ..Default::default()
        };

        let gru_outputs = b
            .builder
            .gru_with_options(x, w, r, steps, hidden_size, options)
            .map_err(map_op_error)?;

        let mut result = ConversionResult::default();

        if wants_sequence {
            let seq = gru_outputs.get(1).copied().ok_or_else(|| {
                OnnxError::InvalidShape("GRU missing sequence output".to_string())
            })?;
            let mapped =
                map_onnx_sequence_output(b, node_name, seq, context, &outputs[0], bidirectional)?;
            let mapped = maybe_cast_for_rnn(
                b,
                mapped,
                compute_f32,
                input_dtype,
                &format!("{label}_y_cast"),
            )?;
            record_node_output(b, &outputs[0], &format!("{label}_y"), mapped);
            result.output_types.insert(outputs[0].clone(), input_dtype);
        }

        if wants_hidden {
            let hidden = maybe_cast_for_rnn(
                b,
                gru_outputs[0],
                compute_f32,
                input_dtype,
                &format!("{label}_y_h_cast"),
            )?;
            let out_name = outputs.get(1).expect("checked above");
            record_node_output(b, out_name, &format!("{label}_y_h"), hidden);
            result.output_types.insert(out_name.clone(), input_dtype);
        }

        Ok(result)
    }

    fn convert_lstm(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "LSTM expects at least 3 inputs (X, W, R), got {}",
                inputs.len()
            )));
        }

        let direction = validate_rnn_attrs(node, node_name, "LSTM")?;
        let bidirectional = direction == "both";
        // sequence_lens (input 4) is unsupported; initial_h/initial_c (5/6)
        // map to WebNN initialHiddenState/initialCellState. Peephole (7) is
        // unsupported.
        if inputs.get(4).is_some_and(|n| !n.is_empty()) {
            return Err(OnnxError::unsupported_op(
                "LSTM(sequence_lens)",
                node_name.to_string(),
            ));
        }
        if inputs.get(7).is_some_and(|n| !n.is_empty()) {
            return Err(OnnxError::unsupported_op(
                "LSTM(peephole)",
                node_name.to_string(),
            ));
        }

        let hidden_size = require_hidden_size(node, "LSTM")?;
        let gate_bias_len = 4u32 * hidden_size;
        let input_dtype = rnn_input_dtype(context, &inputs[0]);
        let compute_f32 = input_dtype == DataType::Float16;

        let x = maybe_cast_for_rnn(
            b,
            b.resolve_operand(&inputs[0])?,
            compute_f32,
            DataType::Float32,
            &format!("{node_name}_x_f32"),
        )?;
        let w = maybe_cast_for_rnn(
            b,
            b.resolve_operand(&inputs[1])?,
            compute_f32,
            DataType::Float32,
            &format!("{node_name}_w_f32"),
        )?;
        let r = maybe_cast_for_rnn(
            b,
            b.resolve_operand(&inputs[2])?,
            compute_f32,
            DataType::Float32,
            &format!("{node_name}_r_f32"),
        )?;
        let steps = resolve_steps(context, &inputs[0]);

        let num_directions = if bidirectional { 2 } else { 1 };
        let (bias, recurrent_bias) = split_combined_bias(
            b,
            node_name,
            inputs.get(3).map(String::as_str),
            gate_bias_len,
            compute_f32,
            num_directions,
        )?;

        let outputs = node.output.as_slice();
        let wants_sequence = outputs.first().is_some_and(|name| !name.is_empty());
        let wants_hidden = outputs.get(1).is_some_and(|name| !name.is_empty());
        let wants_cell = outputs.get(2).is_some_and(|name| !name.is_empty());

        let mut initial_state = |idx: usize, tag: &str| -> Result<Option<u32>, OnnxError> {
            match inputs.get(idx).filter(|n| !n.is_empty()) {
                Some(name) => {
                    let op = maybe_cast_for_rnn(
                        b,
                        b.resolve_operand(name)?,
                        compute_f32,
                        DataType::Float32,
                        &format!("{node_name}_{tag}_f32"),
                    )?;
                    Ok(Some(operand_index(op)))
                }
                None => Ok(None),
            }
        };
        let initial_hidden_state = initial_state(5, "h0")?;
        let initial_cell_state = initial_state(6, "c0")?;

        let label = output_label(node, node_name);
        let options = MLLstmOptions {
            label: label.clone(),
            bias,
            recurrent_bias,
            initial_hidden_state,
            initial_cell_state,
            return_sequence: wants_sequence,
            direction,
            ..Default::default()
        };

        let lstm_outputs = b
            .builder
            .lstm_with_options(x, w, r, steps, hidden_size, options)
            .map_err(map_op_error)?;

        let mut result = ConversionResult::default();

        if wants_sequence {
            let seq = lstm_outputs.get(2).copied().ok_or_else(|| {
                OnnxError::InvalidShape("LSTM missing sequence output".to_string())
            })?;
            let mapped =
                map_onnx_sequence_output(b, node_name, seq, context, &outputs[0], bidirectional)?;
            let mapped = maybe_cast_for_rnn(
                b,
                mapped,
                compute_f32,
                input_dtype,
                &format!("{label}_y_cast"),
            )?;
            record_node_output(b, &outputs[0], &format!("{label}_y"), mapped);
            result.output_types.insert(outputs[0].clone(), input_dtype);
        }

        if wants_hidden {
            let hidden = maybe_cast_for_rnn(
                b,
                lstm_outputs[0],
                compute_f32,
                input_dtype,
                &format!("{label}_y_h_cast"),
            )?;
            let out_name = outputs.get(1).expect("checked above");
            record_node_output(b, out_name, &format!("{label}_y_h"), hidden);
            result.output_types.insert(out_name.clone(), input_dtype);
        }

        if wants_cell {
            let cell = maybe_cast_for_rnn(
                b,
                lstm_outputs[1],
                compute_f32,
                input_dtype,
                &format!("{label}_y_c_cast"),
            )?;
            let out_name = outputs.get(2).expect("checked above");
            record_node_output(b, out_name, &format!("{label}_y_c"), cell);
            result.output_types.insert(out_name.clone(), input_dtype);
        }

        Ok(result)
    }
}

fn slice_operand(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    starts: &[i64],
    sizes: &[i64],
    strides: &[u32],
    label: &str,
) -> Result<MLOperand, OnnxError> {
    let starts: Vec<u32> = starts
        .iter()
        .map(|&value| {
            u32::try_from(value).map_err(|_| {
                OnnxError::InvalidShape(format!(
                    "RotaryEmbedding slice start {value} is out of range"
                ))
            })
        })
        .collect::<Result<_, _>>()?;
    b.builder
        .slice_with_options(
            input,
            &starts,
            &i64_slice_to_mldim(sizes)?,
            MLSliceOptions {
                label: label.to_string(),
                strides: strides.to_vec(),
            },
        )
        .map_err(map_op_error)
}

fn require_hidden_size(node: &NodeProto, op: &str) -> Result<u32, OnnxError> {
    for attr in node.attribute.as_slice() {
        if attr.name.as_str() == "hidden_size" && attr.i > 0 {
            return u32::try_from(attr.i).map_err(|_| {
                OnnxError::InvalidShape(format!("{op} hidden_size {} is out of range", attr.i))
            });
        }
    }
    Err(OnnxError::MissingAttribute {
        attr: "hidden_size".to_string(),
        op: op.to_string(),
    })
}

/// Validate shared RNN attributes and map the ONNX direction to WebNN's
/// ("forward" | "backward" | "both").
fn validate_rnn_attrs(node: &NodeProto, node_name: &str, op: &str) -> Result<String, OnnxError> {
    let mut webnn_direction = "forward".to_string();
    for attr in node.attribute.as_slice() {
        match attr.name.as_str() {
            "direction" => {
                let direction = String::from_utf8_lossy(&attr.s).to_lowercase();
                webnn_direction = match direction.as_str() {
                    "" | "forward" => "forward".to_string(),
                    "reverse" => "backward".to_string(),
                    "bidirectional" => "both".to_string(),
                    other => {
                        return Err(OnnxError::unsupported_op(
                            format!("{op}(direction={other})"),
                            node_name.to_string(),
                        ));
                    }
                };
            }
            "layout" => {
                let layout = String::from_utf8_lossy(&attr.s);
                if !layout.is_empty() && layout != "zrh" && layout != "iofg" {
                    return Err(OnnxError::unsupported_op(
                        format!("{op}(layout={layout})"),
                        node_name.to_string(),
                    ));
                }
            }
            "activations" if !attr.strings.is_empty() => {
                return Err(OnnxError::unsupported_op(
                    format!("{op}(custom activations)"),
                    node_name.to_string(),
                ));
            }
            _ => {}
        }
    }
    Ok(webnn_direction)
}

fn reject_optional_rnn_inputs(
    inputs: &[String],
    first_optional_index: usize,
    node_name: &str,
    op: &str,
) -> Result<(), OnnxError> {
    for (idx, name) in inputs.iter().enumerate().skip(first_optional_index) {
        if !name.is_empty() {
            return Err(OnnxError::unsupported_op(
                format!("{op} optional input {idx}"),
                node_name.to_string(),
            ));
        }
    }
    Ok(())
}

fn resolve_steps(context: &ConversionContext, x_name: &str) -> u32 {
    match context.input_rank(x_name) {
        Some(3) => context
            .resolve_shape(x_name)
            .and_then(|shape| shape.first().copied())
            .filter(|&dim| dim > 0)
            .and_then(|dim| u32::try_from(dim).ok())
            .unwrap_or(1),
        _ => 1,
    }
}

/// Split ONNX combined bias `[1, 2*gate_bias_len]` into WebNN `bias` and `recurrent_bias` `[1, gate_bias_len]`.
fn split_combined_bias(
    b: &mut OnnxBuilder<'_, '_, '_>,
    node_name: &str,
    bias_name: Option<&str>,
    gate_bias_len: u32,
    compute_f32: bool,
    num_directions: u32,
) -> Result<(Option<u32>, Option<u32>), OnnxError> {
    let Some(name) = bias_name.filter(|n| !n.is_empty()) else {
        return Ok((None, None));
    };

    let combined = b.resolve_operand(name)?;
    let combined = maybe_cast_for_rnn(
        b,
        combined,
        compute_f32,
        DataType::Float32,
        &format!("{node_name}_bias_f32"),
    )?;
    let half = gate_bias_len;
    let bias = slice_with_params(
        b,
        combined,
        &format!("{node_name}_bias"),
        &[0, 0],
        &[
            MLDimension::Static(num_directions),
            MLDimension::Static(half),
        ],
    )?;
    let recurrent_bias = slice_with_params(
        b,
        combined,
        &format!("{node_name}_recurrent_bias"),
        &[0, half],
        &[
            MLDimension::Static(num_directions),
            MLDimension::Static(half),
        ],
    )?;
    Ok((
        Some(operand_index(bias)),
        Some(operand_index(recurrent_bias)),
    ))
}

/// Map WebNN sequence `[steps, num_directions, batch, hidden]` to ONNX `Y` layout.
fn map_onnx_sequence_output(
    b: &mut OnnxBuilder<'_, '_, '_>,
    node_name: &str,
    seq: MLOperand,
    context: &ConversionContext,
    onnx_output: &str,
    bidirectional: bool,
) -> Result<MLOperand, OnnxError> {
    let expected_rank = context.resolve_shape(onnx_output).map(|shape| shape.len());

    // Bidirectional: WebNN [steps, 2, batch, hidden] already matches ONNX Y.
    if bidirectional {
        return Ok(seq);
    }

    match expected_rank {
        Some(3) => {
            let opts = MLSqueezeOptions {
                label: format!("{node_name}_squeeze_dir"),
                axes: vec![1],
            };
            map_op_result(b.builder.squeeze_with_options(seq, opts))
        }
        Some(4) => Ok(seq),
        _ => {
            // Unidirectional forward: squeeze `num_directions` when rank is unspecified.
            let opts = MLSqueezeOptions {
                label: format!("{node_name}_squeeze_dir"),
                axes: vec![1],
            };
            let squeezed = map_op_result(b.builder.squeeze_with_options(seq, opts))?;
            if let Some(shape) = context.resolve_shape(onnx_output) {
                if shape.len() == 4 {
                    let unsqueeze_opts = MLUnsqueezeOptions {
                        label: format!("{node_name}_unsqueeze_dir"),
                        axes: vec![1],
                    };
                    return map_op_result(
                        b.builder.unsqueeze_with_options(squeezed, unsqueeze_opts),
                    );
                }
            }
            Ok(squeezed)
        }
    }
}

fn rnn_input_dtype(context: &ConversionContext, input: &str) -> DataType {
    context
        .value_types
        .get(input)
        .copied()
        .unwrap_or(DataType::Float32)
}

fn maybe_cast_for_rnn(
    b: &mut OnnxBuilder<'_, '_, '_>,
    operand: MLOperand,
    should_cast: bool,
    target_type: DataType,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    if !should_cast {
        return Ok(operand);
    }
    b.builder
        .cast_with_options(
            operand,
            map_ast_data_type(target_type)?,
            OnnxBuilder::labeled_options(label),
        )
        .map_err(map_op_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn converts_microsoft_interleaved_rotary_embedding() {
        let handler = RnnHandler;
        let node = NodeProto {
            op_type: "RotaryEmbedding".to_string(),
            domain: "com.microsoft".to_string(),
            name: "rope".to_string(),
            input: vec![
                "x".to_string(),
                "position_ids".to_string(),
                "cos_cache".to_string(),
                "sin_cache".to_string(),
            ],
            output: vec!["y".to_string()],
            attribute: vec![crate::protos::onnx::AttributeProto {
                name: "interleaved".to_string(),
                i: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let initializers = HashMap::new();
        let value_shapes = HashMap::from([
            ("x".to_string(), vec![1, 2, 128]),
            ("position_ids".to_string(), vec![1, 2]),
            ("cos_cache".to_string(), vec![16, 32]),
            ("sin_cache".to_string(), vec![16, 32]),
        ]);
        let value_types = HashMap::from([
            ("x".to_string(), DataType::Float32),
            ("position_ids".to_string(), DataType::Int64),
            ("cos_cache".to_string(), DataType::Float32),
            ("sin_cache".to_string(), DataType::Float32),
        ]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let context = ConversionContext {
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
