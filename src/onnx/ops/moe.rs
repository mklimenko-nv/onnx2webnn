/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// com.microsoft.MoE (mixture of experts) decomposition.
//
// WebNN has no routing/gather-based sparse dispatch, so the lowering runs
// every expert densely and blends the results with the routing weights:
//
//   1. routing: softmax over the top-k router logits per row. The top-k mask
//      is built with k iterations of reduceMax + equal (+ suppression), so
//      exact ties may momentarily select more than k experts - negligible in
//      float and identical in expectation to ORT's first-hit tie-break.
//   2. experts: X expanded to [E, R, H], batched matmul against the
//      per-expert weights (stored [E, out, in], hence transposed on the last
//      two axes), bias, activation, second matmul + bias -> [E, R, H].
//   3. blend: multiply by routing weights [E, R, 1] and reduceSum over E.
//
// Activation support: fused interleaved SwiGLU (`swiglu_fusion=1`, the
// gpt-oss export layout: [g0, l0, g1, l1, ...]) with ORT's clamp semantics
//   gate = min(gate, limit); linear = clamp(linear, +/-limit);
//   out  = gate * sigmoid(alpha*gate) * (linear + beta)
// plus plain relu/gelu/sigmoid. Rejected: fc3 (unfused gating), sparse
// mixer, and other fusion layouts.
//
// Dense execution trades FLOPs (num_experts / k times more) for a static
// WebNN graph; weights stay shared constants, so memory matches the model.

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{
    expand_with_shape, i64_slice_to_mldim, output_label, record_node_output, reshape_with_shape,
};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::conv::lookup_shape;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLClampOptions, MLReduceOptions, MLTransposeOptions};
use rustnn::DataType;

pub struct MoeHandler;

impl OpHandler for MoeHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "MoE" | "QMoE")
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let node_name = if !node.name.is_empty() {
            node.name.clone()
        } else {
            "unnamed".to_string()
        };
        convert_moe(node, &node_name, context, b)
    }
}

fn scalar_const(
    b: &mut OnnxBuilder<'_, '_, '_>,
    name: &str,
    dtype: DataType,
    value: f32,
) -> Result<MLOperand, OnnxError> {
    match dtype {
        DataType::Float16 => b.register_constant_from_bytes(
            name,
            DataType::Float16,
            &[1],
            half::f16::from_f32(value).to_le_bytes().to_vec(),
        )?,
        _ => b.register_constant_from_bytes(
            name,
            DataType::Float32,
            &[1],
            value.to_le_bytes().to_vec(),
        )?,
    }
    b.resolve_operand(name)
}

fn ml_float(dtype: DataType) -> MLOperandDataType {
    match dtype {
        DataType::Float16 => MLOperandDataType::Float16,
        _ => MLOperandDataType::Float32,
    }
}

/// Dequantize QMoE expert weights (uint8; 4-bit packs two values per byte,
/// low nibble first; symmetric zero point `1 << (bits-1)`) with blockwise
/// scales `[E, out, in/block_size]` through `dequantizeLinear`; expanded to a
/// float constant only when the block layout cannot be expressed that way.
#[allow(clippy::too_many_arguments)]
fn dequantize_expert_weights(
    b: &mut OnnxBuilder<'_, '_, '_>,
    context: &ConversionContext,
    weight_name: &str,
    scales_name: &str,
    block_size: i64,
    bits: i64,
    dtype: DataType,
    const_name: &str,
) -> Result<(String, Vec<i64>), OnnxError> {
    let per_byte = (8 / bits) as usize;
    let zero_point = f32::from(1u8 << (bits - 1));
    let w_tensor = context
        .initializers
        .get(weight_name)
        .copied()
        .ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "QMoE weight '{weight_name}' must be an initializer"
            ))
        })?;
    let s_tensor = context
        .initializers
        .get(scales_name)
        .copied()
        .ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "QMoE scales '{scales_name}' must be an initializer"
            ))
        })?;
    if w_tensor.dims.len() != 3 || s_tensor.dims.len() != 3 {
        return Err(OnnxError::InvalidShape(format!(
            "QMoE expects 3-D weights/scales, got {:?} and {:?}",
            w_tensor.dims, s_tensor.dims
        )));
    }
    let (experts, out_rows, packed) = (w_tensor.dims[0], w_tensor.dims[1], w_tensor.dims[2]);
    let in_cols = packed * per_byte as i64;
    let blocks = (in_cols + block_size - 1) / block_size;
    if s_tensor.dims != [experts, out_rows, blocks] {
        return Err(OnnxError::InvalidShape(format!(
            "QMoE scales {:?} do not match weights [E={experts}, out={out_rows}] with \
             {blocks} blocks of {block_size}",
            s_tensor.dims
        )));
    }

    let mut bytes = crate::onnx::builder::tensor_proto_to_bytes(w_tensor)?;
    let (rows_total, packed_u, in_u, blocks_u) = (
        (experts * out_rows) as usize,
        packed as usize,
        in_cols as usize,
        blocks as usize,
    );
    if bytes.len() < rows_total * packed_u {
        return Err(OnnxError::InvalidShape(format!(
            "QMoE weight payload too small for [E={experts}, out={out_rows}, in={in_cols}]"
        )));
    }

    // Blockwise dequantizeLinear needs `in` divisible by the block count.
    if in_cols % blocks == 0 {
        let shape_u32 = [experts as u32, out_rows as u32, in_cols as u32];
        let block_shape = [experts as u32, out_rows as u32, blocks as u32];
        let (q_dtype, zp_byte, zp_len) = if bits == 4 {
            (DataType::Uint4, 0x88u8, (rows_total * blocks_u).div_ceil(2))
        } else {
            (DataType::Uint8, 0x80u8, rows_total * blocks_u)
        };
        let q_name = format!("{const_name}__q");
        // Drop any trailing padding and move the payload without copying.
        bytes.truncate(rows_total * packed_u);
        b.register_constant_from_bytes(&q_name, q_dtype, &shape_u32, bytes)?;
        let s_dtype = crate::onnx::convert::map_onnx_data_type(s_tensor.data_type)?;
        let s_bytes = crate::onnx::builder::tensor_proto_to_bytes(s_tensor)?;
        let s_name = format!("{const_name}__scales");
        b.register_constant_from_bytes(&s_name, s_dtype, &block_shape, s_bytes)?;
        let zp_name = format!("{const_name}__zp");
        b.register_constant_from_bytes(&zp_name, q_dtype, &block_shape, vec![zp_byte; zp_len])?;
        let q = b.resolve_operand(&q_name)?;
        let s = b.resolve_operand(&s_name)?;
        let zp = b.resolve_operand(&zp_name)?;
        let mut weights = b
            .builder
            .dequantize_linear_with_zeropoint(q, s, zp)
            .map_err(map_op_error)?;
        if s_dtype != dtype {
            weights = b
                .builder
                .cast_with_options(
                    weights,
                    crate::onnx::builder::map_ast_data_type(dtype)?,
                    OnnxBuilder::labeled_options(&format!("{const_name}__cast")),
                )
                .map_err(map_op_error)?;
        }
        b.record_operand(&[const_name], weights);
        return Ok((const_name.to_string(), vec![experts, out_rows, in_cols]));
    }

    let scales = crate::onnx::ops::matmul::decode_float_tensor_as_f32(s_tensor)?;
    if scales.len() < rows_total * blocks_u {
        return Err(OnnxError::InvalidShape(format!(
            "QMoE scale payload too small for [E={experts}, out={out_rows}, in={in_cols}]"
        )));
    }

    let mut values = vec![0f32; rows_total * in_u];
    for row in 0..rows_total {
        let row_bytes = &bytes[row * packed_u..(row + 1) * packed_u];
        let row_scales = &scales[row * blocks_u..(row + 1) * blocks_u];
        let row_out = &mut values[row * in_u..(row + 1) * in_u];
        for (i, value) in row_out.iter_mut().enumerate() {
            let q = if per_byte == 2 {
                let byte = row_bytes[i / 2];
                if i % 2 == 0 {
                    byte & 0x0F
                } else {
                    byte >> 4
                }
            } else {
                row_bytes[i]
            };
            *value = (f32::from(q) - zero_point) * row_scales[i / block_size as usize];
        }
    }

    let shape_u32 = [experts as u32, out_rows as u32, in_cols as u32];
    match dtype {
        DataType::Float16 => {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                .collect();
            b.register_constant_from_bytes(const_name, DataType::Float16, &shape_u32, bytes)?;
        }
        _ => {
            b.register_constant_from_bytes(
                const_name,
                DataType::Float32,
                &shape_u32,
                bytemuck::cast_slice(&values).to_vec(),
            )?;
        }
    }
    Ok((const_name.to_string(), vec![experts, out_rows, in_cols]))
}

#[allow(clippy::too_many_lines)]
fn convert_moe(
    node: &NodeProto,
    node_name: &str,
    context: &ConversionContext,
    b: &mut OnnxBuilder<'_, '_, '_>,
) -> Result<ConversionResult, OnnxError> {
    let op_type = node.op_type.as_str();
    // QMoE interleaves scale inputs: X, logits, fc1_w, fc1_scales, fc1_b,
    // fc2_w, fc2_scales, fc2_b, fc3... MoE: X, logits, fc1_w, fc1_b, fc2_w,
    // fc2_b, fc3...
    let is_quant = op_type == "QMoE";
    let (w1_idx, b1_idx, w2_idx, b2_idx, fc3_start) = if is_quant {
        (2usize, 4usize, 5usize, 7usize, 8usize)
    } else {
        (2usize, 3usize, 4usize, 5usize, 6usize)
    };

    let inputs = node.input.as_slice();
    if inputs.len() <= w2_idx {
        return Err(OnnxError::InvalidShape(format!(
            "{op_type} expects at least {} inputs, got {}",
            w2_idx + 1,
            inputs.len()
        )));
    }
    // fc3 (separate gating projection) is only used by unfused swiglu exports.
    if inputs.iter().skip(fc3_start).any(|name| !name.is_empty()) {
        return Err(OnnxError::unsupported_op(
            format!("{op_type}(fc3)"),
            node_name.to_string(),
        ));
    }

    let mut k = 1i64;
    let mut normalize = 0i64;
    let mut use_sparse_mixer = 0i64;
    let mut swiglu_fusion = 0i64;
    let mut activation = "relu".to_string();
    let mut alpha = 1.0f32;
    let mut beta = 0.0f32;
    let mut limit = f32::INFINITY;
    let mut weight_bits = 4i64;
    let mut block_size = 0i64;
    for attr in &node.attribute {
        match attr.name.as_str() {
            "expert_weight_bits" => weight_bits = attr.i,
            "block_size" => block_size = attr.i,
            "k" => k = attr.i,
            "normalize_routing_weights" => normalize = attr.i,
            "use_sparse_mixer" => use_sparse_mixer = attr.i,
            "swiglu_fusion" => swiglu_fusion = attr.i,
            "activation_type" => activation = String::from_utf8_lossy(&attr.s).to_string(),
            "activation_alpha" => alpha = attr.f,
            "activation_beta" => beta = attr.f,
            "swiglu_limit" => limit = attr.f,
            _ => {}
        }
    }
    if use_sparse_mixer != 0 {
        return Err(OnnxError::unsupported_op(
            "MoE(use_sparse_mixer)",
            node_name.to_string(),
        ));
    }
    if activation == "swiglu" && swiglu_fusion != 1 {
        return Err(OnnxError::unsupported_op(
            format!("MoE(swiglu_fusion={swiglu_fusion})"),
            node_name.to_string(),
        ));
    }
    if !matches!(activation.as_str(), "swiglu" | "relu" | "gelu" | "sigmoid") {
        return Err(OnnxError::unsupported_op(
            format!("MoE(activation={activation})"),
            node_name.to_string(),
        ));
    }

    let x_shape = lookup_shape(&inputs[0], context).ok_or_else(|| {
        OnnxError::InvalidShape(format!(
            "MoE requires a static input shape for '{}'",
            inputs[0]
        ))
    })?;
    if is_quant && weight_bits != 4 && weight_bits != 8 {
        return Err(OnnxError::unsupported_op(
            format!("QMoE(expert_weight_bits={weight_bits})"),
            node_name.to_string(),
        ));
    }
    if is_quant && block_size <= 0 {
        return Err(OnnxError::InvalidShape(format!(
            "QMoE requires a positive block_size, got {block_size}"
        )));
    }

    let dtype = context
        .value_types
        .get(inputs[0].as_str())
        .copied()
        .unwrap_or(DataType::Float32);

    // For QMoE, dequantize the packed expert weights into float constants and
    // reuse the dense MoE lowering below (same trade-off as MatMulBnb4).
    let (w1_name, w1_shape) = if is_quant {
        dequantize_expert_weights(
            b,
            context,
            &inputs[w1_idx],
            &inputs[w1_idx + 1],
            block_size,
            weight_bits,
            dtype,
            &format!("{}__fc1_deq", output_label(node, node_name)),
        )?
    } else {
        let shape = lookup_shape(&inputs[w1_idx], context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "MoE requires a static fc1 weight shape for '{}'",
                inputs[w1_idx]
            ))
        })?;
        (inputs[w1_idx].clone(), shape)
    };
    let (w2_name, w2_shape) = if is_quant {
        dequantize_expert_weights(
            b,
            context,
            &inputs[w2_idx],
            &inputs[w2_idx + 1],
            block_size,
            weight_bits,
            dtype,
            &format!("{}__fc2_deq", output_label(node, node_name)),
        )?
    } else {
        let shape = lookup_shape(&inputs[w2_idx], context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "MoE requires a static fc2 weight shape for '{}'",
                inputs[w2_idx]
            ))
        })?;
        (inputs[w2_idx].clone(), shape)
    };
    if x_shape.is_empty() || w1_shape.len() != 3 || w2_shape.len() != 3 {
        return Err(OnnxError::InvalidShape(format!(
            "MoE expects input [.., hidden], fc1 [E,fc1_out,hidden], fc2 [E,hidden,inter], \
             got x={x_shape:?} fc1={w1_shape:?} fc2={w2_shape:?}"
        )));
    }
    let hidden = *x_shape.last().unwrap();
    let num_experts = w1_shape[0];
    let fc1_out = w1_shape[1];
    let rows: i64 = x_shape[..x_shape.len() - 1].iter().product();
    if w1_shape[2] != hidden || w2_shape[0] != num_experts || w2_shape[1] != hidden {
        return Err(OnnxError::InvalidShape(format!(
            "MoE weight shapes inconsistent with hidden={hidden}: fc1={w1_shape:?} fc2={w2_shape:?}"
        )));
    }
    let inter = w2_shape[2];
    if activation == "swiglu" && fc1_out != 2 * inter {
        return Err(OnnxError::InvalidShape(format!(
            "MoE fused swiglu expects fc1 rows = 2*inter ({}), got {fc1_out}",
            2 * inter
        )));
    }
    let k = k.clamp(1, num_experts);

    let big = if matches!(dtype, DataType::Float16) {
        60000.0f32
    } else {
        3.0e38f32
    };
    let label = output_label(node, node_name);

    let x_in = b.resolve_operand(&inputs[0])?;
    let logits = b.resolve_operand(&inputs[1])?;

    // --- Routing weights [R, E] ---
    let weights = {
        let logits = if x_shape.len() != 2 {
            reshape_with_shape(
                b,
                logits,
                &format!("{label}_logits2d"),
                i64_slice_to_mldim(&[rows, num_experts])?,
            )?
        } else {
            logits
        };
        if k >= num_experts {
            b.builder
                .softmax_with_options(
                    logits,
                    1,
                    OnnxBuilder::labeled_options(&format!("{label}_probs")),
                )
                .map_err(map_op_error)?
        } else {
            let big_c = scalar_const(b, &format!("{label}__big"), dtype, big)?;
            // Iteratively peel off the k largest logits to form a 0/1 mask.
            let mut work = logits;
            let mut selected: Option<MLOperand> = None;
            for i in 0..k {
                let row_max = b
                    .builder
                    .reduce_max_with_options(
                        work,
                        MLReduceOptions {
                            label: format!("{label}_top{i}_max"),
                            axes: Some(vec![1]),
                            keep_dimensions: true,
                        },
                    )
                    .map_err(map_op_error)?;
                let is_max_bool = b
                    .builder
                    .equal_with_options(
                        work,
                        row_max,
                        OnnxBuilder::labeled_options(&format!("{label}_top{i}_eq")),
                    )
                    .map_err(map_op_error)?;
                let is_max = b
                    .builder
                    .cast_with_options(
                        is_max_bool,
                        ml_float(dtype),
                        OnnxBuilder::labeled_options(&format!("{label}_top{i}_mask")),
                    )
                    .map_err(map_op_error)?;
                selected = Some(match selected {
                    None => is_max,
                    Some(acc) => b
                        .builder
                        .add_with_options(
                            acc,
                            is_max,
                            OnnxBuilder::labeled_options(&format!("{label}_top{i}_acc")),
                        )
                        .map_err(map_op_error)?,
                });
                if i + 1 < k {
                    let penalty = b
                        .builder
                        .mul_with_options(
                            is_max,
                            big_c,
                            OnnxBuilder::labeled_options(&format!("{label}_top{i}_penalty")),
                        )
                        .map_err(map_op_error)?;
                    work = b
                        .builder
                        .sub_with_options(
                            work,
                            penalty,
                            OnnxBuilder::labeled_options(&format!("{label}_top{i}_next")),
                        )
                        .map_err(map_op_error)?;
                }
            }
            let selected = selected.expect("k >= 1");

            if normalize != 0 {
                // softmax restricted to the selected experts == softmax of the
                // top-k logits renormalized to sum 1 (ORT semantics).
                let one = scalar_const(b, &format!("{label}__one"), dtype, 1.0)?;
                let unselected = b
                    .builder
                    .sub_with_options(
                        one,
                        selected,
                        OnnxBuilder::labeled_options(&format!("{label}_unselected")),
                    )
                    .map_err(map_op_error)?;
                let penalty = b
                    .builder
                    .mul_with_options(
                        unselected,
                        big_c,
                        OnnxBuilder::labeled_options(&format!("{label}_mask_penalty")),
                    )
                    .map_err(map_op_error)?;
                let masked = b
                    .builder
                    .sub_with_options(
                        logits,
                        penalty,
                        OnnxBuilder::labeled_options(&format!("{label}_masked_logits")),
                    )
                    .map_err(map_op_error)?;
                b.builder
                    .softmax_with_options(
                        masked,
                        1,
                        OnnxBuilder::labeled_options(&format!("{label}_probs")),
                    )
                    .map_err(map_op_error)?
            } else {
                let probs = b
                    .builder
                    .softmax_with_options(
                        logits,
                        1,
                        OnnxBuilder::labeled_options(&format!("{label}_softmax")),
                    )
                    .map_err(map_op_error)?;
                b.builder
                    .mul_with_options(
                        probs,
                        selected,
                        OnnxBuilder::labeled_options(&format!("{label}_probs")),
                    )
                    .map_err(map_op_error)?
            }
        }
    };

    // --- Dense expert evaluation ---
    let x2d = if x_shape.len() != 2 {
        reshape_with_shape(
            b,
            x_in,
            &format!("{label}_x2d"),
            i64_slice_to_mldim(&[rows, hidden])?,
        )?
    } else {
        x_in
    };
    let x3d = reshape_with_shape(
        b,
        x2d,
        &format!("{label}_x3d"),
        i64_slice_to_mldim(&[1, rows, hidden])?,
    )?;
    let x_e = expand_with_shape(
        b,
        x3d,
        &format!("{label}_x_e"),
        i64_slice_to_mldim(&[num_experts, rows, hidden])?,
    )?;

    let bmm_transposed = |b: &mut OnnxBuilder<'_, '_, '_>,
                          x: MLOperand,
                          w_name: &str,
                          bias_name: Option<&str>,
                          out_cols: i64,
                          tag: &str|
     -> Result<MLOperand, OnnxError> {
        let w = b.resolve_operand(w_name)?;
        // Weights are [E, out, in]; matmul wants [E, in, out].
        let w_t = b
            .builder
            .transpose_with_options(
                w,
                MLTransposeOptions {
                    label: format!("{label}_{tag}_wt"),
                    permutation: vec![0, 2, 1],
                },
            )
            .map_err(map_op_error)?;
        let mut y = b
            .builder
            .matmul_with_options(
                x,
                w_t,
                OnnxBuilder::labeled_options(&format!("{label}_{tag}_matmul")),
            )
            .map_err(map_op_error)?;
        if let Some(bias_name) = bias_name {
            let bias = b.resolve_operand(bias_name)?;
            let bias = reshape_with_shape(
                b,
                bias,
                &format!("{label}_{tag}_bias3d"),
                i64_slice_to_mldim(&[num_experts, 1, out_cols])?,
            )?;
            y = b
                .builder
                .add_with_options(
                    y,
                    bias,
                    OnnxBuilder::labeled_options(&format!("{label}_{tag}_biased")),
                )
                .map_err(map_op_error)?;
        }
        Ok(y)
    };

    let fc1_bias = inputs
        .get(b1_idx)
        .filter(|n| !n.is_empty())
        .map(String::as_str);
    let fc2_bias = inputs
        .get(b2_idx)
        .filter(|n| !n.is_empty())
        .map(String::as_str);
    let fc1 = bmm_transposed(b, x_e, &w1_name, fc1_bias, fc1_out, "fc1")?;

    let activated = if activation == "swiglu" {
        // Interleaved fused layout: [g0, l0, g1, l1, ...] -> [.., inter, 2].
        let pairs = reshape_with_shape(
            b,
            fc1,
            &format!("{label}_pairs"),
            i64_slice_to_mldim(&[num_experts, rows, inter, 2])?,
        )?;
        let take = |b: &mut OnnxBuilder<'_, '_, '_>,
                    idx: u32,
                    tag: &str|
         -> Result<MLOperand, OnnxError> {
            let sliced = crate::onnx::builder_helpers::slice_with_params(
                b,
                pairs,
                &format!("{label}_{tag}_slice"),
                &[0, 0, 0, idx],
                &i64_slice_to_mldim(&[num_experts, rows, inter, 1])?,
            )?;
            reshape_with_shape(
                b,
                sliced,
                &format!("{label}_{tag}"),
                i64_slice_to_mldim(&[num_experts, rows, inter])?,
            )
        };
        let gate = take(b, 0, "gate")?;
        let linear = take(b, 1, "linear")?;

        let gate = b
            .builder
            .clamp_with_options(
                gate,
                MLClampOptions {
                    label: format!("{label}_gate_clamp"),
                    min_value: None,
                    max_value: limit.is_finite().then(|| serde_json::json!(limit)),
                },
            )
            .map_err(map_op_error)?;
        let linear = b
            .builder
            .clamp_with_options(
                linear,
                MLClampOptions {
                    label: format!("{label}_linear_clamp"),
                    min_value: limit.is_finite().then(|| serde_json::json!(-limit)),
                    max_value: limit.is_finite().then(|| serde_json::json!(limit)),
                },
            )
            .map_err(map_op_error)?;

        let alpha_c = scalar_const(b, &format!("{label}__alpha"), dtype, alpha)?;
        let beta_c = scalar_const(b, &format!("{label}__beta"), dtype, beta)?;
        let gated = b
            .builder
            .mul_with_options(
                gate,
                alpha_c,
                OnnxBuilder::labeled_options(&format!("{label}_gate_alpha")),
            )
            .map_err(map_op_error)?;
        let sig = b
            .builder
            .sigmoid_with_options(
                gated,
                OnnxBuilder::labeled_options(&format!("{label}_gate_sig")),
            )
            .map_err(map_op_error)?;
        let silu = b
            .builder
            .mul_with_options(
                gate,
                sig,
                OnnxBuilder::labeled_options(&format!("{label}_gate_silu")),
            )
            .map_err(map_op_error)?;
        let lin_beta = b
            .builder
            .add_with_options(
                linear,
                beta_c,
                OnnxBuilder::labeled_options(&format!("{label}_linear_beta")),
            )
            .map_err(map_op_error)?;
        b.builder
            .mul_with_options(
                silu,
                lin_beta,
                OnnxBuilder::labeled_options(&format!("{label}_swiglu")),
            )
            .map_err(map_op_error)?
    } else {
        let opts = OnnxBuilder::labeled_options(&format!("{label}_act"));
        match activation.as_str() {
            "relu" => b
                .builder
                .relu_with_options(fc1, opts)
                .map_err(map_op_error)?,
            "gelu" => b
                .builder
                .gelu_with_options(fc1, opts)
                .map_err(map_op_error)?,
            "sigmoid" => b
                .builder
                .sigmoid_with_options(fc1, opts)
                .map_err(map_op_error)?,
            other => unreachable!("activation {other} validated above"),
        }
    };

    let expert_out = bmm_transposed(b, activated, &w2_name, fc2_bias, hidden, "fc2")?;

    // --- Blend with routing weights ---
    let w_t = b
        .builder
        .transpose_with_options(
            weights,
            MLTransposeOptions {
                label: format!("{label}_weights_er"),
                permutation: vec![1, 0],
            },
        )
        .map_err(map_op_error)?;
    let w_e = reshape_with_shape(
        b,
        w_t,
        &format!("{label}_weights_er1"),
        i64_slice_to_mldim(&[num_experts, rows, 1])?,
    )?;
    let weighted = b
        .builder
        .mul_with_options(
            expert_out,
            w_e,
            OnnxBuilder::labeled_options(&format!("{label}_weighted")),
        )
        .map_err(map_op_error)?;
    let summed = b
        .builder
        .reduce_sum_with_options(
            weighted,
            MLReduceOptions {
                label: format!("{label}_sum"),
                axes: Some(vec![0]),
                keep_dimensions: false,
            },
        )
        .map_err(map_op_error)?;

    let out = if x_shape.len() != 2 {
        reshape_with_shape(
            b,
            summed,
            &format!("{label}_out"),
            i64_slice_to_mldim(&x_shape)?,
        )?
    } else {
        summed
    };

    if let Some(onnx_out) = node.output.first().filter(|name| !name.is_empty()) {
        record_node_output(b, onnx_out, &label, out);
    } else {
        b.record_operand(&[&label], out);
    }
    Ok(ConversionResult::default())
}
