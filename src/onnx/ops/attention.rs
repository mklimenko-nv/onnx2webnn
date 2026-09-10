/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// com.microsoft.GroupQueryAttention decomposition.
//
// Lowered as reshape/transpose -> concat with the KV cache -> scaled q*k^T ->
// static causal mask -> softmax -> *v, following the layout ORT uses:
//   query [B, S, H*Dh], key/value [B, S, kvH*Dh], past_{key,value} [B, kvH, P, Dh].
//
// The mask is a compile-time constant that allows key position j for query
// row i iff j <= P + i. This assumes the KV cache is fully populated and the
// batch is unpadded (`seqlens_k == P + S - 1`), which is how transformers.js
// drives GQA for batch-1 web inference; the runtime `seqlens_k` /
// `total_sequence_length` inputs are therefore ignored.
//
// Packed QKV (empty key/value inputs, query = [B, S, (H + 2*kvH)*Dh]) is
// split along the hidden axis first. With do_rotary=1 the cos/sin caches
// (inputs 7/8, [max_seq, rotary_dim/2]) are applied to q and k at positions
// P..P+S before attention. Rejected: softcap and local (sliding-window)
// attention.

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{
    expand_with_shape, i64_slice_to_mldim, output_label, record_node_output, reshape_with_shape,
};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::conv::lookup_shape;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::{MLSliceOptions, MLSplitOptions, MLTransposeOptions};
use rustnn::DataType;

pub struct AttentionHandler;

impl OpHandler for AttentionHandler {
    fn supports(&self, op_type: &str) -> bool {
        op_type == "GroupQueryAttention"
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
        convert_group_query_attention(node, &node_name, context, b)
    }
}

/// Output shapes for shape propagation: (attention output, present key, present value).
pub(crate) fn gqa_output_shapes(
    node: &NodeProto,
    value_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Option<[Vec<i64>; 3]> {
    let ins = node.input.as_slice();
    let q_shape = value_shapes.get(ins.first()?.as_str())?;
    let past_shape = value_shapes.get(ins.get(3)?.as_str())?;
    if q_shape.len() != 3 || past_shape.len() != 4 {
        return None;
    }
    let mut present = past_shape.clone();
    present[2] = past_shape[2].checked_add(q_shape[1])?;
    Some([q_shape.clone(), present.clone(), present])
}

fn transpose(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    permutation: Vec<u32>,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    b.builder
        .transpose_with_options(
            input,
            MLTransposeOptions {
                label: label.to_string(),
                permutation,
            },
        )
        .map_err(map_op_error)
}

#[allow(clippy::too_many_arguments)]
fn heads_layout(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    batch: i64,
    seq: i64,
    heads: i64,
    head_dim: i64,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    // [B, S, heads*Dh] -> [B, S, heads, Dh] -> [B, heads, S, Dh]
    let split = reshape_with_shape(
        b,
        input,
        &format!("{label}_split"),
        i64_slice_to_mldim(&[batch, seq, heads, head_dim])?,
    )?;
    transpose(b, split, vec![0, 2, 1, 3], &format!("{label}_bhsd"))
}

/// Rotate `x` ([B, S, heads, Dh]) with the GQA cos/sin caches
/// ([max_seq, rotary_dim/2]) at positions `past_seq..past_seq+seq`.
#[allow(clippy::too_many_arguments)]
fn apply_rotary(
    b: &mut OnnxBuilder<'_, '_, '_>,
    x: MLOperand,
    cos_cache: MLOperand,
    sin_cache: MLOperand,
    cache_shape: &[i64],
    batch: i64,
    seq: i64,
    past_seq: i64,
    heads: i64,
    head_dim: i64,
    interleaved: bool,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    if cache_shape.len() != 2 || cache_shape[0] < past_seq + seq {
        return Err(OnnxError::InvalidShape(format!(
            "GroupQueryAttention rotary cache {cache_shape:?} too small for positions {past_seq}..{}",
            past_seq + seq
        )));
    }
    let half = cache_shape[1];
    let rotary_dim = half * 2;
    if rotary_dim > head_dim {
        return Err(OnnxError::InvalidShape(format!(
            "GroupQueryAttention rotary_dim {rotary_dim} exceeds head_dim {head_dim}"
        )));
    }
    let strided_slice = |b: &mut OnnxBuilder<'_, '_, '_>,
                         input: MLOperand,
                         starts: &[i64],
                         sizes: &[i64],
                         strides: Vec<u32>,
                         tag: &str|
     -> Result<MLOperand, OnnxError> {
        let starts: Vec<u32> = starts.iter().map(|&v| v as u32).collect();
        b.builder
            .slice_with_options(
                input,
                &starts,
                &i64_slice_to_mldim(sizes)?,
                MLSliceOptions {
                    label: format!("{label}_{tag}"),
                    strides,
                },
            )
            .map_err(map_op_error)
    };
    // cos/sin rows for the current positions, broadcastable to [B, S, heads, half].
    let mut cos = strided_slice(
        b,
        cos_cache,
        &[past_seq, 0],
        &[seq, half],
        Vec::new(),
        "cos_rows",
    )?;
    let mut sin = strided_slice(
        b,
        sin_cache,
        &[past_seq, 0],
        &[seq, half],
        Vec::new(),
        "sin_rows",
    )?;
    let cache_view = i64_slice_to_mldim(&[1, seq, 1, half])?;
    cos = reshape_with_shape(b, cos, &format!("{label}_cos"), cache_view.clone())?;
    sin = reshape_with_shape(b, sin, &format!("{label}_sin"), cache_view)?;

    let rotate_shape = [batch, seq, heads, rotary_dim];
    let x_rotate = if rotary_dim < head_dim {
        strided_slice(b, x, &[0, 0, 0, 0], &rotate_shape, Vec::new(), "rotate")?
    } else {
        x
    };
    let half_shape = [batch, seq, heads, half];
    let (x1, x2) = if interleaved {
        (
            strided_slice(
                b,
                x_rotate,
                &[0, 0, 0, 0],
                &rotate_shape,
                vec![1, 1, 1, 2],
                "even",
            )?,
            strided_slice(
                b,
                x_rotate,
                &[0, 0, 0, 1],
                &[batch, seq, heads, rotary_dim - 1],
                vec![1, 1, 1, 2],
                "odd",
            )?,
        )
    } else {
        (
            strided_slice(
                b,
                x_rotate,
                &[0, 0, 0, 0],
                &half_shape,
                Vec::new(),
                "first_half",
            )?,
            strided_slice(
                b,
                x_rotate,
                &[0, 0, 0, half],
                &half_shape,
                Vec::new(),
                "second_half",
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
        let component = i64_slice_to_mldim(&[batch, seq, heads, half, 1])?;
        let real = reshape_with_shape(b, real, &format!("{label}_real"), component.clone())?;
        let imag = reshape_with_shape(b, imag, &format!("{label}_imag"), component)?;
        let joined = b.builder.concat(&[real, imag], 4).map_err(map_op_error)?;
        reshape_with_shape(
            b,
            joined,
            &format!("{label}_interleaved"),
            i64_slice_to_mldim(&rotate_shape)?,
        )?
    } else {
        b.builder.concat(&[real, imag], 3).map_err(map_op_error)?
    };
    if rotary_dim < head_dim {
        let tail = strided_slice(
            b,
            x,
            &[0, 0, 0, rotary_dim],
            &[batch, seq, heads, head_dim - rotary_dim],
            Vec::new(),
            "tail",
        )?;
        return b.builder.concat(&[rotated, tail], 3).map_err(map_op_error);
    }
    Ok(rotated)
}

fn convert_group_query_attention(
    node: &NodeProto,
    node_name: &str,
    context: &ConversionContext,
    b: &mut OnnxBuilder<'_, '_, '_>,
) -> Result<ConversionResult, OnnxError> {
    let inputs = node.input.as_slice();
    if inputs.len() < 5 {
        return Err(OnnxError::InvalidShape(format!(
            "GroupQueryAttention expects at least 5 inputs, got {}",
            inputs.len()
        )));
    }
    let packed_qkv = inputs[1].is_empty() || inputs[2].is_empty();

    let mut num_heads = 0i64;
    let mut kv_num_heads = 0i64;
    let mut scale = 0.0f32;
    let mut do_rotary = 0i64;
    let mut rotary_interleaved = 0i64;
    let mut softcap = 0.0f32;
    let mut local_window_size = -1i64;
    for attr in &node.attribute {
        match attr.name.as_str() {
            "num_heads" => num_heads = attr.i,
            "kv_num_heads" => kv_num_heads = attr.i,
            "scale" => scale = attr.f,
            "do_rotary" => do_rotary = attr.i,
            "rotary_interleaved" => rotary_interleaved = attr.i,
            "softcap" => softcap = attr.f,
            "local_window_size" => local_window_size = attr.i,
            _ => {}
        }
    }
    let rotary = if do_rotary != 0 {
        let cos_name = inputs.get(7).filter(|n| !n.is_empty());
        let sin_name = inputs.get(8).filter(|n| !n.is_empty());
        match (cos_name, sin_name) {
            (Some(cos), Some(sin)) => Some((cos.clone(), sin.clone())),
            _ => {
                return Err(OnnxError::InvalidShape(format!(
                    "GroupQueryAttention {node_name} has do_rotary=1 but no cos/sin caches"
                )))
            }
        }
    } else {
        None
    };
    if softcap != 0.0 {
        return Err(OnnxError::unsupported_op(
            "GroupQueryAttention(softcap)",
            node_name.to_string(),
        ));
    }
    if local_window_size >= 0 {
        return Err(OnnxError::unsupported_op(
            "GroupQueryAttention(local_window_size)",
            node_name.to_string(),
        ));
    }
    if num_heads <= 0 || kv_num_heads <= 0 || num_heads % kv_num_heads != 0 {
        return Err(OnnxError::InvalidShape(format!(
            "GroupQueryAttention requires num_heads divisible by kv_num_heads, \
             got num_heads={num_heads} kv_num_heads={kv_num_heads}"
        )));
    }

    let q_shape = lookup_shape(&inputs[0], context).ok_or_else(|| {
        OnnxError::InvalidShape(format!(
            "GroupQueryAttention requires a static query shape for '{}'",
            inputs[0]
        ))
    })?;
    let past_shape = lookup_shape(&inputs[3], context).ok_or_else(|| {
        OnnxError::InvalidShape(format!(
            "GroupQueryAttention requires a static past_key shape for '{}'",
            inputs[3]
        ))
    })?;
    if q_shape.len() != 3 || past_shape.len() != 4 || q_shape.iter().any(|&d| d <= 0) {
        return Err(OnnxError::InvalidShape(format!(
            "GroupQueryAttention expects query [B,S,H*Dh] and past_key [B,kvH,P,Dh], \
             got {q_shape:?} and {past_shape:?}"
        )));
    }
    let (batch, seq) = (q_shape[0], q_shape[1]);
    let head_dim = past_shape[3];
    let past_seq = past_shape[2];
    let total_seq = past_seq + seq;
    let expected_hidden = if packed_qkv {
        (num_heads + 2 * kv_num_heads) * head_dim
    } else {
        num_heads * head_dim
    };
    if q_shape[2] != expected_hidden {
        return Err(OnnxError::InvalidShape(format!(
            "GroupQueryAttention hidden {} != expected {expected_hidden}",
            q_shape[2]
        )));
    }
    if scale == 0.0 {
        scale = 1.0 / (head_dim as f32).sqrt();
    }

    let dtype = context
        .value_types
        .get(inputs[0].as_str())
        .copied()
        .unwrap_or(DataType::Float32);
    let label = output_label(node, node_name);

    let (q_in, k_in, v_in) = if packed_qkv {
        let qkv = b.resolve_operand(&inputs[0])?;
        let sizes = [
            (num_heads * head_dim) as u32,
            (kv_num_heads * head_dim) as u32,
            (kv_num_heads * head_dim) as u32,
        ];
        let mut parts = b
            .builder
            .split_with_options(
                qkv,
                &sizes,
                MLSplitOptions {
                    label: format!("{label}_qkv_split"),
                    axis: 2,
                },
            )
            .map_err(map_op_error)?;
        let v = parts.pop().expect("three split outputs");
        let k = parts.pop().expect("three split outputs");
        let q = parts.pop().expect("three split outputs");
        (q, k, v)
    } else {
        (
            b.resolve_operand(&inputs[0])?,
            b.resolve_operand(&inputs[1])?,
            b.resolve_operand(&inputs[2])?,
        )
    };
    let past_key = b.resolve_operand(&inputs[3])?;
    let past_value = b.resolve_operand(&inputs[4])?;

    let (q, k) = if let Some((cos_name, sin_name)) = &rotary {
        let cache_shape = lookup_shape(cos_name, context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "GroupQueryAttention requires a static cos_cache shape for '{cos_name}'"
            ))
        })?;
        let cos_cache = b.resolve_operand(cos_name)?;
        let sin_cache = b.resolve_operand(sin_name)?;
        let mut rotated = Vec::with_capacity(2);
        for (input, heads, tag) in [(q_in, num_heads, "q"), (k_in, kv_num_heads, "k")] {
            let bshd = reshape_with_shape(
                b,
                input,
                &format!("{label}_{tag}_split"),
                i64_slice_to_mldim(&[batch, seq, heads, head_dim])?,
            )?;
            let bshd = apply_rotary(
                b,
                bshd,
                cos_cache,
                sin_cache,
                &cache_shape,
                batch,
                seq,
                past_seq,
                heads,
                head_dim,
                rotary_interleaved != 0,
                &format!("{label}_{tag}_rotary"),
            )?;
            rotated.push(transpose(
                b,
                bshd,
                vec![0, 2, 1, 3],
                &format!("{label}_{tag}_bhsd"),
            )?);
        }
        let k = rotated.pop().expect("two rotated operands");
        let q = rotated.pop().expect("two rotated operands");
        (q, k)
    } else {
        (
            heads_layout(
                b,
                q_in,
                batch,
                seq,
                num_heads,
                head_dim,
                &format!("{label}_q"),
            )?,
            heads_layout(
                b,
                k_in,
                batch,
                seq,
                kv_num_heads,
                head_dim,
                &format!("{label}_k"),
            )?,
        )
    };
    let v = heads_layout(
        b,
        v_in,
        batch,
        seq,
        kv_num_heads,
        head_dim,
        &format!("{label}_v"),
    )?;

    // Append the new keys/values to the cache along the sequence axis. When
    // there is no past (P == 0) the concat degenerates; emit it only if needed.
    let concat_cache = |b: &mut OnnxBuilder<'_, '_, '_>,
                        past: MLOperand,
                        new: MLOperand,
                        tag: &str|
     -> Result<MLOperand, OnnxError> {
        if past_seq == 0 {
            return Ok(new);
        }
        b.builder
            .concat_with_options(
                &[past, new],
                2,
                OnnxBuilder::labeled_options(&format!("{label}_{tag}")),
            )
            .map_err(map_op_error)
    };
    let present_key = concat_cache(b, past_key, k, "present_key")?;
    let present_value = concat_cache(b, past_value, v, "present_value")?;

    // Repeat KV heads across each query-head group: [B,kvH,T,Dh] -> [B,H,T,Dh].
    let group = num_heads / kv_num_heads;
    let expand_heads = |b: &mut OnnxBuilder<'_, '_, '_>,
                        input: MLOperand,
                        tag: &str|
     -> Result<MLOperand, OnnxError> {
        if group == 1 {
            return Ok(input);
        }
        let grouped = reshape_with_shape(
            b,
            input,
            &format!("{label}_{tag}_grouped"),
            i64_slice_to_mldim(&[batch, kv_num_heads, 1, total_seq, head_dim])?,
        )?;
        let expanded = expand_with_shape(
            b,
            grouped,
            &format!("{label}_{tag}_expanded"),
            i64_slice_to_mldim(&[batch, kv_num_heads, group, total_seq, head_dim])?,
        )?;
        reshape_with_shape(
            b,
            expanded,
            &format!("{label}_{tag}_heads"),
            i64_slice_to_mldim(&[batch, num_heads, total_seq, head_dim])?,
        )
    };
    let key_heads = expand_heads(b, present_key, "key")?;
    let value_heads = expand_heads(b, present_value, "value")?;

    // scores = (q * scale) @ k^T + causal_mask
    let scale_name = format!("{label}__scale");
    register_scalar(b, &scale_name, dtype, scale)?;
    let scale_op = b.resolve_operand(&scale_name)?;
    let q_scaled = b
        .builder
        .mul_with_options(
            q,
            scale_op,
            OnnxBuilder::labeled_options(&format!("{label}_q_scaled")),
        )
        .map_err(map_op_error)?;
    let key_t = transpose(b, key_heads, vec![0, 1, 3, 2], &format!("{label}_key_t"))?;
    let scores = b
        .builder
        .matmul_with_options(
            q_scaled,
            key_t,
            OnnxBuilder::labeled_options(&format!("{label}_scores")),
        )
        .map_err(map_op_error)?;

    // Static causal mask [S, T]: key j is visible to query i iff j <= P + i.
    // (Assumes an unpadded batch, i.e. seqlens_k == P + S - 1.)
    let mask_name = format!("{label}__causal_mask");
    register_causal_mask(b, &mask_name, dtype, seq, past_seq, total_seq)?;
    let mask = b.resolve_operand(&mask_name)?;
    let masked = b
        .builder
        .add_with_options(
            scores,
            mask,
            OnnxBuilder::labeled_options(&format!("{label}_masked")),
        )
        .map_err(map_op_error)?;

    let probs = b
        .builder
        .softmax_with_options(
            masked,
            3,
            OnnxBuilder::labeled_options(&format!("{label}_probs")),
        )
        .map_err(map_op_error)?;
    let context_heads = b
        .builder
        .matmul_with_options(
            probs,
            value_heads,
            OnnxBuilder::labeled_options(&format!("{label}_context")),
        )
        .map_err(map_op_error)?;

    // [B,H,S,Dh] -> [B,S,H,Dh] -> [B,S,H*Dh]
    let context_bshd = transpose(
        b,
        context_heads,
        vec![0, 2, 1, 3],
        &format!("{label}_context_bshd"),
    )?;
    let output = reshape_with_shape(
        b,
        context_bshd,
        &format!("{label}_output"),
        i64_slice_to_mldim(&[batch, seq, num_heads * head_dim])?,
    )?;

    let outputs = node.output.as_slice();
    if let Some(out) = outputs.first().filter(|name| !name.is_empty()) {
        record_node_output(b, out, &label, output);
    }
    if let Some(out) = outputs.get(1).filter(|name| !name.is_empty()) {
        record_node_output(b, out, &format!("{label}_present_key_out"), present_key);
    }
    if let Some(out) = outputs.get(2).filter(|name| !name.is_empty()) {
        record_node_output(b, out, &format!("{label}_present_value_out"), present_value);
    }
    Ok(ConversionResult::default())
}

fn register_scalar(
    b: &mut OnnxBuilder<'_, '_, '_>,
    name: &str,
    dtype: DataType,
    value: f32,
) -> Result<(), OnnxError> {
    match dtype {
        DataType::Float16 => b.register_constant_from_bytes(
            name,
            DataType::Float16,
            &[1],
            half::f16::from_f32(value).to_le_bytes().to_vec(),
        ),
        _ => b.register_constant_from_bytes(
            name,
            DataType::Float32,
            &[1],
            value.to_le_bytes().to_vec(),
        ),
    }
}

fn register_causal_mask(
    b: &mut OnnxBuilder<'_, '_, '_>,
    name: &str,
    dtype: DataType,
    seq: i64,
    past_seq: i64,
    total_seq: i64,
) -> Result<(), OnnxError> {
    let (seq, past_seq, total_seq) = (seq as usize, past_seq as usize, total_seq as usize);
    // Large finite negative keeps exp() at exactly 0 without producing NaN
    // from (-inf) - (-inf) style arithmetic inside softmax decompositions.
    let neg = match dtype {
        DataType::Float16 => -60000.0f32,
        _ => -3.0e38f32,
    };
    let mut mask = vec![0f32; seq * total_seq];
    for i in 0..seq {
        for j in 0..total_seq {
            if j > past_seq + i {
                mask[i * total_seq + j] = neg;
            }
        }
    }
    let shape = [seq as u32, total_seq as u32];
    match dtype {
        DataType::Float16 => {
            let bytes: Vec<u8> = mask
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                .collect();
            b.register_constant_from_bytes(name, DataType::Float16, &shape, bytes)
        }
        _ => b.register_constant_from_bytes(
            name,
            DataType::Float32,
            &shape,
            bytemuck::cast_slice(&mask).to_vec(),
        ),
    }
}
