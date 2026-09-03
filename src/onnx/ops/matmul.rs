/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MatMul, Gemm, MatMulInteger, and the com.microsoft quantized matmuls
// (MatMulNBits, MatMulBnb4)

use crate::onnx::builder::{map_op_error, operand_index, tensor_proto_to_bytes, OnnxBuilder};
use crate::onnx::builder_helpers::{
    i64_slice_to_mldim, output_label, record_node_output, reshape_with_shape, slice_with_params,
};
use crate::onnx::convert::{map_onnx_data_type, OnnxError};
use crate::onnx::ops::conv::lookup_shape;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::{NodeProto, TensorProto_DataType};
use rustnn::mlcontext::MLOperand;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLGemmOptions, MLTransposeOptions};
use rustnn::DataType;

/// bitsandbytes 4-bit codebooks, copied verbatim from ONNX Runtime's
/// `blockwise_quant_block_bnb4.h` so dequantization matches ORT bit-for-bit.
const FP4_QUANT_MAP: [f32; 16] = [
    0.0,
    5.208_333_5e-3,
    0.666_666_7,
    1.0,
    0.333_333_34,
    0.5,
    0.166_666_67,
    0.25,
    -0.0,
    -5.208_333_5e-3,
    -0.666_666_7,
    -1.0,
    -0.333_333_34,
    -0.5,
    -0.166_666_67,
    -0.25,
];

const NF4_QUANT_MAP: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// Decode a float32/float16 initializer into f32 values (MatMulBnb4 absmax,
/// QMoE scales).
pub(crate) fn decode_float_tensor_as_f32(
    tensor: &crate::protos::onnx::TensorProto,
) -> Result<Vec<f32>, OnnxError> {
    let bytes = tensor_proto_to_bytes(tensor)?;
    if tensor.data_type == TensorProto_DataType::Float as i32 {
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else if tensor.data_type == TensorProto_DataType::Float16 as i32 {
        Ok(bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect())
    } else {
        Err(OnnxError::InvalidShape(format!(
            "expected float/float16 tensor, got data_type={}",
            tensor.data_type
        )))
    }
}

pub struct MatMulHandler;

impl OpHandler for MatMulHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "MatMul" | "Gemm" | "MatMulNBits" | "MatMulInteger" | "MatMulBnb4"
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
            "MatMul" => self.convert_matmul(node, &node_name, b),
            "Gemm" => self.convert_gemm(node, &node_name, context, b),
            "MatMulNBits" => self.convert_matmul_nbits(node, &node_name, context, b),
            "MatMulInteger" => self.convert_matmul_integer(node, &node_name, context, b),
            "MatMulBnb4" => self.convert_matmul_bnb4(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

impl MatMulHandler {
    fn convert_matmul(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let a = b.resolve_operand(&inputs[0])?;
        let b_in = b.resolve_operand(&inputs[1])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .matmul_with_options(a, b_in, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_gemm(
        &self,
        node: &NodeProto,
        node_name: &str,
        _context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Gemm expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        let mut alpha = 1.0f64;
        let mut beta = 1.0f64;
        let mut trans_a = false;
        let mut trans_b = false;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "alpha" if attr.f != 0.0 => alpha = attr.f as f64,
                "beta" if attr.f != 0.0 => beta = attr.f as f64,
                "transA" if attr.i != 0 => trans_a = true,
                "transB" if attr.i != 0 => trans_b = true,
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let a = b.resolve_operand(&inputs[0])?;
        let b_in = b.resolve_operand(&inputs[1])?;
        let c = inputs
            .get(2)
            .map(|name| b.resolve_operand(name))
            .transpose()?;

        let opts = MLGemmOptions {
            label: output_name.clone(),
            alpha,
            beta,
            a_transpose: trans_a,
            b_transpose: trans_b,
            c: c.map(operand_index),
        };
        let out = b
            .builder
            .gemm_with_options(a, b_in, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    /// Lower `com.microsoft.MatMulBnb4` by dequantizing the constant packed
    /// weights at conversion time and emitting a plain `matmul`.
    ///
    /// bitsandbytes FP4/NF4 uses a 16-entry codebook (not affine quantization),
    /// so WebNN `dequantizeLinear` cannot express it; since `B`/`absmax` are
    /// always initializers, decoding them into a dense `[K, N]` float constant
    /// is exact. Note this trades the 4-bit weight footprint for full-precision
    /// constants in the WebNN graph.
    /// MatMulBnb4 with the weight kept packed: unpack nibbles, look them up
    /// in the codebook, scale per block, then matmul.
    #[allow(clippy::too_many_arguments)]
    fn convert_matmul_bnb4_packed(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
        packed: &[u8],
        quant_map: &[f32; 16],
        absmax: &[f32],
        n: i64,
        k: i64,
        block_size: i64,
    ) -> Result<ConversionResult, OnnxError> {
        use crate::onnx::builder::map_ast_data_type;
        use rustnn::operator_options::MLGatherOptions;

        let label = output_label(node, node_name);
        let a_name = node
            .input
            .as_slice()
            .first()
            .map(|s| s.as_str())
            .unwrap_or("");
        let dtype = context
            .value_types
            .get(a_name)
            .copied()
            .unwrap_or(DataType::Float32);
        let total = (n * k) as usize;
        let n_blocks = absmax.len() as i64;
        let half = packed.len() as u32;

        let const_bytes = |b: &mut OnnxBuilder<'_, '_, '_>,
                           tag: &str,
                           dt: DataType,
                           shape: &[u32],
                           bytes: &[u8]|
         -> Result<MLOperand, OnnxError> {
            let name = format!("{label}__{tag}");
            b.register_constant_from_bytes(&name, dt, shape, bytes)?;
            b.resolve_operand(&name)
        };
        let float_bytes = |values: &[f32]| -> Vec<u8> {
            match dtype {
                DataType::Float16 => values
                    .iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                    .collect(),
                _ => bytemuck::cast_slice(values).to_vec(),
            }
        };

        let blob = const_bytes(b, "blob", DataType::Uint8, &[half], packed)?;
        let c16 = const_bytes(b, "c16", DataType::Int32, &[1], &16i32.to_le_bytes())?;
        let codebook = const_bytes(b, "codebook", dtype, &[16], &float_bytes(quant_map))?;
        let scales = const_bytes(
            b,
            "absmax",
            dtype,
            &[n_blocks as u32, 1],
            &float_bytes(absmax),
        )?;

        let opts = |tag: &str| OnnxBuilder::labeled_options(&format!("{label}__{tag}"));
        let x = b
            .builder
            .cast_with_options(blob, map_ast_data_type(DataType::Int32)?, opts("i32"))
            .map_err(map_op_error)?;
        let hi = b
            .builder
            .div_with_options(x, c16, opts("hi"))
            .map_err(map_op_error)?;
        let hi16 = b
            .builder
            .mul_with_options(hi, c16, opts("hi16"))
            .map_err(map_op_error)?;
        let lo = b
            .builder
            .sub_with_options(x, hi16, opts("lo"))
            .map_err(map_op_error)?;
        let column = i64_slice_to_mldim(&[i64::from(half), 1])?;
        let hi = reshape_with_shape(b, hi, &format!("{label}__hi_col"), column.clone())?;
        let lo = reshape_with_shape(b, lo, &format!("{label}__lo_col"), column)?;
        // Element 2i is the high nibble, 2i+1 the low nibble.
        let codes = b
            .builder
            .concat_with_options(&[hi, lo], 1, opts("codes"))
            .map_err(map_op_error)?;
        let codes = reshape_with_shape(
            b,
            codes,
            &format!("{label}__codes_flat"),
            i64_slice_to_mldim(&[total as i64])?,
        )?;
        let values = b
            .builder
            .gather_with_options(
                codebook,
                codes,
                MLGatherOptions {
                    label: format!("{label}__lookup"),
                    axis: 0,
                },
            )
            .map_err(map_op_error)?;
        let blocked = reshape_with_shape(
            b,
            values,
            &format!("{label}__blocked"),
            i64_slice_to_mldim(&[n_blocks, block_size])?,
        )?;
        let scaled = b
            .builder
            .mul_with_options(blocked, scales, opts("scaled"))
            .map_err(map_op_error)?;
        let weights_nk = reshape_with_shape(
            b,
            scaled,
            &format!("{label}__weights_nk"),
            i64_slice_to_mldim(&[n, k])?,
        )?;
        let weights = b
            .builder
            .transpose_with_options(
                weights_nk,
                MLTransposeOptions {
                    label: format!("{label}__weights_kn"),
                    permutation: vec![1, 0],
                },
            )
            .map_err(map_op_error)?;

        let a = b.resolve_operand(a_name)?;
        let out = b
            .builder
            .matmul_with_options(a, weights, OnnxBuilder::labeled_options(&label))
            .map_err(map_op_error)?;
        if let Some(onnx_out) = node.output.first().filter(|name| !name.is_empty()) {
            record_node_output(b, onnx_out, &label, out);
        } else {
            b.record_operand(&[&label], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_matmul_bnb4(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulBnb4 expects 3 inputs (A, B, absmax), got {}",
                inputs.len()
            )));
        }

        let mut k = 0i64;
        let mut n = 0i64;
        let mut block_size = 0i64;
        let mut quant_type = 1i64;
        for attr in &node.attribute {
            match attr.name.as_str() {
                "K" => k = attr.i,
                "N" => n = attr.i,
                "block_size" => block_size = attr.i,
                "quant_type" => quant_type = attr.i,
                _ => {}
            }
        }
        if k <= 0 || n <= 0 || block_size <= 0 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulBnb4 requires positive K/N/block_size, got K={k} N={n} block_size={block_size}"
            )));
        }
        let quant_map: &[f32; 16] = match quant_type {
            0 => &FP4_QUANT_MAP,
            1 => &NF4_QUANT_MAP,
            other => {
                return Err(OnnxError::unsupported_op(
                    format!("MatMulBnb4(quant_type={other})"),
                    node_name.to_string(),
                ));
            }
        };

        let b_tensor = context
            .initializers
            .get(inputs[1].as_str())
            .copied()
            .ok_or_else(|| {
                OnnxError::unsupported_op("MatMulBnb4(non-constant B)", node_name.to_string())
            })?;
        if b_tensor.data_type != TensorProto_DataType::Uint8 as i32 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulBnb4 B must be packed uint8, got data_type={}",
                b_tensor.data_type
            )));
        }
        let absmax_tensor = context
            .initializers
            .get(inputs[2].as_str())
            .copied()
            .ok_or_else(|| {
                OnnxError::unsupported_op("MatMulBnb4(non-constant absmax)", node_name.to_string())
            })?;

        let total = (n as usize)
            .checked_mul(k as usize)
            .ok_or_else(|| OnnxError::InvalidShape("MatMulBnb4 N*K overflow".into()))?;
        let packed = tensor_proto_to_bytes(b_tensor)?;
        if packed.len() < total.div_ceil(2) {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulBnb4 B holds {} bytes, need {} for N*K={total} nibbles",
                packed.len(),
                total.div_ceil(2)
            )));
        }
        let absmax = decode_float_tensor_as_f32(absmax_tensor)?;
        let n_blocks = total.div_ceil(block_size as usize);
        if absmax.len() < n_blocks {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulBnb4 absmax holds {} values, need {n_blocks}",
                absmax.len()
            )));
        }

        // Dense fallback below only when the weight is not a whole number of blocks.
        if total % block_size as usize == 0 {
            return self.convert_matmul_bnb4_packed(
                node,
                node_name,
                context,
                b,
                &packed[..total.div_ceil(2)],
                quant_map,
                &absmax[..n_blocks],
                n,
                k,
                block_size,
            );
        }

        // B is the flattened row-major [N, K] weight; emit it transposed as
        // [K, N] so the matmul consumes it directly. Element 2i sits in the
        // high nibble, 2i+1 in the low nibble (bitsandbytes packing).
        let (n_usize, k_usize) = (n as usize, k as usize);
        let mut weights_kn = vec![0f32; total];
        for idx in 0..total {
            let byte = packed[idx / 2];
            let code = if idx % 2 == 0 { byte >> 4 } else { byte & 0x0F };
            let value = quant_map[code as usize] * absmax[idx / block_size as usize];
            let (row, col) = (idx / k_usize, idx % k_usize);
            weights_kn[col * n_usize + row] = value;
        }

        let label = output_label(node, node_name);
        let a = b.resolve_operand(&inputs[0])?;
        let a_is_f16 = matches!(
            context.value_types.get(inputs[0].as_str()),
            Some(DataType::Float16)
        );
        let weights_name = format!("{label}__weights_kn");
        let weight_shape = [k as u32, n as u32];
        if a_is_f16 {
            let bytes: Vec<u8> = weights_kn
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                .collect();
            b.register_constant_from_bytes(
                &weights_name,
                DataType::Float16,
                &weight_shape,
                &bytes,
            )?;
        } else {
            b.register_constant_from_bytes(
                &weights_name,
                DataType::Float32,
                &weight_shape,
                bytemuck::cast_slice(&weights_kn),
            )?;
        }
        let weights = b.resolve_operand(&weights_name)?;

        let out = b
            .builder
            .matmul_with_options(a, weights, OnnxBuilder::labeled_options(&label))
            .map_err(map_op_error)?;
        if let Some(onnx_out) = node.output.first().filter(|name| !name.is_empty()) {
            record_node_output(b, onnx_out, &label, out);
        }
        Ok(ConversionResult::default())
    }

    /// Lower `MatMulInteger` as centered float matmul, mirroring `ConvInteger`:
    /// cast both operands to float32, subtract their zero points, `matmul`,
    /// and cast the product back to `int32`.
    fn convert_matmul_integer(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 || inputs.len() > 4 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulInteger expects 2 to 4 inputs (A, B[, a_zero_point, b_zero_point]), got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let a = self.centered_integer_operand(
            b,
            &inputs[0],
            inputs
                .get(2)
                .filter(|name| !name.is_empty())
                .map(String::as_str),
            context,
            // A per-row zero point [M] must become [M, 1] to broadcast over K.
            Some(&[-1, 1]),
            &format!("{output_name}_a"),
        )?;
        let b_in = self.centered_integer_operand(
            b,
            &inputs[1],
            inputs
                .get(3)
                .filter(|name| !name.is_empty())
                .map(String::as_str),
            context,
            // B per-column zero point [N] already aligns with the trailing dim.
            None,
            &format!("{output_name}_b"),
        )?;

        let product = b
            .builder
            .matmul_with_options(
                a,
                b_in,
                OnnxBuilder::labeled_options(&format!("{output_name}_matmul")),
            )
            .map_err(map_op_error)?;
        let out = b
            .builder
            .cast_with_options(
                product,
                MLOperandDataType::Int32,
                OnnxBuilder::labeled_options(&output_name),
            )
            .map_err(map_op_error)?;

        let mut result = ConversionResult::default();
        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
            result
                .output_types
                .insert(onnx_out.clone(), DataType::Int32);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(result)
    }

    /// Cast a quantized operand to float32 and subtract its (optional) zero
    /// point. `vector_zp_shape` reshapes a 1-D zero point before subtraction;
    /// `-1` stands for the zero point's own length.
    fn centered_integer_operand(
        &self,
        b: &mut OnnxBuilder<'_, '_, '_>,
        operand_name: &str,
        zero_point_name: Option<&str>,
        context: &ConversionContext,
        vector_zp_shape: Option<&[i64]>,
        label: &str,
    ) -> Result<MLOperand, OnnxError> {
        let operand = b.resolve_operand(operand_name)?;
        let as_float = b
            .builder
            .cast_with_options(
                operand,
                MLOperandDataType::Float32,
                OnnxBuilder::labeled_options(&format!("{label}_float")),
            )
            .map_err(map_op_error)?;
        let Some(zero_point_name) = zero_point_name else {
            return Ok(as_float);
        };

        let zero_point = b.resolve_operand(zero_point_name)?;
        let mut zero_point_float = b
            .builder
            .cast_with_options(
                zero_point,
                MLOperandDataType::Float32,
                OnnxBuilder::labeled_options(&format!("{label}_zero_point_float")),
            )
            .map_err(map_op_error)?;

        if let Some(template) = vector_zp_shape {
            let zp_shape = lookup_shape(zero_point_name, context);
            if let Some(zp_shape) = zp_shape.filter(|s| s.len() == 1 && s[0] > 1) {
                let target: Vec<i64> = template
                    .iter()
                    .map(|&d| if d == -1 { zp_shape[0] } else { d })
                    .collect();
                zero_point_float = reshape_with_shape(
                    b,
                    zero_point_float,
                    &format!("{label}_zero_point_reshape"),
                    i64_slice_to_mldim(&target)?,
                )?;
            }
        }

        b.builder
            .sub_with_options(
                as_float,
                zero_point_float,
                OnnxBuilder::labeled_options(&format!("{label}_centered")),
            )
            .map_err(map_op_error)
    }

    /// Lower `com.microsoft.MatMulNBits` the same way ORT's WebNN EP does:
    /// `dequantizeLinear` -> reshape `[N,K]` -> transpose `[K,N]` -> `matmul` (+ optional bias).
    ///
    /// Supported: bits=4, constant packed `B`, constant `scales`, optional constant
    /// zero_points, optional bias.
    /// Rejected: bits!=4, `g_idx`, non-constant `B`/`scales`/`zero_points`.
    fn convert_matmul_nbits(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits expects at least 3 inputs (A, B, scales), got {}",
                inputs.len()
            )));
        }

        let mut k = 0i64;
        let mut n = 0i64;
        let mut bits = 4i64;
        let mut block_size = 32i64;
        for attr in &node.attribute {
            match attr.name.as_str() {
                "K" => k = attr.i,
                "N" => n = attr.i,
                "bits" => bits = attr.i,
                "block_size" => block_size = attr.i,
                _ => {}
            }
        }
        if bits != 4 && bits != 8 {
            return Err(OnnxError::unsupported_op(
                format!("MatMulNBits(bits={bits})"),
                node_name.to_string(),
            ));
        }
        if k <= 0 || n <= 0 || block_size < 16 || !(block_size as u64).is_power_of_two() {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits requires positive K/N and power-of-two block_size>=16, \
                 got K={k} N={n} block_size={block_size}"
            )));
        }
        if inputs.get(4).is_some_and(|name| !name.is_empty()) {
            return Err(OnnxError::unsupported_op(
                "MatMulNBits(g_idx)",
                node_name.to_string(),
            ));
        }

        let b_name = inputs[1].as_str();
        let scales_name = inputs[2].as_str();
        let zero_points_name = inputs
            .get(3)
            .filter(|name| !name.is_empty())
            .map(String::as_str);
        let bias_name = inputs
            .get(5)
            .filter(|name| !name.is_empty())
            .map(String::as_str);

        let b_tensor = context.initializers.get(b_name).copied().ok_or_else(|| {
            OnnxError::unsupported_op("MatMulNBits(non-constant B)", node_name.to_string())
        })?;
        if b_tensor.data_type != TensorProto_DataType::Uint8 as i32 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits B must be uint8 packed weights, got data_type={}",
                b_tensor.data_type
            )));
        }
        if b_tensor.dims.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits B must have shape [N, n_blocks, blob_size], got {:?}",
                b_tensor.dims
            )));
        }
        let n_attr = n as u32;
        let k_attr = k as u32;
        let block_size_u = block_size as u32;
        let n_blocks = b_tensor.dims[1] as u32;
        let blob_size = b_tensor.dims[2] as u32;
        if b_tensor.dims[0] != n {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits B dim0 {} does not match N={n}",
                b_tensor.dims[0]
            )));
        }
        let expected_blocks = k_attr.div_ceil(block_size_u);
        if n_blocks != expected_blocks {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits n_blocks {n_blocks} != ceil(K/block_size)={expected_blocks}"
            )));
        }
        let expected_blob = (block_size_u * bits as u32).div_ceil(8);
        if blob_size != expected_blob {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits blob_size {blob_size} != block_size*bits/8={expected_blob}"
            )));
        }

        let label = output_label(node, node_name);
        let packed = tensor_proto_to_bytes(b_tensor)?;
        // 4-bit: reinterpret packed blobs as uint4 with doubled last dim
        // (= block_size); 8-bit blobs are already one value per byte.
        let (weight_dtype, weight_shape) = if bits == 4 {
            (DataType::Uint4, [n_attr, n_blocks, blob_size * 2])
        } else {
            (DataType::Uint8, [n_attr, n_blocks, blob_size])
        };
        let b_uint4_name = format!("{label}__B_quant");
        b.register_constant_from_bytes(&b_uint4_name, weight_dtype, &weight_shape, &packed)?;
        let b_uint4 = b.resolve_operand(&b_uint4_name)?;

        let scales_tensor = context
            .initializers
            .get(scales_name)
            .copied()
            .ok_or_else(|| {
                OnnxError::unsupported_op("MatMulNBits(non-constant scales)", node_name.to_string())
            })?;
        let scales_dtype = map_onnx_data_type(scales_tensor.data_type)?;
        let scales_bytes = tensor_proto_to_bytes(scales_tensor)?;
        let scales_shape_name = format!("{label}__scales");
        b.register_constant_from_bytes(
            &scales_shape_name,
            scales_dtype,
            &[n_attr, n_blocks, 1],
            &scales_bytes,
        )?;
        let scales = b.resolve_operand(&scales_shape_name)?;

        let zero_point = register_matmul_nbits_zero_point(
            b,
            context,
            zero_points_name,
            n_attr,
            n_blocks,
            bits,
            &format!("{label}__zero_point"),
        )?;

        let dequantized = b
            .builder
            .dequantize_linear_with_zeropoint(b_uint4, scales, zero_point)
            .map_err(map_op_error)?;
        // The last block may be zero-padded (K not a multiple of block_size,
        // e.g. K=8 with block_size=32): reshape to the padded width and slice
        // the valid K columns before use.
        let padded_k = (n_blocks * block_size_u) as i64;
        let weights = if padded_k != k {
            let padded = reshape_with_shape(
                b,
                dequantized,
                &format!("{label}__weights_nk_padded"),
                i64_slice_to_mldim(&[n, padded_k])?,
            )?;
            slice_with_params(
                b,
                padded,
                &format!("{label}__weights_nk"),
                &[0, 0],
                &[
                    rustnn::operator_options::MLDimension::Static(n_attr),
                    rustnn::operator_options::MLDimension::Static(k_attr),
                ],
            )?
        } else {
            reshape_with_shape(
                b,
                dequantized,
                &format!("{label}__weights_nk"),
                i64_slice_to_mldim(&[n, k])?,
            )?
        };
        let weights = b
            .builder
            .transpose_with_options(
                weights,
                MLTransposeOptions {
                    label: format!("{label}__weights_kn"),
                    permutation: vec![1, 0],
                },
            )
            .map_err(map_op_error)?;

        let a = b.resolve_operand(&inputs[0])?;
        let mut out = b
            .builder
            .matmul_with_options(a, weights, OnnxBuilder::labeled_options(&label))
            .map_err(map_op_error)?;
        if let Some(bias_name) = bias_name {
            let bias = b.resolve_operand(bias_name)?;
            out = b
                .builder
                .add_with_options(
                    out,
                    bias,
                    OnnxBuilder::labeled_options(&format!("{label}__bias")),
                )
                .map_err(map_op_error)?;
        }

        if let Some(onnx_out) = node.output.first().filter(|name| !name.is_empty()) {
            record_node_output(b, onnx_out, &label, out);
        }
        Ok(ConversionResult::default())
    }
}

fn register_matmul_nbits_zero_point(
    b: &mut OnnxBuilder<'_, '_, '_>,
    context: &ConversionContext,
    zero_points_name: Option<&str>,
    n: u32,
    n_blocks: u32,
    bits: i64,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    let zp_shape = [n, n_blocks, 1];
    let element_count = (n as usize)
        .checked_mul(n_blocks as usize)
        .ok_or_else(|| OnnxError::InvalidShape("MatMulNBits zero_point size overflow".into()))?;
    // 4-bit zero points are packed two per byte; 8-bit are one per byte.
    let (packed_len, default_byte, zp_dtype) = if bits == 4 {
        (element_count.div_ceil(2), 0x88u8, DataType::Uint4)
    } else {
        (element_count, 0x80u8, DataType::Uint8)
    };

    let packed = if let Some(name) = zero_points_name {
        let tensor = context.initializers.get(name).copied().ok_or_else(|| {
            OnnxError::unsupported_op("MatMulNBits(non-constant zero_points)", label.to_string())
        })?;
        if tensor.data_type != TensorProto_DataType::Uint8 as i32 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits zero_points must be packed uint8, got data_type={}",
                tensor.data_type
            )));
        }
        let bytes = tensor_proto_to_bytes(tensor)?;
        if bytes.len() != packed_len {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits zero_points packed length {} != expected {packed_len}",
                bytes.len()
            )));
        }
        bytes
    } else {
        vec![default_byte; packed_len]
    };

    b.register_constant_from_bytes(label, zp_dtype, &zp_shape, &packed)?;
    b.resolve_operand(label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, TensorProto, TensorProto_DataType};
    use rustnn::graph::pack_uint4;
    use std::collections::HashMap;

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
    fn test_matmul_handler_supports() {
        let handler = MatMulHandler;
        assert!(handler.supports("MatMul"));
        assert!(handler.supports("Gemm"));
        assert!(handler.supports("MatMulNBits"));
    }

    #[test]
    fn test_convert_matmul() {
        let handler = MatMulHandler;
        let node = create_test_node("MatMul", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_gemm_simple() {
        let handler = MatMulHandler;
        let node = create_test_node("Gemm", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn converts_matmul_nbits_4bit_without_zero_points() {
        let handler = MatMulHandler;
        let mut node = create_test_node("MatMulNBits", vec!["a", "b_q4", "scales"], vec!["y"]);
        node.domain = "com.microsoft".to_string();
        node.attribute = vec![
            AttributeProto {
                name: "K".to_string(),
                i: 32,
                ..Default::default()
            },
            AttributeProto {
                name: "N".to_string(),
                i: 16,
                ..Default::default()
            },
            AttributeProto {
                name: "bits".to_string(),
                i: 4,
                ..Default::default()
            },
            AttributeProto {
                name: "block_size".to_string(),
                i: 32,
                ..Default::default()
            },
        ];

        // B: [N=16, n_blocks=1, blob_size=16] packed uint8 (= 512 uint4 values).
        let values: Vec<u8> = (0..512).map(|v| (v % 16) as u8).collect();
        let packed = pack_uint4(&values);
        let b_tensor = TensorProto {
            name: "b_q4".to_string(),
            data_type: TensorProto_DataType::Uint8 as i32,
            dims: vec![16, 1, 16],
            raw_data: packed,
            ..Default::default()
        };
        let scale_bytes: Vec<u8> = (0..16).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        let scales = TensorProto {
            name: "scales".to_string(),
            data_type: TensorProto_DataType::Float as i32,
            dims: vec![16, 1],
            raw_data: scale_bytes,
            ..Default::default()
        };

        let mut initializers = HashMap::new();
        initializers.insert("b_q4".to_string(), &b_tensor);
        initializers.insert("scales".to_string(), &scales);
        let value_shapes = HashMap::from([
            ("a".to_string(), vec![2, 32]),
            ("b_q4".to_string(), vec![16, 1, 16]),
            ("scales".to_string(), vec![16, 1]),
        ]);
        let value_types = HashMap::from([
            ("a".to_string(), DataType::Float32),
            ("b_q4".to_string(), DataType::Uint8),
            ("scales".to_string(), DataType::Float32),
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

    #[test]
    fn rejects_matmul_nbits_with_g_idx() {
        let handler = MatMulHandler;
        let mut node = create_test_node(
            "MatMulNBits",
            vec!["a", "b", "scales", "", "g_idx"],
            vec!["y"],
        );
        node.attribute = vec![
            AttributeProto {
                name: "K".to_string(),
                i: 32,
                ..Default::default()
            },
            AttributeProto {
                name: "N".to_string(),
                i: 16,
                ..Default::default()
            },
            AttributeProto {
                name: "bits".to_string(),
                i: 4,
                ..Default::default()
            },
            AttributeProto {
                name: "block_size".to_string(),
                i: 32,
                ..Default::default()
            },
        ];
        let err = crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedOps(_)));
    }
}
