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

// Convolution operators: Conv, ConvTranspose, ConvInteger
//
// Maps ONNX Conv/ConvTranspose to WebNN conv2d/convTranspose2d (NCHW layout).
//
// ONNX layout assumptions (the spec defaults):
//   * input X : (N, C_in, ...spatial)
//   * filter W: Conv          -> (M, C_in / group, kH, kW, ...)
//                ConvTranspose -> (C_in, M / group, kH, kW, ...)
//   * bias B  : (M,)  (optional)
//
// WebNN defaults match ONNX:
//   * inputLayout  = "nchw"
//   * filterLayout = "oihw" for conv2d
//   * filterLayout = "iohw" for convTranspose2d
//
// Spatial dimensionality:
//   * 2D (4-D input)              -> conv2d / convTranspose2d directly
//   * 1D (3-D input)              -> reshape -> conv2d -> reshape
//   * Anything else (1D w/o shape info, 3D, etc.) -> UnsupportedOp error.

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{
    i64_slice_to_mldim, optional_operand_index, output_label, record_node_output,
};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLConv2dOptions, MLConvTranspose2dOptions};
use rustnn::DataType;

pub struct ConvHandler;

impl OpHandler for ConvHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Conv" | "ConvTranspose" | "ConvInteger")
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Conv" => self.convert_conv(node, &node_name, context, b, false),
            "ConvTranspose" => self.convert_conv(node, &node_name, context, b, true),
            "ConvInteger" => self.convert_conv_integer(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

#[derive(Debug, Clone)]
struct ConvAttrs {
    auto_pad: String,
    dilations: Option<Vec<i64>>,
    group: i64,
    kernel_shape: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    output_padding: Option<Vec<i64>>,
    output_shape: Option<Vec<i64>>,
}

fn parse_conv_attrs(node: &NodeProto) -> ConvAttrs {
    let mut attrs = ConvAttrs {
        auto_pad: "NOTSET".to_string(),
        dilations: None,
        group: 1,
        kernel_shape: None,
        pads: None,
        strides: None,
        output_padding: None,
        output_shape: None,
    };

    for attr in node.attribute.as_slice() {
        match attr.name.as_str() {
            "auto_pad" => {
                if let Ok(s) = String::from_utf8(attr.s.clone()) {
                    if !s.is_empty() {
                        attrs.auto_pad = s;
                    }
                }
            }
            "dilations" if !attr.ints.is_empty() => {
                attrs.dilations = Some(attr.ints.clone());
            }
            "group" if attr.i > 0 => {
                attrs.group = attr.i;
            }
            "kernel_shape" if !attr.ints.is_empty() => {
                attrs.kernel_shape = Some(attr.ints.clone());
            }
            "pads" if !attr.ints.is_empty() => {
                attrs.pads = Some(attr.ints.clone());
            }
            "strides" if !attr.ints.is_empty() => {
                attrs.strides = Some(attr.ints.clone());
            }
            "output_padding" if !attr.ints.is_empty() => {
                attrs.output_padding = Some(attr.ints.clone());
            }
            "output_shape" if !attr.ints.is_empty() => {
                attrs.output_shape = Some(attr.ints.clone());
            }
            _ => {}
        }
    }

    attrs
}

pub(crate) fn lookup_shape(name: &str, context: &ConversionContext) -> Option<Vec<i64>> {
    if let Some(s) = context.value_shapes.get(name) {
        return Some(s.clone());
    }
    let sanitized = sanitize_identifier(name);
    if let Some(s) = context.value_shapes.get(&sanitized) {
        return Some(s.clone());
    }
    if let Some(init) = context.initializers.get(name) {
        return Some(init.dims.as_slice().to_vec());
    }
    None
}

fn onnx_pads_to_webnn(pads: &[i64], spatial_rank: usize) -> Vec<i64> {
    if pads.len() != 2 * spatial_rank {
        return pads.to_vec();
    }
    let mut out = Vec::with_capacity(2 * spatial_rank);
    for i in 0..spatial_rank {
        out.push(pads[i]);
        out.push(pads[i + spatial_rank]);
    }
    out
}

fn i64_vec_to_u32(values: &[i64]) -> Result<Vec<u32>, OnnxError> {
    values
        .iter()
        .map(|&v| {
            u32::try_from(v).map_err(|_| {
                OnnxError::InvalidShape(format!("negative or oversized conv option value: {v}"))
            })
        })
        .collect()
}

fn build_conv2d_options(attrs: &ConvAttrs, label: &str) -> Result<MLConv2dOptions, OnnxError> {
    let strides = attrs.strides.clone().unwrap_or_else(|| vec![1, 1]);
    let dilations = attrs.dilations.clone().unwrap_or_else(|| vec![1, 1]);
    let pads = attrs.pads.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);

    if strides.len() != 2 {
        return Err(OnnxError::InvalidShape(format!(
            "conv2d: strides must have length 2, got {:?}",
            strides
        )));
    }
    if dilations.len() != 2 {
        return Err(OnnxError::InvalidShape(format!(
            "conv2d: dilations must have length 2, got {:?}",
            dilations
        )));
    }

    let mut opts = MLConv2dOptions {
        label: label.to_string(),
        strides: i64_vec_to_u32(&strides)?,
        dilations: i64_vec_to_u32(&dilations)?,
        groups: attrs.group as u32,
        ..Default::default()
    };

    if map_auto_pad(&attrs.auto_pad) == "explicit" {
        let effective_pads = if attrs.auto_pad == "VALID" {
            vec![0, 0, 0, 0]
        } else {
            onnx_pads_to_webnn(&pads, 2)
        };
        if effective_pads.len() != 4 {
            return Err(OnnxError::InvalidShape(format!(
                "conv2d: pads must yield 4 values for 2D, got {:?}",
                effective_pads
            )));
        }
        opts.padding = i64_vec_to_u32(&effective_pads)?;
    }

    Ok(opts)
}

fn build_conv_transpose2d_options(
    attrs: &ConvAttrs,
    label: &str,
) -> Result<MLConvTranspose2dOptions, OnnxError> {
    let strides = attrs.strides.clone().unwrap_or_else(|| vec![1, 1]);
    let dilations = attrs.dilations.clone().unwrap_or_else(|| vec![1, 1]);
    let pads = attrs.pads.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);

    if strides.len() != 2 || dilations.len() != 2 {
        return Err(OnnxError::InvalidShape(format!(
            "convTranspose2d: expected length-2 strides/dilations, got strides={:?} dilations={:?}",
            strides, dilations
        )));
    }

    let mut opts = MLConvTranspose2dOptions {
        label: label.to_string(),
        strides: i64_vec_to_u32(&strides)?,
        dilations: i64_vec_to_u32(&dilations)?,
        groups: attrs.group as u32,
        ..Default::default()
    };

    if map_auto_pad(&attrs.auto_pad) == "explicit" {
        let effective_pads = if attrs.auto_pad == "VALID" {
            vec![0, 0, 0, 0]
        } else {
            onnx_pads_to_webnn(&pads, 2)
        };
        if effective_pads.len() != 4 {
            return Err(OnnxError::InvalidShape(format!(
                "convTranspose2d: pads must yield 4 values for 2D, got {:?}",
                effective_pads
            )));
        }
        opts.padding = i64_vec_to_u32(&effective_pads)?;
    }

    if let Some(op) = attrs.output_padding.as_ref() {
        if op.len() == 2 {
            opts.output_padding = i64_vec_to_u32(op)?;
        }
    }
    if let Some(os) = attrs.output_shape.as_ref() {
        let sizes: Vec<i64> = if os.len() == 2 {
            os.clone()
        } else if os.len() >= 2 {
            os[os.len() - 2..].to_vec()
        } else {
            Vec::new()
        };
        if sizes.len() == 2 {
            opts.output_sizes = Some(i64_vec_to_u32(&sizes)?);
        }
    }

    Ok(opts)
}

fn map_auto_pad(auto_pad: &str) -> &'static str {
    match auto_pad {
        "SAME_UPPER" => "same-upper",
        "SAME_LOWER" => "same-lower",
        _ => "explicit",
    }
}

fn conv2d_spatial_out(
    in_spatial: i64,
    kernel: i64,
    stride: i64,
    dilation: i64,
    pad_begin: i64,
    pad_end: i64,
) -> i64 {
    let effective_kernel = (kernel - 1) * dilation + 1;
    (in_spatial + pad_begin + pad_end - effective_kernel) / stride + 1
}

impl ConvHandler {
    fn convert_conv_integer(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 || inputs.len() > 4 {
            return Err(OnnxError::InvalidShape(format!(
                "ConvInteger expects 2 to 4 inputs (X, W[, x_zero_point, w_zero_point]), got {}",
                inputs.len()
            )));
        }

        let input_raw = inputs[0].as_str();
        let filter_raw = inputs[1].as_str();
        let input_shape = lookup_shape(input_raw, context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "ConvInteger requires known input shape for '{input_raw}'"
            ))
        })?;
        let filter_shape = lookup_shape(filter_raw, context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "ConvInteger requires known filter shape for '{filter_raw}'"
            ))
        })?;
        let rank = input_shape.len();
        if !(rank == 3 || rank == 4) || filter_shape.len() != rank {
            return Err(OnnxError::unsupported_op(
                format!("ConvInteger{}D", input_shape.len().saturating_sub(2)),
                node_name,
            ));
        }

        let output_name = output_label(node, node_name);
        let input = self.center_quantized_operand(
            b,
            input_raw,
            inputs
                .get(2)
                .filter(|name| !name.is_empty())
                .map(String::as_str),
            context,
            false,
            &input_shape,
            &format!("{output_name}_input"),
        )?;
        let filter = self.center_quantized_operand(
            b,
            filter_raw,
            inputs
                .get(3)
                .filter(|name| !name.is_empty())
                .map(String::as_str),
            context,
            true,
            &filter_shape,
            &format!("{output_name}_filter"),
        )?;

        let mut attrs = parse_conv_attrs(node);
        // 1-D ConvInteger: run the centered float operands through the same
        // trailing-unit-dim conv2d emulation regular 1-D Conv uses.
        let (input, filter) = if rank == 3 {
            attrs.strides = Some(extend_with_one(attrs.strides.as_deref(), 1, 2));
            attrs.dilations = Some(extend_with_one(attrs.dilations.as_deref(), 1, 2));
            attrs.pads = Some(extend_pads_to_2d(attrs.pads.as_deref()));
            attrs.kernel_shape = attrs
                .kernel_shape
                .as_ref()
                .map(|ks| extend_with_one(Some(ks.as_slice()), 1, 2));
            let x4d = b
                .builder
                .reshape_with_options(
                    input,
                    i64_slice_to_mldim(&[input_shape[0], input_shape[1], input_shape[2], 1])?,
                    OnnxBuilder::labeled_options(&format!("{output_name}_x4d")),
                )
                .map_err(map_op_error)?;
            let w4d = b
                .builder
                .reshape_with_options(
                    filter,
                    i64_slice_to_mldim(&[filter_shape[0], filter_shape[1], filter_shape[2], 1])?,
                    OnnxBuilder::labeled_options(&format!("{output_name}_w4d")),
                )
                .map_err(map_op_error)?;
            (x4d, w4d)
        } else {
            (input, filter)
        };
        let conv = b
            .builder
            .conv2_with_options(
                input,
                filter,
                build_conv2d_options(&attrs, &format!("{output_name}_conv2d"))?,
            )
            .map_err(map_op_error)?;
        let mut out = b
            .builder
            .cast_with_options(
                conv,
                MLOperandDataType::Int32,
                OnnxBuilder::labeled_options(&format!("{output_name}_i32")),
            )
            .map_err(map_op_error)?;
        if rank == 3 {
            let strides = attrs.strides.clone().unwrap_or_else(|| vec![1, 1]);
            let dilations = attrs.dilations.clone().unwrap_or_else(|| vec![1, 1]);
            let pads = attrs.pads.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);
            let effective_pads = if attrs.auto_pad == "VALID" {
                vec![0, 0, 0, 0]
            } else if map_auto_pad(&attrs.auto_pad) == "explicit" {
                onnx_pads_to_webnn(&pads, 2)
            } else {
                vec![0, 0, 0, 0]
            };
            let spatial_out = conv2d_spatial_out(
                input_shape[2],
                filter_shape[2],
                strides[0],
                dilations[0],
                effective_pads[0],
                effective_pads[1],
            );
            out = b
                .builder
                .reshape_with_options(
                    out,
                    i64_slice_to_mldim(&[input_shape[0], filter_shape[0], spatial_out])?,
                    OnnxBuilder::labeled_options(&output_name),
                )
                .map_err(map_op_error)?;
        }

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        let mut result = ConversionResult::default();
        if let Some(onnx_out) = node.output.first() {
            result
                .output_types
                .insert(onnx_out.clone(), DataType::Int32);
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn center_quantized_operand(
        &self,
        b: &mut OnnxBuilder<'_, '_, '_>,
        operand_name: &str,
        zero_point_name: Option<&str>,
        context: &ConversionContext,
        is_filter: bool,
        operand_shape: &[i64],
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
        let zero_point_float = b
            .builder
            .cast_with_options(
                zero_point,
                MLOperandDataType::Float32,
                OnnxBuilder::labeled_options(&format!("{label}_zero_point_float")),
            )
            .map_err(map_op_error)?;
        let zero_point_shape = if is_filter
            && lookup_shape(zero_point_name, context).is_some_and(|shape| shape.len() == 1)
        {
            // ONNX permits a scalar or one zero point per output channel. WebNN
            // broadcasting aligns trailing dimensions, so reshape [M] to
            // [M, 1, 1, ...] for OIHW filters.
            let mut shape = vec![1i64; operand_shape.len()];
            shape[0] = operand_shape[0];
            Some(shape)
        } else {
            None
        };
        let centered_zero_point = if let Some(shape) = zero_point_shape {
            b.builder
                .reshape_with_options(
                    zero_point_float,
                    i64_slice_to_mldim(&shape)?,
                    OnnxBuilder::labeled_options(&format!("{label}_zero_point_reshape")),
                )
                .map_err(map_op_error)?
        } else {
            zero_point_float
        };

        b.builder
            .sub_with_options(
                as_float,
                centered_zero_point,
                OnnxBuilder::labeled_options(&format!("{label}_centered")),
            )
            .map_err(map_op_error)
    }

    fn convert_conv(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
        transpose: bool,
    ) -> Result<ConversionResult, OnnxError> {
        let op_label = if transpose { "ConvTranspose" } else { "Conv" };
        let inputs = node.input.as_slice();
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 2 or 3 inputs (X, W[, B]), got {}",
                op_label,
                inputs.len()
            )));
        }

        let input_raw = inputs[0].as_str();
        let filter_raw = inputs[1].as_str();
        let bias_raw = inputs.get(2).map(|s| s.as_str());

        let output_name = output_label(node, node_name);
        let attrs = parse_conv_attrs(node);

        let filter_shape = lookup_shape(filter_raw, context);
        let input_shape = lookup_shape(input_raw, context);
        let spatial_rank = if let Some(ks) = attrs.kernel_shape.as_ref() {
            ks.len()
        } else if let Some(fs) = filter_shape.as_ref() {
            if fs.len() >= 2 {
                fs.len() - 2
            } else {
                return Err(OnnxError::InvalidShape(format!(
                    "{}: filter '{}' has rank {} (need >= 2)",
                    op_label,
                    filter_raw,
                    fs.len()
                )));
            }
        } else if let Some(is) = input_shape.as_ref() {
            if is.len() >= 2 {
                is.len() - 2
            } else {
                return Err(OnnxError::InvalidShape(format!(
                    "{}: cannot determine spatial rank from input '{}' of rank {}",
                    op_label,
                    input_raw,
                    is.len()
                )));
            }
        } else {
            return Err(OnnxError::InvalidShape(format!(
                "{}: cannot determine spatial rank - provide kernel_shape attribute or filter/input shape info",
                op_label,
            )));
        };

        match spatial_rank {
            2 => self.emit_conv_2d(
                node,
                &output_name,
                input_raw,
                filter_raw,
                bias_raw,
                &attrs,
                transpose,
                b,
            ),
            1 => self.emit_conv_1d_via_2d(
                node,
                node_name,
                &output_name,
                input_raw,
                filter_raw,
                bias_raw,
                &attrs,
                transpose,
                input_shape.as_deref(),
                filter_shape.as_deref(),
                b,
            ),
            _ => Err(OnnxError::unsupported_op(
                format!("{}{}D", op_label, spatial_rank),
                node_name.to_string(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_conv_2d(
        &self,
        node: &NodeProto,
        output_name: &str,
        input_raw: &str,
        filter_raw: &str,
        bias_raw: Option<&str>,
        attrs: &ConvAttrs,
        transpose: bool,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let input = b.resolve_operand(input_raw)?;
        let filter = b.resolve_operand(filter_raw)?;
        let conv_label = sanitize_identifier(&format!("{output_name}_conv2d"));

        let out = if transpose {
            let mut opts = build_conv_transpose2d_options(attrs, &conv_label)?;
            opts.bias = optional_operand_index(b, bias_raw)?;
            b.builder
                .conv_transpose2d_with_options(input, filter, opts)
                .map_err(map_op_error)?
        } else {
            let mut opts = build_conv2d_options(attrs, &conv_label)?;
            opts.bias = optional_operand_index(b, bias_raw)?;
            b.builder
                .conv2_with_options(input, filter, opts)
                .map_err(map_op_error)?
        };

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, output_name, out);
        } else {
            b.record_operand(&[output_name], out);
        }
        Ok(ConversionResult::default())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_conv_1d_via_2d(
        &self,
        node: &NodeProto,
        node_name: &str,
        output_name: &str,
        input_raw: &str,
        filter_raw: &str,
        bias_raw: Option<&str>,
        attrs: &ConvAttrs,
        transpose: bool,
        input_shape: Option<&[i64]>,
        filter_shape: Option<&[i64]>,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let input_shape = input_shape.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "1D Conv emulation requires known shape for input of node {}",
                node_name
            ))
        })?;
        let filter_shape = filter_shape.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "1D Conv emulation requires known shape for filter of node {}",
                node_name
            ))
        })?;
        if input_shape.len() != 3 || filter_shape.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "1D Conv emulation expects rank-3 input/filter, got input {:?} filter {:?}",
                input_shape, filter_shape
            )));
        }

        let mut attrs_2d = attrs.clone();
        attrs_2d.strides = Some(extend_with_one(attrs.strides.as_deref(), 1, 2));
        attrs_2d.dilations = Some(extend_with_one(attrs.dilations.as_deref(), 1, 2));
        attrs_2d.pads = Some(extend_pads_to_2d(attrs.pads.as_deref()));
        attrs_2d.kernel_shape = attrs
            .kernel_shape
            .as_ref()
            .map(|ks| extend_with_one(Some(ks.as_slice()), 1, 2));
        if transpose {
            attrs_2d.output_padding = Some(extend_with_one(attrs.output_padding.as_deref(), 0, 2));
            attrs_2d.output_shape = None;
        }

        let reshape_in_label = sanitize_identifier(&format!("{node_name}_x4d"));
        let reshape_w_label = sanitize_identifier(&format!("{node_name}_w4d"));
        let conv_label = sanitize_identifier(&format!("{node_name}_conv2d"));

        let in_4d = i64_slice_to_mldim(&[input_shape[0], input_shape[1], input_shape[2], 1])?;
        let w_4d = i64_slice_to_mldim(&[filter_shape[0], filter_shape[1], filter_shape[2], 1])?;

        let input = b.resolve_operand(input_raw)?;
        let filter = b.resolve_operand(filter_raw)?;

        let x4d = b
            .builder
            .reshape_with_options(
                input,
                in_4d,
                OnnxBuilder::labeled_options(&reshape_in_label),
            )
            .map_err(map_op_error)?;
        b.record_operand(&[&reshape_in_label], x4d);

        let w4d = b
            .builder
            .reshape_with_options(filter, w_4d, OnnxBuilder::labeled_options(&reshape_w_label))
            .map_err(map_op_error)?;
        b.record_operand(&[&reshape_w_label], w4d);

        let conv_out = if transpose {
            let mut opts = build_conv_transpose2d_options(&attrs_2d, &conv_label)?;
            opts.bias = optional_operand_index(b, bias_raw)?;
            b.builder
                .conv_transpose2d_with_options(x4d, w4d, opts)
                .map_err(map_op_error)?
        } else {
            let mut opts = build_conv2d_options(&attrs_2d, &conv_label)?;
            opts.bias = optional_operand_index(b, bias_raw)?;
            b.builder
                .conv2_with_options(x4d, w4d, opts)
                .map_err(map_op_error)?
        };
        b.record_operand(&[&conv_label], conv_out);

        let strides = attrs_2d.strides.clone().unwrap_or_else(|| vec![1, 1]);
        let dilations = attrs_2d.dilations.clone().unwrap_or_else(|| vec![1, 1]);
        let pads = attrs_2d.pads.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);
        let effective_pads = if attrs.auto_pad == "VALID" {
            vec![0, 0, 0, 0]
        } else if map_auto_pad(&attrs.auto_pad) == "explicit" {
            onnx_pads_to_webnn(&pads, 2)
        } else {
            vec![0, 0, 0, 0]
        };
        let kernel_h = filter_shape[2];
        let spatial_out = conv2d_spatial_out(
            input_shape[2],
            kernel_h,
            strides[0],
            dilations[0],
            effective_pads[0],
            effective_pads[1],
        );
        let out_3d = i64_slice_to_mldim(&[input_shape[0], filter_shape[0], spatial_out])?;
        let final_out = b
            .builder
            .reshape_with_options(conv_out, out_3d, OnnxBuilder::labeled_options(output_name))
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, output_name, final_out);
        } else {
            b.record_operand(&[output_name], final_out);
        }
        Ok(ConversionResult::default())
    }
}

fn extend_with_one(src: Option<&[i64]>, fill: i64, target_len: usize) -> Vec<i64> {
    let mut out = src.map(|v| v.to_vec()).unwrap_or_default();
    while out.len() < target_len {
        out.push(fill);
    }
    out
}

fn extend_pads_to_2d(pads: Option<&[i64]>) -> Vec<i64> {
    match pads {
        Some(p) if p.len() == 2 => vec![p[0], 0, p[1], 0],
        Some(p) if p.len() == 4 => p.to_vec(),
        _ => vec![0, 0, 0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, NodeProto};
    use std::collections::HashMap;

    fn make_node(
        op_type: &str,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
        attrs: Vec<AttributeProto>,
    ) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            attribute: attrs,
            ..Default::default()
        }
    }

    fn int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        }
    }

    fn ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            ..Default::default()
        }
    }

    fn string_attr(name: &str, value: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            ..Default::default()
        }
    }

    fn make_context<'a>(
        initializers: &'a HashMap<String, &'a crate::protos::onnx::TensorProto>,
        value_shapes: &'a HashMap<String, Vec<i64>>,
        const_values: &'a HashMap<String, Vec<i64>>,
        value_ids: &'a HashMap<String, String>,
        value_types: &'a HashMap<String, rustnn::DataType>,
    ) -> ConversionContext<'a> {
        ConversionContext {
            initializers,
            value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values,
            value_ids,
            value_types,
        }
    }

    #[test]
    fn supports_conv_ops() {
        let h = ConvHandler;
        assert!(h.supports("Conv"));
        assert!(h.supports("ConvTranspose"));
        assert!(h.supports("ConvInteger"));
        assert!(!h.supports("MatMul"));
        assert!(!h.supports("Pool"));
    }

    #[test]
    fn conv2d_basic_defaults() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![ints_attr("kernel_shape", vec![3, 3])],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 224, 224]);
        value_shapes.insert("w".to_string(), vec![64, 3, 3, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap();
    }

    #[test]
    fn conv2d_with_strides_pads_dilations_groups() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w", "b"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("strides", vec![2, 2]),
                ints_attr("pads", vec![1, 1, 1, 1]),
                ints_attr("dilations", vec![1, 1]),
                int_attr("group", 4),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 4, 112, 112]);
        value_shapes.insert("w".to_string(), vec![8, 1, 3, 3]);
        value_shapes.insert("b".to_string(), vec![8]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap();
    }

    #[test]
    fn conv2d_pads_layout_reordered() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("pads", vec![1, 2, 3, 4]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 32, 32]);
        value_shapes.insert("w".to_string(), vec![8, 3, 3, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap();
    }

    #[test]
    fn conv2d_auto_pad_same_upper() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                string_attr("auto_pad", "SAME_UPPER"),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 32, 32]);
        value_shapes.insert("w".to_string(), vec![8, 3, 3, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap();
    }

    #[test]
    fn conv2d_auto_pad_valid_zeroes_pads() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                string_attr("auto_pad", "VALID"),
                ints_attr("pads", vec![1, 1, 1, 1]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 32, 32]);
        value_shapes.insert("w".to_string(), vec![8, 3, 3, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap();
    }

    #[test]
    fn conv_transpose_basic() {
        let h = ConvHandler;
        let node = make_node(
            "ConvTranspose",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("strides", vec![2, 2]),
                ints_attr("output_padding", vec![1, 1]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 16, 32, 32]);
        value_shapes.insert("w".to_string(), vec![16, 8, 3, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap();
    }

    #[test]
    fn conv_transpose_output_shape_full_form() {
        let h = ConvHandler;
        let node = make_node(
            "ConvTranspose",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("output_shape", vec![1, 8, 64, 64]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 16, 32, 32]);
        value_shapes.insert("w".to_string(), vec![16, 8, 3, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap();
    }

    #[test]
    fn conv1d_emulated_via_2d() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3]),
                ints_attr("strides", vec![2]),
                ints_attr("pads", vec![1, 1]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 16, 64]);
        value_shapes.insert("w".to_string(), vec![8, 16, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap();
    }

    #[test]
    fn conv_3d_unsupported() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["x", "w"],
            vec!["y"],
            vec![ints_attr("kernel_shape", vec![3, 3, 3])],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3, 16, 16, 16]);
        value_shapes.insert("w".to_string(), vec![8, 3, 3, 3, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        let err = crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap_err();
        match err {
            OnnxError::UnsupportedOps(ops) => {
                assert!(
                    ops[0].op.contains("3D"),
                    "expected 3D in op label, got {}",
                    ops[0].op
                );
            }
            other => panic!("expected UnsupportedOp, got {:?}", other),
        }
    }

    #[test]
    fn conv_resolves_input_aliases() {
        let h = ConvHandler;
        let node = make_node(
            "Conv",
            vec!["onnx_x", "onnx_w"],
            vec!["y"],
            vec![ints_attr("kernel_shape", vec![3, 3])],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("onnx_x".to_string(), vec![1, 3, 32, 32]);
        value_shapes.insert("onnx_w".to_string(), vec![8, 3, 3, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let ctx = make_context(
            &initializers,
            &value_shapes,
            &const_values,
            &value_ids,
            &value_types,
        );

        crate::onnx::ops::convert_handler_with_context(&h, &node, &ctx).unwrap();
    }

    #[test]
    fn onnx_pads_to_webnn_reorders() {
        assert_eq!(onnx_pads_to_webnn(&[1, 2, 3, 4], 2), vec![1, 3, 2, 4]);
        assert_eq!(onnx_pads_to_webnn(&[5, 6], 1), vec![5, 6]);
    }
}
