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

// Pooling operators: MaxPool, AveragePool, LpPool, GlobalMaxPool, GlobalAveragePool
//
// Maps ONNX pooling ops to WebNN maxPool2d / averagePool2d (NCHW layout).

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{i64_slice_to_mldim, output_label, record_node_output};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::operator_options::MLPool2dOptions;

pub struct PoolHandler;

impl OpHandler for PoolHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "MaxPool" | "AveragePool" | "LpPool" | "GlobalMaxPool" | "GlobalAveragePool"
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
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "MaxPool" => self.convert_pool(node, &node_name, context, b, PoolKind::Max),
            "AveragePool" => self.convert_pool(node, &node_name, context, b, PoolKind::Average),
            "LpPool" => {
                let p = lp_pool_p(node);
                if p != 2 {
                    return Err(OnnxError::unsupported_op(
                        format!("LpPool(p={p})"),
                        node_name,
                    ));
                }
                self.convert_pool(node, &node_name, context, b, PoolKind::L2)
            }
            "GlobalMaxPool" => {
                self.convert_global_pool(node, &node_name, context, b, PoolKind::Max)
            }
            "GlobalAveragePool" => {
                self.convert_global_pool(node, &node_name, context, b, PoolKind::Average)
            }
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PoolKind {
    Max,
    Average,
    L2,
}

#[derive(Debug, Clone)]
struct PoolAttrs {
    kernel_shape: Option<Vec<i64>>,
    strides: Option<Vec<i64>>,
    dilations: Option<Vec<i64>>,
    pads: Option<Vec<i64>>,
    auto_pad: String,
    ceil_mode: bool,
    count_include_pad: bool,
}

fn parse_pool_attrs(node: &NodeProto) -> PoolAttrs {
    let mut attrs = PoolAttrs {
        kernel_shape: None,
        strides: None,
        dilations: None,
        pads: None,
        auto_pad: "NOTSET".to_string(),
        ceil_mode: false,
        count_include_pad: false,
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
            "kernel_shape" if !attr.ints.is_empty() => {
                attrs.kernel_shape = Some(attr.ints.clone());
            }
            "strides" if !attr.ints.is_empty() => {
                attrs.strides = Some(attr.ints.clone());
            }
            "dilations" if !attr.ints.is_empty() => {
                attrs.dilations = Some(attr.ints.clone());
            }
            "pads" if !attr.ints.is_empty() => {
                attrs.pads = Some(attr.ints.clone());
            }
            "ceil_mode" => {
                attrs.ceil_mode = attr.i != 0;
            }
            "count_include_pad" => {
                attrs.count_include_pad = attr.i != 0;
            }
            _ => {}
        }
    }

    attrs
}

fn lookup_shape(name: &str, context: &ConversionContext) -> Option<Vec<i64>> {
    if let Some(s) = context.value_shapes.get(name) {
        return Some(s.clone());
    }
    let sanitized = sanitize_identifier(name);
    if let Some(s) = context.value_shapes.get(&sanitized) {
        return Some(s.clone());
    }
    None
}

fn map_auto_pad(auto_pad: &str) -> &'static str {
    match auto_pad {
        "SAME_UPPER" => "same-upper",
        "SAME_LOWER" => "same-lower",
        _ => "explicit",
    }
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
                OnnxError::InvalidShape(format!("negative or oversized pool option value: {v}"))
            })
        })
        .collect()
}

fn pool2d_spatial_out(
    in_spatial: i64,
    kernel: i64,
    stride: i64,
    dilation: i64,
    pad_begin: i64,
    pad_end: i64,
    ceil_mode: bool,
) -> i64 {
    let effective_kernel = (kernel - 1) * dilation + 1;
    let numerator = in_spatial + pad_begin + pad_end - effective_kernel;
    if ceil_mode {
        (numerator + stride - 1) / stride + 1
    } else {
        numerator / stride + 1
    }
}

/// WebNN pool2d options plus, for `AveragePool(count_include_pad=1)`, the
/// explicit zero padding to apply beforehand so the pads count in the average.
#[derive(Debug)]
struct Pool2dPlan {
    opts: MLPool2dOptions,
    /// [top, bottom, left, right] zero padding to apply before pooling.
    pre_pad: Option<[u32; 4]>,
}

fn build_pool_2d_options(
    attrs: &PoolAttrs,
    kernel: &[i64],
    kind: PoolKind,
    label: &str,
) -> Result<Pool2dPlan, OnnxError> {
    let strides = attrs.strides.clone().unwrap_or_else(|| vec![1, 1]);
    let dilations = attrs.dilations.clone().unwrap_or_else(|| vec![1, 1]);
    let pads = attrs.pads.clone().unwrap_or_else(|| vec![0, 0, 0, 0]);
    if strides.len() != 2 || dilations.len() != 2 || kernel.len() != 2 {
        return Err(OnnxError::InvalidShape(format!(
            "pool2d: expected length-2 kernel/strides/dilations, got kernel={:?} strides={:?} dilations={:?}",
            kernel, strides, dilations
        )));
    }

    let mut opts = MLPool2dOptions {
        label: label.to_string(),
        window_dimensions: Some(i64_vec_to_u32(kernel)?),
        strides: i64_vec_to_u32(&strides)?,
        dilations: if matches!(kind, PoolKind::Max) || dilations.iter().any(|&d| d != 1) {
            i64_vec_to_u32(&dilations)?
        } else {
            Vec::new()
        },
        output_shape_rounding: if attrs.ceil_mode {
            "ceil".to_string()
        } else {
            String::new()
        },
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
                "pool2d: padding must yield 4 values for 2D, got {:?}",
                effective_pads
            )));
        }
        opts.padding = i64_vec_to_u32(&effective_pads)?;
    }

    // WebNN averagePool2d never counts padding; ONNX count_include_pad=1 does.
    // Pad explicitly with zeros and pool without padding instead.
    let mut pre_pad = None;
    if matches!(kind, PoolKind::Average)
        && attrs.count_include_pad
        && opts.padding.iter().any(|&p| p != 0)
    {
        pre_pad = Some([
            opts.padding[0],
            opts.padding[1],
            opts.padding[2],
            opts.padding[3],
        ]);
        opts.padding = vec![0, 0, 0, 0];
    } else if matches!(kind, PoolKind::Average)
        && attrs.count_include_pad
        && map_auto_pad(&attrs.auto_pad) != "explicit"
    {
        return Err(OnnxError::unsupported_op(
            format!(
                "AveragePool(count_include_pad=1, auto_pad={})",
                attrs.auto_pad
            ),
            String::new(),
        ));
    }

    Ok(Pool2dPlan { opts, pre_pad })
}

/// Zero-pad the spatial dims of an NCHW operand ([top, bottom, left, right]).
fn zero_pad_spatial(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: rustnn::mlcontext::MLOperand,
    pads: [u32; 4],
    label: &str,
) -> Result<rustnn::mlcontext::MLOperand, OnnxError> {
    let out = b
        .builder
        .pad_with_options(
            input,
            vec![0, 0, pads[0], pads[2]],
            vec![0, 0, pads[1], pads[3]],
            rustnn::operator_options::MLPadOptions {
                label: label.to_string(),
                mode: "constant".to_string(),
                value: None,
            },
        )
        .map_err(map_op_error)?;
    b.record_operand(&[label], out);
    Ok(out)
}

impl PoolHandler {
    fn emit_pool(
        &self,
        b: &mut OnnxBuilder<'_, '_, '_>,
        node: &NodeProto,
        output_name: &str,
        input: rustnn::mlcontext::MLOperand,
        opts: MLPool2dOptions,
        kind: PoolKind,
    ) -> Result<ConversionResult, OnnxError> {
        let out = match kind {
            PoolKind::Max => b.builder.max_pool2d_with_options(input, opts),
            PoolKind::Average => b.builder.average_pool2d_with_options(input, opts),
            PoolKind::L2 => b.builder.l2_pool2d_with_options(input, opts),
        }
        .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, output_name, out);
        } else {
            b.record_operand(&[output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_pool(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
        kind: PoolKind,
    ) -> Result<ConversionResult, OnnxError> {
        let op_label = match kind {
            PoolKind::Max => "MaxPool",
            PoolKind::Average => "AveragePool",
            PoolKind::L2 => "LpPool",
        };

        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                op_label,
                inputs.len()
            )));
        }
        if matches!(kind, PoolKind::Max) && node.output.len() > 1 {
            return Err(OnnxError::unsupported_op(
                "MaxPool(with indices output)".to_string(),
                node_name.to_string(),
            ));
        }

        let input_raw = inputs[0].as_str();
        let input_shape = lookup_shape(input_raw, context);
        let output_name = output_label(node, node_name);
        let attrs = parse_pool_attrs(node);

        let kernel = attrs
            .kernel_shape
            .clone()
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "kernel_shape".to_string(),
                op: op_label.to_string(),
            })?;
        let spatial_rank = kernel.len();

        match spatial_rank {
            2 => {
                let mut input = b.resolve_operand(input_raw)?;
                let plan = build_pool_2d_options(&attrs, &kernel, kind, &output_name)?;
                if let Some(pads) = plan.pre_pad {
                    input = zero_pad_spatial(b, input, pads, &format!("{output_name}_padded"))?;
                }
                self.emit_pool(b, node, &output_name, input, plan.opts, kind)
            }
            1 => self.emit_pool_1d_via_2d(
                node,
                node_name,
                &output_name,
                input_raw,
                &attrs,
                &kernel,
                kind,
                input_shape.as_deref(),
                b,
            ),
            _ => Err(OnnxError::unsupported_op(
                format!("{}{}D", op_label, spatial_rank),
                node_name.to_string(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_pool_1d_via_2d(
        &self,
        node: &NodeProto,
        node_name: &str,
        output_name: &str,
        input_raw: &str,
        attrs: &PoolAttrs,
        kernel: &[i64],
        kind: PoolKind,
        input_shape: Option<&[i64]>,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let input_shape = input_shape.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "1D pool emulation requires known shape for input of node {}",
                node_name
            ))
        })?;
        if input_shape.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "1D pool emulation expects rank-3 input, got {:?}",
                input_shape
            )));
        }

        let mut attrs_2d = attrs.clone();
        attrs_2d.strides = Some(extend_with(attrs.strides.as_deref(), 1, 2));
        attrs_2d.dilations = Some(extend_with(attrs.dilations.as_deref(), 1, 2));
        attrs_2d.pads = Some(extend_pads_to_2d(attrs.pads.as_deref()));
        let mut kernel_2d = kernel.to_vec();
        if kernel_2d.len() == 1 {
            kernel_2d.push(1);
        }
        attrs_2d.kernel_shape = Some(kernel_2d.clone());

        let reshape_in_label = sanitize_identifier(&format!("{node_name}_x4d"));
        let pool_label = sanitize_identifier(&format!("{node_name}_pool2d"));

        let in_4d = i64_slice_to_mldim(&[input_shape[0], input_shape[1], input_shape[2], 1])?;
        let input = b.resolve_operand(input_raw)?;
        let mut x4d = b
            .builder
            .reshape_with_options(
                input,
                in_4d,
                OnnxBuilder::labeled_options(&reshape_in_label),
            )
            .map_err(map_op_error)?;
        b.record_operand(&[&reshape_in_label], x4d);

        let plan = build_pool_2d_options(&attrs_2d, &kernel_2d, kind, &pool_label)?;
        if let Some(pads) = plan.pre_pad {
            x4d = zero_pad_spatial(b, x4d, pads, &format!("{pool_label}_padded"))?;
        }
        let pool_opts = plan.opts;
        let pooled = match kind {
            PoolKind::Max => b.builder.max_pool2d_with_options(x4d, pool_opts),
            PoolKind::Average => b.builder.average_pool2d_with_options(x4d, pool_opts),
            PoolKind::L2 => b.builder.l2_pool2d_with_options(x4d, pool_opts),
        }
        .map_err(map_op_error)?;
        b.record_operand(&[&pool_label], pooled);

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
        let spatial_out = pool2d_spatial_out(
            input_shape[2],
            kernel[0],
            strides[0],
            dilations[0],
            effective_pads[0],
            effective_pads[1],
            attrs.ceil_mode,
        );
        let out_3d = i64_slice_to_mldim(&[input_shape[0], input_shape[1], spatial_out])?;
        let final_out = b
            .builder
            .reshape_with_options(pooled, out_3d, OnnxBuilder::labeled_options(output_name))
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, output_name, final_out);
        } else {
            b.record_operand(&[output_name], final_out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_global_pool(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
        kind: PoolKind,
    ) -> Result<ConversionResult, OnnxError> {
        if matches!(kind, PoolKind::L2) {
            return Err(OnnxError::unsupported_op("GlobalLpPool", node_name));
        }
        let op_label = match kind {
            PoolKind::Max => "GlobalMaxPool",
            PoolKind::Average => "GlobalAveragePool",
            PoolKind::L2 => "GlobalLpPool",
        };
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                op_label,
                inputs.len()
            )));
        }

        let input_raw = inputs[0].as_str();
        let input_shape = lookup_shape(input_raw, context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "{}: input '{}' shape is unknown - required to determine spatial window size",
                op_label, input_raw
            ))
        })?;
        if input_shape.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "{}: input must be at least rank-3 (N, C, spatial...), got {:?}",
                op_label, input_shape
            )));
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(input_raw)?;
        let spatial = &input_shape[2..];

        match spatial.len() {
            2 => {
                let opts = MLPool2dOptions {
                    label: output_name.clone(),
                    ..Default::default()
                };
                let out = match kind {
                    PoolKind::Max => b.builder.global_max_pool_with_options(input, opts),
                    PoolKind::Average => b.builder.global_average_pool_with_options(input, opts),
                    PoolKind::L2 => unreachable!("GlobalLpPool handled above"),
                }
                .map_err(map_op_error)?;
                if let Some(onnx_out) = node.output.first() {
                    record_node_output(b, onnx_out, &output_name, out);
                } else {
                    b.record_operand(&[&output_name], out);
                }
                Ok(ConversionResult::default())
            }
            1 => {
                let reshape_in_label = sanitize_identifier(&format!("{node_name}_x4d"));
                let pool_label = sanitize_identifier(&format!("{node_name}_pool2d"));
                let in_4d = i64_slice_to_mldim(&[input_shape[0], input_shape[1], spatial[0], 1])?;
                let x4d = b
                    .builder
                    .reshape_with_options(
                        input,
                        in_4d,
                        OnnxBuilder::labeled_options(&reshape_in_label),
                    )
                    .map_err(map_op_error)?;
                b.record_operand(&[&reshape_in_label], x4d);

                let pool_opts = MLPool2dOptions {
                    label: pool_label.clone(),
                    window_dimensions: Some(vec![spatial[0] as u32, 1]),
                    ..Default::default()
                };
                let pooled = match kind {
                    PoolKind::Max => b.builder.max_pool2d_with_options(x4d, pool_opts),
                    PoolKind::Average => b.builder.average_pool2d_with_options(x4d, pool_opts),
                    PoolKind::L2 => unreachable!("GlobalLpPool handled above"),
                }
                .map_err(map_op_error)?;
                b.record_operand(&[&pool_label], pooled);

                let out_3d = i64_slice_to_mldim(&[input_shape[0], input_shape[1], 1])?;
                let final_out = b
                    .builder
                    .reshape_with_options(
                        pooled,
                        out_3d,
                        OnnxBuilder::labeled_options(&output_name),
                    )
                    .map_err(map_op_error)?;
                if let Some(onnx_out) = node.output.first() {
                    record_node_output(b, onnx_out, &output_name, final_out);
                } else {
                    b.record_operand(&[&output_name], final_out);
                }
                Ok(ConversionResult::default())
            }
            _ => Err(OnnxError::unsupported_op(
                format!("{}{}D", op_label, spatial.len()),
                node_name.to_string(),
            )),
        }
    }
}

fn lp_pool_p(node: &NodeProto) -> i64 {
    for attr in node.attribute.as_slice() {
        if attr.name.as_str() == "p" {
            return attr.i;
        }
    }
    2
}

fn extend_with(src: Option<&[i64]>, fill: i64, target_len: usize) -> Vec<i64> {
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
    fn supports_pool_ops() {
        let h = PoolHandler;
        assert!(h.supports("MaxPool"));
        assert!(h.supports("AveragePool"));
        assert!(h.supports("GlobalMaxPool"));
        assert!(h.supports("GlobalAveragePool"));
        assert!(!h.supports("Conv"));
    }

    #[test]
    fn maxpool2d_basic() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("strides", vec![2, 2]),
                ints_attr("pads", vec![1, 1, 1, 1]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 64, 112, 112]);
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
    fn maxpool2d_pads_layout_reordered() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("pads", vec![1, 2, 3, 4]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 64, 32, 32]);
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
    fn maxpool2d_with_ceil_mode() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![2, 2]),
                int_attr("ceil_mode", 1),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 7, 7]);
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
    fn maxpool2d_auto_pad_same_upper() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                string_attr("auto_pad", "SAME_UPPER"),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 32, 32]);
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
    fn averagepool2d_basic() {
        let h = PoolHandler;
        let node = make_node(
            "AveragePool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![2, 2]),
                ints_attr("strides", vec![2, 2]),
            ],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 14, 14]);
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
    fn averagepool_count_include_pad_without_padding_is_plain_pool() {
        let attrs = parse_pool_attrs(&make_node(
            "AveragePool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![16, 16]),
                ints_attr("strides", vec![16, 16]),
                int_attr("count_include_pad", 1),
            ],
        ));
        let plan = build_pool_2d_options(&attrs, &[16, 16], PoolKind::Average, "y").unwrap();
        assert!(plan.pre_pad.is_none());
        assert_eq!(plan.opts.window_dimensions, Some(vec![16, 16]));
    }

    #[test]
    fn averagepool_count_include_pad_moves_padding_to_explicit_pad() {
        let attrs = parse_pool_attrs(&make_node(
            "AveragePool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("pads", vec![1, 2, 3, 4]),
                int_attr("count_include_pad", 1),
            ],
        ));
        let plan = build_pool_2d_options(&attrs, &[3, 3], PoolKind::Average, "y").unwrap();
        // ONNX [top, left, bottom, right] -> WebNN [top, bottom, left, right]
        assert_eq!(plan.pre_pad, Some([1, 3, 2, 4]));
        assert_eq!(plan.opts.padding, vec![0, 0, 0, 0]);

        // Without count_include_pad the padding stays on the pool op.
        let attrs = parse_pool_attrs(&make_node(
            "AveragePool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                ints_attr("pads", vec![1, 2, 3, 4]),
            ],
        ));
        let plan = build_pool_2d_options(&attrs, &[3, 3], PoolKind::Average, "y").unwrap();
        assert!(plan.pre_pad.is_none());
        assert_eq!(plan.opts.padding, vec![1, 3, 2, 4]);
    }

    #[test]
    fn averagepool_count_include_pad_with_auto_pad_rejected() {
        let attrs = parse_pool_attrs(&make_node(
            "AveragePool",
            vec!["x"],
            vec!["y"],
            vec![
                ints_attr("kernel_shape", vec![3, 3]),
                int_attr("count_include_pad", 1),
                string_attr("auto_pad", "SAME_UPPER"),
            ],
        ));
        let err = build_pool_2d_options(&attrs, &[3, 3], PoolKind::Average, "y").unwrap_err();
        match err {
            OnnxError::UnsupportedOps(ops) => assert!(ops[0].op.contains("count_include_pad")),
            other => panic!("expected UnsupportedOp, got {:?}", other),
        }
    }

    #[test]
    fn global_average_pool_2d() {
        let h = PoolHandler;
        let node = make_node("GlobalAveragePool", vec!["x"], vec!["y"], vec![]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 2048, 7, 7]);
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
    fn global_max_pool_2d() {
        let h = PoolHandler;
        let node = make_node("GlobalMaxPool", vec!["x"], vec!["y"], vec![]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 16, 14, 14]);
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
    fn maxpool_missing_kernel_shape_errors() {
        let h = PoolHandler;
        let node = make_node("MaxPool", vec!["x"], vec!["y"], vec![]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 14, 14]);
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
            OnnxError::MissingAttribute { attr, .. } => {
                assert_eq!(attr, "kernel_shape");
            }
            other => panic!("expected MissingAttribute, got {:?}", other),
        }
    }

    #[test]
    fn maxpool_rejects_indices_output() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
            vec!["y", "indices"],
            vec![ints_attr("kernel_shape", vec![2, 2])],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 8, 14, 14]);
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
                assert!(ops[0].op.contains("indices"));
            }
            other => panic!("expected UnsupportedOp, got {:?}", other),
        }
    }

    #[test]
    fn maxpool1d_emulated_via_2d() {
        let h = PoolHandler;
        let node = make_node(
            "MaxPool",
            vec!["x"],
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
}
