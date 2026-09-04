/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// ONNX Resize -> WebNN resample2d (4-D NCHW, spatial axes [2, 3] for now).

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{
    i64_slice_to_mldim, output_label, record_node_output, reshape_with_shape,
};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::conv::lookup_shape;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::{NodeProto, TensorProto, TensorProto_DataType};
use rustnn::operator_options::MLResample2dOptions;

pub struct ResizeHandler;

impl OpHandler for ResizeHandler {
    fn supports(&self, op_type: &str) -> bool {
        op_type == "Resize"
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

        let inputs = node.input.as_slice();
        if inputs.is_empty() || inputs[0].is_empty() {
            return Err(OnnxError::InvalidShape(
                "Resize expects a non-empty data input".to_string(),
            ));
        }

        let mode = parse_resize_mode(node)?;
        // 3-D input (N, C, W): resample along W through a 4-D (N, C, 1, W) view.
        if let Some(shape) = lookup_shape(&inputs[0], context).filter(|s| s.len() == 3) {
            return convert_1d_resize(node, &node_name, context, b, &mode, &shape);
        }
        let axes: [usize; 2] = [2, 3];
        let (spatial_scales, spatial_sizes) =
            resolve_spatial_resample_params(node, context, &axes)?;

        let input = b.resolve_operand(&inputs[0])?;
        let output_name = output_label(node, &node_name);
        let opts = MLResample2dOptions {
            label: output_name.clone(),
            mode,
            scales: spatial_scales,
            sizes: spatial_sizes,
            axes: axes.iter().map(|&a| a as u32).collect(),
        };
        let out = b
            .builder
            .resample2d_with_options(input, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }
}

/// Resize of a rank-3 tensor along its last axis (e.g. temporal position
/// embeddings): reshape to (N, C, 1, W), resample2d, reshape back.
fn convert_1d_resize(
    node: &NodeProto,
    node_name: &str,
    context: &ConversionContext,
    b: &mut OnnxBuilder<'_, '_, '_>,
    mode: &str,
    shape: &[i64],
) -> Result<ConversionResult, OnnxError> {
    let inputs = node.input.as_slice();
    let optional = |idx: usize| {
        inputs
            .get(idx)
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str())
            .filter(|name| !is_empty_optional_tensor(name, context))
    };
    let (scales, sizes, out_w) = if let Some(name) = optional(3) {
        let sizes = read_int64_sizes(name, context).ok_or_else(|| {
            OnnxError::InvalidShape(format!("Resize sizes value '{name}' not found"))
        })?;
        if sizes.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "Resize sizes length {} does not match 3-D input",
                sizes.len()
            )));
        }
        let w = u32::try_from(sizes[2].max(1)).unwrap_or(1);
        (vec![1.0, 1.0], Some(vec![1, w]), i64::from(w))
    } else if let Some(name) = optional(2) {
        let scales = read_float32_initializer(name, context)?
            .filter(|s| s.len() == 3)
            .ok_or_else(|| {
                OnnxError::InvalidShape(format!(
                    "Resize scales value '{name}' must hold 3 values for a 3-D input"
                ))
            })?;
        let out_w = ((shape[2] as f32) * scales[2]).floor().max(1.0) as i64;
        (vec![1.0, scales[2]], None, out_w)
    } else {
        return Err(OnnxError::InvalidShape(
            "Resize requires sizes or scales".to_string(),
        ));
    };

    let output_name = output_label(node, node_name);
    let input = b.resolve_operand(&inputs[0])?;
    let x4 = reshape_with_shape(
        b,
        input,
        &format!("{output_name}_4d"),
        i64_slice_to_mldim(&[shape[0], shape[1], 1, shape[2]])?,
    )?;
    let resampled = b
        .builder
        .resample2d_with_options(
            x4,
            MLResample2dOptions {
                label: format!("{output_name}_resample"),
                mode: mode.to_string(),
                scales,
                sizes,
                axes: vec![2, 3],
            },
        )
        .map_err(map_op_error)?;
    let out = reshape_with_shape(
        b,
        resampled,
        &output_name,
        i64_slice_to_mldim(&[shape[0], shape[1], out_w])?,
    )?;
    if let Some(onnx_out) = node.output.first() {
        record_node_output(b, onnx_out, &output_name, out);
    } else {
        b.record_operand(&[&output_name], out);
    }
    Ok(ConversionResult::default())
}

fn parse_resize_mode(node: &NodeProto) -> Result<String, OnnxError> {
    let mut mode = "nearest".to_string();
    for attr in node.attribute.as_slice() {
        if attr.name.as_str() == "mode" {
            if let Ok(s) = String::from_utf8(attr.s.clone()) {
                if !s.is_empty() {
                    mode = s;
                }
            }
        }
    }

    match mode.as_str() {
        "nearest" => Ok("nearest-neighbor".to_string()),
        "linear" => Ok("linear".to_string()),
        "cubic" => Err(OnnxError::unsupported_op("Resize", node.name.as_str())),
        other => Err(OnnxError::InvalidShape(format!(
            "Resize mode '{other}' is not supported"
        ))),
    }
}

fn resolve_spatial_resample_params(
    node: &NodeProto,
    context: &ConversionContext,
    axes: &[usize; 2],
) -> Result<(Vec<f32>, Option<Vec<u32>>), OnnxError> {
    let inputs = node.input.as_slice();
    let input_shape = inputs
        .first()
        .and_then(|name| context.resolve_shape(name))
        .map(|shape| shape.as_slice());

    // Empty-string OR empty-tensor placeholders mean "optional input absent".
    let scales_name = inputs
        .get(2)
        .filter(|s| !s.is_empty())
        .map(|s| s.as_str())
        .filter(|name| !is_empty_optional_tensor(name, context));
    let sizes_name = inputs
        .get(3)
        .filter(|s| !s.is_empty())
        .map(|s| s.as_str())
        .filter(|name| !is_empty_optional_tensor(name, context));

    if let Some(name) = sizes_name {
        let sizes = read_int64_sizes(name, context).ok_or_else(|| {
            OnnxError::InvalidShape(format!("Resize sizes value '{name}' not found"))
        })?;
        if sizes.len() == 4 {
            // Preserve explicit ONNX output sizes. Deriving scales from inferred
            // input shapes is unnecessary and can be wrong when an earlier
            // best-effort shape pass was later refined.
            let spatial_sizes: Vec<u32> = axes
                .iter()
                .map(|&axis| u32::try_from(sizes[axis].max(1)).unwrap_or(1))
                .collect();
            return Ok((vec![1.0, 1.0], Some(spatial_sizes)));
        }
        return Err(OnnxError::InvalidShape(format!(
            "Resize sizes length {} is not supported for 4-D input",
            sizes.len()
        )));
    }

    let input_shape = input_shape
        .ok_or_else(|| OnnxError::InvalidShape("Resize input shape is unknown".to_string()))?;
    if input_shape.len() != 4 {
        return Err(OnnxError::unsupported_op("Resize", node.name.as_str()));
    }

    if let Some(name) = scales_name {
        if let Some(sizes) = read_int64_sizes(name, context) {
            if sizes.len() == input_shape.len() {
                let scales = spatial_scales_from_sizes(input_shape, axes, &sizes)?;
                return Ok((scales, None));
            }
        }
        if let Some(scales) = read_float32_initializer(name, context)? {
            if !scales.is_empty() {
                let out = spatial_scales_from_scales(input_shape, axes, &scales)?;
                return Ok((out, None));
            }
        }
    }

    Err(OnnxError::InvalidShape(
        "Resize requires scales or sizes input".to_string(),
    ))
}

/// True when `name` refers to a zero-element constant/initializer (optional absent).
fn is_empty_optional_tensor(name: &str, context: &ConversionContext) -> bool {
    if let Some(values) = context.const_values.get(name).or_else(|| {
        context
            .const_values
            .get(&crate::onnx::convert::sanitize_identifier(name))
    }) {
        return values.is_empty();
    }
    if let Some(tensor) = context.initializers.get(name).or_else(|| {
        context
            .initializers
            .get(&crate::onnx::convert::sanitize_identifier(name))
    }) {
        return crate::onnx::builder::tensor_element_count(tensor) == 0;
    }
    false
}

fn read_int64_sizes(name: &str, context: &ConversionContext) -> Option<Vec<i64>> {
    if let Some(values) = context.const_values.get(name).or_else(|| {
        context
            .const_values
            .get(&crate::onnx::convert::sanitize_identifier(name))
    }) {
        if !values.is_empty() {
            return Some(values.clone());
        }
    }
    read_int64_initializer(name, context)
}

fn spatial_scales_from_sizes(
    input_shape: &[i64],
    axes: &[usize; 2],
    sizes: &[i64],
) -> Result<Vec<f32>, OnnxError> {
    if sizes.len() != input_shape.len() {
        return Err(OnnxError::InvalidShape(format!(
            "Resize sizes length {} does not match input rank {}",
            sizes.len(),
            input_shape.len()
        )));
    }

    let mut scales = Vec::with_capacity(2);
    for &axis in axes {
        let in_dim = input_shape[axis].max(1) as f32;
        let out_dim = sizes[axis].max(1) as f32;
        scales.push(out_dim / in_dim);
    }
    Ok(scales)
}

fn spatial_scales_from_scales(
    input_shape: &[i64],
    axes: &[usize; 2],
    scales: &[f32],
) -> Result<Vec<f32>, OnnxError> {
    match scales.len() {
        2 => Ok(scales.to_vec()),
        len if len == input_shape.len() => {
            let mut out = Vec::with_capacity(2);
            for &axis in axes {
                out.push(scales[axis]);
            }
            Ok(out)
        }
        other => Err(OnnxError::InvalidShape(format!(
            "Resize scales length {other} is not supported for 4-D input"
        ))),
    }
}

fn read_int64_initializer(name: &str, context: &ConversionContext) -> Option<Vec<i64>> {
    let tensor = context.initializers.get(name).or_else(|| {
        context
            .initializers
            .get(&crate::onnx::convert::sanitize_identifier(name))
    })?;
    read_int64_tensor_proto(tensor)
}

fn read_float32_initializer(
    name: &str,
    context: &ConversionContext,
) -> Result<Option<Vec<f32>>, OnnxError> {
    let Some(tensor) = context.initializers.get(name).or_else(|| {
        context
            .initializers
            .get(&crate::onnx::convert::sanitize_identifier(name))
    }) else {
        return Ok(None);
    };

    if tensor.data_type != TensorProto_DataType::Float as i32 {
        return Ok(None);
    }

    if !tensor.float_data.is_empty() {
        return Ok(Some(tensor.float_data.clone()));
    }

    if !tensor.raw_data.is_empty() {
        let values = tensor
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        return Ok(Some(values));
    }

    Ok(None)
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
    use crate::protos::onnx::{AttributeProto, NodeProto, TensorProto};
    use std::collections::HashMap;

    fn create_resize_node(inputs: Vec<&str>, mode: &str) -> NodeProto {
        let mut node = NodeProto {
            op_type: "Resize".to_string(),
            name: "test_resize".to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec!["Y".to_string()],
            ..Default::default()
        };
        node.attribute.push(AttributeProto {
            name: "mode".to_string(),
            s: mode.as_bytes().to_vec(),
            ..Default::default()
        });
        node
    }

    #[test]
    fn test_resize_handler_supports() {
        let handler = ResizeHandler;
        assert!(handler.supports("Resize"));
        assert!(!handler.supports("Upsample"));
    }

    #[test]
    fn test_convert_resize_sizes() {
        let handler = ResizeHandler;
        let node = create_resize_node(vec!["X", "", "", "sizes"], "nearest");

        let sizes_tensor = TensorProto {
            name: "sizes".to_string(),
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![4],
            int64_data: vec![1, 1, 6, 6],
            ..Default::default()
        };
        let mut initializers: HashMap<String, &TensorProto> = HashMap::new();
        initializers.insert("sizes".to_string(), &sizes_tensor);

        let mut value_shapes = HashMap::new();
        value_shapes.insert("X".to_string(), vec![1, 1, 4, 4]);

        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &HashMap::new(),
            value_ids: &HashMap::new(),
            value_types: &HashMap::new(),
        };

        let (scales, sizes) = resolve_spatial_resample_params(&node, &context, &[2, 3]).unwrap();
        assert_eq!(scales, vec![1.0, 1.0]);
        assert_eq!(sizes, Some(vec![6, 6]));
        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
    }
}
