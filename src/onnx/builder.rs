/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! ONNX -> [`MLGraphBuilder`] bridge (operand map, naming, rustnn error mapping).

use crate::onnx::convert::{map_onnx_data_type, sanitize_identifier, OnnxError};
use crate::protos::onnx::{TensorProto, TensorProto_DataType};
use rustnn::error::{Error as RustnnError, GraphBuilderError};
use rustnn::graph::Dimension;
use rustnn::mlcontext::MLOperandDescriptor;
use rustnn::mlcontext::{MLGraph, MLGraphBuilder, MLOperand};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::MLOperatorOptions;
use rustnn::DataType;
use std::collections::{HashMap, HashSet};

pub struct OnnxBuilder<'a, 'ctx, 'bld> {
    pub builder: &'a mut MLGraphBuilder<'ctx, 'bld>,
    operands: HashMap<String, MLOperand>,
    /// Operand ids registered via `input()` - cannot be passed directly to `build()`.
    input_operands: HashSet<u32>,
    /// Operand ids registered via `constant()` - cannot be passed directly to `build()`.
    constant_operands: HashSet<u32>,
    /// Sanitized + raw ONNX names registered as graph inputs.
    input_names: HashSet<String>,
    /// Zero-element optional-input placeholders (e.g. empty `Resize` roi/scales).
    /// These are not materialized as WebNN constants because 0-sized dims are invalid.
    empty_optional_values: HashSet<String>,
    /// ONNX sequence values (SplitToSequence outputs) -> element count. The
    /// elements themselves are registered as `{name}__seq{i}` operands.
    sequences: HashMap<String, usize>,
}

/// Operand index inside the builder graph (`MLOperand::id` is `pub(crate)` in rustnn).
pub(crate) fn operand_index(op: MLOperand) -> u32 {
    #[repr(C)]
    struct Layout {
        id: usize,
    }
    // Safety: `MLOperand` is `{ id: usize }` only (rustnn `mlcontext.rs`).
    let layout: Layout = unsafe { std::mem::transmute(op) };
    layout.id as u32
}

impl<'a, 'ctx, 'bld> OnnxBuilder<'a, 'ctx, 'bld> {
    pub fn new(builder: &'a mut MLGraphBuilder<'ctx, 'bld>) -> Self {
        Self {
            builder,
            operands: HashMap::new(),
            input_operands: HashSet::new(),
            constant_operands: HashSet::new(),
            input_names: HashSet::new(),
            empty_optional_values: HashSet::new(),
            sequences: HashMap::new(),
        }
    }

    /// Name under which sequence element `i` of sequence `name` is registered.
    pub fn sequence_element_key(name: &str, index: usize) -> String {
        format!("{}__seq{index}", sanitize_identifier(name))
    }

    pub fn record_sequence(&mut self, name: &str, count: usize) {
        self.sequences.insert(name.to_string(), count);
        self.sequences.insert(sanitize_identifier(name), count);
    }

    pub fn sequence_element_count(&self, name: &str) -> Option<usize> {
        self.sequences
            .get(name)
            .or_else(|| self.sequences.get(&sanitize_identifier(name)))
            .copied()
    }

    pub fn webnn_id(onnx_name: &str) -> String {
        sanitize_identifier(onnx_name)
    }

    fn insert_name_aliases(set: &mut HashSet<String>, name: &str) {
        if name.is_empty() {
            return;
        }
        set.insert(name.to_string());
        set.insert(sanitize_identifier(name));
        let trimmed = name.trim_start_matches('/');
        if trimmed != name {
            set.insert(trimmed.to_string());
        }
    }

    /// Record a zero-element optional-input placeholder that must not become a WebNN constant.
    pub fn mark_empty_optional(&mut self, name: &str) {
        Self::insert_name_aliases(&mut self.empty_optional_values, name);
    }

    pub fn is_empty_optional(&self, name: &str) -> bool {
        if self.empty_optional_values.contains(name) {
            return true;
        }
        let sanitized = sanitize_identifier(name);
        if self.empty_optional_values.contains(&sanitized) {
            return true;
        }
        let trimmed = name.trim_start_matches('/');
        self.empty_optional_values.contains(trimmed)
    }

    pub fn record_operand(&mut self, keys: &[&str], op: MLOperand) {
        for key in keys {
            if key.is_empty() {
                continue;
            }
            self.operands.insert(key.to_string(), op);
            let sanitized = sanitize_identifier(key);
            self.operands.insert(sanitized, op);
        }
    }

    pub fn resolve_operand(&self, name: &str) -> Result<MLOperand, OnnxError> {
        if self.is_empty_optional(name) {
            return Err(OnnxError::InvalidShape(format!(
                "ONNX value '{name}' is an empty optional-input placeholder and has no WebNN operand"
            )));
        }
        if let Some(&op) = self.operands.get(name) {
            return Ok(op);
        }
        let sanitized = sanitize_identifier(name);
        if let Some(&op) = self.operands.get(&sanitized) {
            return Ok(op);
        }
        let trimmed = name.trim_start_matches('/');
        if let Some(&op) = self.operands.get(trimmed) {
            return Ok(op);
        }
        Err(OnnxError::InvalidShape(format!(
            "no MLOperand registered for ONNX value '{name}' (sanitized: '{sanitized}')"
        )))
    }

    pub fn register_input(
        &mut self,
        name: &str,
        data_type: DataType,
        shape: &[Dimension],
    ) -> Result<(), OnnxError> {
        let id = Self::webnn_id(name);
        let desc = descriptor_from_parts(data_type, shape)?;
        let op = self.builder.input(&id, &desc).map_err(map_rustnn_error)?;
        self.input_operands.insert(operand_index(op));
        self.input_names.insert(name.to_string());
        self.input_names.insert(id.clone());
        self.record_operand(&[name, &id], op);
        Ok(())
    }

    /// Resolve an ONNX graph output for `build()`.
    ///
    /// WebNN rejects graph outputs that are still inputs or constants (see section 8.9.4 `build()`).
    /// Insert `identity` only for those cases; regular op outputs already have graph-safe names.
    pub fn output_operand(&mut self, name: &str) -> Result<MLOperand, OnnxError> {
        let op = self.resolve_operand(name)?;
        let idx = operand_index(op);
        if !self.input_operands.contains(&idx) && !self.constant_operands.contains(&idx) {
            return Ok(op);
        }
        let label = format!("{}__graph_out", Self::webnn_id(name));
        let opts = Self::labeled_options(&label);
        self.builder
            .identity_with_options(op, opts)
            .map_err(map_op_error)
    }

    /// Build-time output key; disambiguate when ONNX reuses an input name as output.
    pub fn build_output_key(&self, output_name: &str) -> String {
        Self::output_key_for(output_name, &self.input_names)
    }

    /// WebNN graph output key for an ONNX output name (used by tests).
    pub fn output_key_for(output_name: &str, input_names: &HashSet<String>) -> String {
        let sanitized = Self::webnn_id(output_name);
        if input_names.contains(&sanitized) || input_names.contains(output_name) {
            format!("{sanitized}__output")
        } else {
            sanitized
        }
    }

    /// Register a constant, handing the byte buffer to rustnn without a copy.
    /// Weight tensors reach hundreds of MB and the buffer stays alive for the
    /// whole backend compile, so every avoided copy matters. rustnn stores
    /// plain little-endian bytes, so no typed round-trip is needed.
    pub fn register_constant_from_bytes(
        &mut self,
        name: &str,
        data_type: DataType,
        shape: &[u32],
        bytes: Vec<u8>,
    ) -> Result<(), OnnxError> {
        let id = Self::webnn_id(name);
        let desc = descriptor_static(data_type, shape)?;
        match data_type {
            DataType::Float32
            | DataType::Float16
            | DataType::Int32
            | DataType::Int64
            | DataType::Uint8
            | DataType::Int8
            | DataType::Uint4
            | DataType::Int4 => {}
            other => {
                return Err(OnnxError::InvalidShape(format!(
                    "unsupported constant data type for builder: {other:?}"
                )));
            }
        }
        let op = self
            .builder
            .constant_from_bytes(&desc, bytes)
            .map_err(map_rustnn_error)?;
        self.constant_operands.insert(operand_index(op));
        self.record_operand(&[name, &id], op);
        Ok(())
    }

    pub fn labeled_options(label: &str) -> MLOperatorOptions {
        MLOperatorOptions {
            label: label.to_string(),
        }
    }

    pub fn finish_build(
        &mut self,
        outputs: HashMap<&str, MLOperand>,
    ) -> Result<MLGraph<'ctx>, OnnxError> {
        let outputs: std::collections::BTreeMap<&str, MLOperand> = outputs.into_iter().collect();
        self.builder.build(&outputs).map_err(map_rustnn_error)
    }
}

pub fn map_rustnn_error(err: RustnnError) -> OnnxError {
    OnnxError::ShapeInference(err.to_string())
}

pub fn map_op_error(err: GraphBuilderError) -> OnnxError {
    OnnxError::ShapeInference(err.to_string())
}

pub fn descriptor_static(
    data_type: DataType,
    shape: &[u32],
) -> Result<MLOperandDescriptor, OnnxError> {
    let dt = map_ast_data_type(data_type)?;
    Ok(MLOperandDescriptor::new(
        dt,
        shape.iter().map(|&d| d as u64).collect(),
    ))
}

pub fn descriptor_from_parts(
    data_type: DataType,
    shape: &[Dimension],
) -> Result<MLOperandDescriptor, OnnxError> {
    let dt = map_ast_data_type(data_type)?;
    let mut dims = Vec::with_capacity(shape.len());
    for dim in shape {
        match dim {
            Dimension::Static(v) => dims.push(*v as u64),
            Dimension::Dynamic(d) => dims.push(d.max_size as u64),
        }
    }
    Ok(MLOperandDescriptor::new(dt, dims))
}

pub fn map_ast_data_type(dt: DataType) -> Result<MLOperandDataType, OnnxError> {
    Ok(match dt {
        DataType::Float32 => MLOperandDataType::Float32,
        DataType::Float16 => MLOperandDataType::Float16,
        DataType::Int32 => MLOperandDataType::Int32,
        DataType::Uint32 => MLOperandDataType::Uint32,
        DataType::Int64 => MLOperandDataType::Int64,
        DataType::Uint64 => MLOperandDataType::Uint64,
        DataType::Int8 => MLOperandDataType::Int8,
        DataType::Uint8 => MLOperandDataType::Uint8,
        DataType::Int4 => MLOperandDataType::Int4,
        DataType::Uint4 => MLOperandDataType::Uint4,
    })
}

pub fn map_onnx_tensor_type(onnx_type: i32) -> Result<MLOperandDataType, OnnxError> {
    map_ast_data_type(map_onnx_data_type(onnx_type)?)
}

/// Number of elements implied by `TensorProto.dims`.
///
/// An empty `dims` list is treated as a scalar (1 element). A dimension of `0`
/// yields an empty tensor (0 elements), which ONNX exporters commonly use as a
/// placeholder for unused optional inputs such as `Resize`'s `roi`/`scales`.
pub fn tensor_element_count(tensor: &TensorProto) -> usize {
    if tensor.dims.is_empty() {
        return 1;
    }
    tensor
        .dims
        .iter()
        .fold(1usize, |acc, &d| acc.saturating_mul(d.max(0) as usize))
}

/// Extract initializer / constant tensor bytes for `constant_from_slice`.
pub fn tensor_proto_to_bytes(tensor: &TensorProto) -> Result<Vec<u8>, OnnxError> {
    if tensor.data_type == TensorProto_DataType::Double as i32 {
        // float64 is lowered to float32 (WebNN has no float64).
        let values: Vec<f64> = if !tensor.raw_data.is_empty() {
            tensor
                .raw_data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        } else {
            tensor.double_data.clone()
        };
        return Ok(values
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect());
    }
    if !tensor.raw_data.is_empty() {
        return Ok(tensor.raw_data.clone());
    }
    if !tensor.float_data.is_empty() {
        return Ok(tensor
            .float_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect());
    }
    if !tensor.int32_data.is_empty() {
        return Ok(match tensor.data_type {
            x if x == TensorProto_DataType::Uint8 as i32
                || x == TensorProto_DataType::Int8 as i32
                || x == TensorProto_DataType::Bool as i32 =>
            {
                tensor.int32_data.iter().map(|&v| v as u8).collect()
            }
            x if x == TensorProto_DataType::Float16 as i32
                || x == TensorProto_DataType::Bfloat16 as i32
                || x == TensorProto_DataType::Uint16 as i32
                || x == TensorProto_DataType::Int16 as i32 =>
            {
                tensor
                    .int32_data
                    .iter()
                    .flat_map(|&v| (v as u16).to_le_bytes())
                    .collect()
            }
            _ => tensor
                .int32_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        });
    }
    if !tensor.int64_data.is_empty() {
        return Ok(tensor
            .int64_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect());
    }
    if !tensor.double_data.is_empty() {
        return Ok(tensor
            .double_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect());
    }
    // Empty optional-input placeholders: dims contain a 0, so there is no payload.
    if tensor_element_count(tensor) == 0 {
        return Ok(Vec::new());
    }
    Err(OnnxError::InvalidShape(format!(
        "tensor '{}' has no payload",
        tensor.name
    )))
}

#[cfg(test)]
mod tests {
    use crate::onnx::convert::{convert_model, ConvertOptions};
    use crate::protos::onnx::{
        tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto, TensorProto_DataType,
        TensorShapeProto, ValueInfoProto,
    };

    #[test]
    fn test_empty_optional_tensor_proto_to_bytes() {
        use super::{tensor_element_count, tensor_proto_to_bytes};
        use crate::protos::onnx::{TensorProto, TensorProto_DataType};

        let tensor = TensorProto {
            name: String::new(),
            dims: vec![0],
            data_type: TensorProto_DataType::Float16 as i32,
            raw_data: Vec::new(),
            ..Default::default()
        };
        assert_eq!(tensor_element_count(&tensor), 0);
        let bytes = tensor_proto_to_bytes(&tensor).expect("empty tensor should be valid");
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_uint8_scalar_int32_payload_uses_one_byte() {
        use super::tensor_proto_to_bytes;
        use crate::protos::onnx::{TensorProto, TensorProto_DataType};

        let tensor = TensorProto {
            name: "zero_point".to_string(),
            data_type: TensorProto_DataType::Uint8 as i32,
            int32_data: vec![127],
            ..Default::default()
        };
        assert_eq!(tensor_proto_to_bytes(&tensor).unwrap(), vec![127]);
    }

    #[test]
    fn test_add_ort_build_succeeds() {
        let dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(2)),
            denotation: String::new(),
        };
        let shape = TensorShapeProto {
            dim: vec![dim.clone(), dim],
        };
        let tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Float.into(),
            shape: Some(shape.clone()),
        };
        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type.clone())),
            denotation: String::new(),
        };

        let a_input = ValueInfoProto {
            name: "a".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let b_input = ValueInfoProto {
            name: "b".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let out = ValueInfoProto {
            name: "c".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let add = NodeProto {
            op_type: "Add".to_string(),
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["c".to_string()],
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![a_input, b_input],
                output: vec![out],
                node: vec![add],
                ..Default::default()
            }),
            ..Default::default()
        };

        convert_model(model, &ConvertOptions::default()).expect("Add graph should build on ORT");
    }
}
