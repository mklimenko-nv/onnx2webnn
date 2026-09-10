/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Einsum operator handler.
//
// WebNN has no general contraction primitive, so equations are lowered to a
// transpose/reshape/matmul pipeline:
//   * labels present in only one input and absent from the output are
//     reduce-summed away first,
//   * the two-input case classifies the remaining labels into batch (shared,
//     kept), contracted (shared, summed) and free (per-input, kept) groups,
//     transposes each operand to [batch..., free..., contracted...] order,
//     flattens to 3-D, contracts with a single batched matmul, and restores
//     the requested output order,
//   * the single-input case is a reduce-sum plus transpose.
//
// Rejected: ellipsis ("..."), repeated labels within one operand (diagonals),
// more than two inputs, and operands whose static shapes are unknown.

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{
    i64_slice_to_mldim, output_label, record_node_output, reshape_with_shape,
};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::conv::lookup_shape;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::{MLReduceOptions, MLTransposeOptions};

pub struct EinsumHandler;

impl OpHandler for EinsumHandler {
    fn supports(&self, op_type: &str) -> bool {
        op_type == "Einsum"
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
        convert_einsum(node, &node_name, context, b)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedEinsum {
    pub input_specs: Vec<Vec<char>>,
    pub output_spec: Vec<char>,
}

/// Parse an einsum equation into per-input and output label lists.
///
/// Rejects ellipsis and repeated labels within a single term; the implicit
/// output (no `->`) is the alphabetically sorted set of labels that appear
/// exactly once across all inputs, following the ONNX/numpy convention.
pub(crate) fn parse_einsum_equation(
    equation: &str,
    n_inputs: usize,
    node_name: &str,
) -> Result<ParsedEinsum, OnnxError> {
    let compact: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.contains("...") {
        return Err(OnnxError::unsupported_op(
            "Einsum(ellipsis)",
            node_name.to_string(),
        ));
    }

    let (lhs, explicit_output) = match compact.split_once("->") {
        Some((lhs, rhs)) => (lhs, Some(rhs)),
        None => (compact.as_str(), None),
    };

    let mut input_specs: Vec<Vec<char>> = Vec::new();
    for term in lhs.split(',') {
        let labels: Vec<char> = term.chars().collect();
        for &c in &labels {
            if !c.is_ascii_alphabetic() {
                return Err(OnnxError::InvalidShape(format!(
                    "Einsum equation {equation:?} contains invalid label {c:?}"
                )));
            }
        }
        let mut seen = labels.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != labels.len() {
            return Err(OnnxError::unsupported_op(
                "Einsum(repeated label in one term)",
                node_name.to_string(),
            ));
        }
        input_specs.push(labels);
    }
    if input_specs.len() != n_inputs {
        return Err(OnnxError::InvalidShape(format!(
            "Einsum equation {equation:?} has {} terms but the node has {n_inputs} inputs",
            input_specs.len()
        )));
    }

    let output_spec: Vec<char> = match explicit_output {
        Some(rhs) => {
            let labels: Vec<char> = rhs.chars().collect();
            let mut seen = labels.clone();
            seen.sort_unstable();
            seen.dedup();
            if seen.len() != labels.len() {
                return Err(OnnxError::InvalidShape(format!(
                    "Einsum equation {equation:?} repeats a label in its output"
                )));
            }
            for &c in &labels {
                if !input_specs.iter().any(|spec| spec.contains(&c)) {
                    return Err(OnnxError::InvalidShape(format!(
                        "Einsum equation {equation:?} output label {c:?} is not an input label"
                    )));
                }
            }
            labels
        }
        None => {
            let mut counts: Vec<(char, usize)> = Vec::new();
            for spec in &input_specs {
                for &c in spec {
                    match counts.iter_mut().find(|(l, _)| *l == c) {
                        Some((_, n)) => *n += 1,
                        None => counts.push((c, 1)),
                    }
                }
            }
            let mut labels: Vec<char> = counts
                .into_iter()
                .filter_map(|(c, n)| (n == 1).then_some(c))
                .collect();
            labels.sort_unstable();
            labels
        }
    };

    Ok(ParsedEinsum {
        input_specs,
        output_spec,
    })
}

/// Static output shape for an einsum node, for shape inference.
pub(crate) fn einsum_output_shape(equation: &str, input_shapes: &[Vec<i64>]) -> Option<Vec<i64>> {
    let parsed = parse_einsum_equation(equation, input_shapes.len(), "shape-inference").ok()?;
    let mut dims: Vec<(char, i64)> = Vec::new();
    for (spec, shape) in parsed.input_specs.iter().zip(input_shapes) {
        if spec.len() != shape.len() {
            return None;
        }
        for (&label, &dim) in spec.iter().zip(shape) {
            match dims.iter().find(|(l, _)| *l == label) {
                Some(&(_, prev)) if prev != dim => return None,
                Some(_) => {}
                None => dims.push((label, dim)),
            }
        }
    }
    parsed
        .output_spec
        .iter()
        .map(|label| dims.iter().find(|(l, _)| l == label).map(|&(_, d)| d))
        .collect()
}

/// One einsum operand together with its current label order and dimensions.
struct Term {
    operand: MLOperand,
    labels: Vec<char>,
    dims: Vec<i64>,
}

impl Term {
    fn dim_of(&self, label: char) -> i64 {
        let idx = self.labels.iter().position(|&l| l == label).expect("label");
        self.dims[idx]
    }
}

fn convert_einsum(
    node: &NodeProto,
    node_name: &str,
    context: &ConversionContext,
    b: &mut OnnxBuilder<'_, '_, '_>,
) -> Result<ConversionResult, OnnxError> {
    let equation = node
        .attribute
        .iter()
        .find(|attr| attr.name == "equation")
        .map(|attr| String::from_utf8_lossy(&attr.s).to_string())
        .ok_or_else(|| {
            OnnxError::InvalidShape(format!("Einsum node {node_name} has no equation"))
        })?;

    let inputs = node.input.as_slice();
    if inputs.is_empty() || inputs.len() > 2 {
        return Err(OnnxError::unsupported_op(
            format!("Einsum({} inputs)", inputs.len()),
            node_name.to_string(),
        ));
    }
    let parsed = parse_einsum_equation(&equation, inputs.len(), node_name)?;

    let label = output_label(node, node_name);
    let mut terms: Vec<Term> = Vec::new();
    for (idx, (name, spec)) in inputs.iter().zip(&parsed.input_specs).enumerate() {
        let shape = lookup_shape(name, context).ok_or_else(|| {
            OnnxError::InvalidShape(format!("Einsum requires a known static shape for '{name}'"))
        })?;
        if shape.len() != spec.len() {
            return Err(OnnxError::InvalidShape(format!(
                "Einsum term {idx} of {equation:?} has {} labels but '{name}' has rank {} (shape {shape:?})",
                spec.len(),
                shape.len()
            )));
        }
        if shape.iter().any(|&d| d <= 0) {
            return Err(OnnxError::InvalidShape(format!(
                "Einsum requires static dimensions for '{name}', got {shape:?}"
            )));
        }
        terms.push(Term {
            operand: b.resolve_operand(name)?,
            labels: spec.clone(),
            dims: shape,
        });
    }

    // Sum away labels unique to a single input and absent from the output.
    for (idx, term) in terms.iter_mut().enumerate() {
        let other_labels: Vec<char> = parsed
            .input_specs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .flat_map(|(_, spec)| spec.iter().copied())
            .collect();
        let summed: Vec<char> = term
            .labels
            .iter()
            .copied()
            .filter(|l| !parsed.output_spec.contains(l) && !other_labels.contains(l))
            .collect();
        if summed.is_empty() {
            continue;
        }
        let axes: Vec<u32> = term
            .labels
            .iter()
            .enumerate()
            .filter_map(|(axis, l)| summed.contains(l).then_some(axis as u32))
            .collect();
        term.operand = b
            .builder
            .reduce_sum_with_options(
                term.operand,
                MLReduceOptions {
                    label: format!("{label}__sum_in{idx}"),
                    axes: Some(axes),
                    keep_dimensions: false,
                },
            )
            .map_err(map_op_error)?;
        let (labels, dims) = term
            .labels
            .iter()
            .zip(&term.dims)
            .filter(|(l, _)| !summed.contains(l))
            .map(|(&l, &d)| (l, d))
            .unzip();
        term.labels = labels;
        term.dims = dims;
    }

    let out = match terms.len() {
        1 => {
            let term = &terms[0];
            transpose_to(b, term, &parsed.output_spec, &format!("{label}__perm"))?
        }
        2 => contract_pair(b, context, &parsed, &terms, &label)?,
        _ => unreachable!("input count validated above"),
    };

    if let Some(onnx_out) = node.output.first() {
        record_node_output(b, onnx_out, &label, out);
    } else {
        b.record_operand(&[&label], out);
    }
    Ok(ConversionResult::default())
}

/// Transpose `term` so its labels appear in `target` order (no-op when already ordered).
fn transpose_to(
    b: &mut OnnxBuilder<'_, '_, '_>,
    term: &Term,
    target: &[char],
    label: &str,
) -> Result<MLOperand, OnnxError> {
    let permutation: Vec<u32> = target
        .iter()
        .map(|l| term.labels.iter().position(|t| t == l).expect("label") as u32)
        .collect();
    if permutation.iter().enumerate().all(|(i, &p)| i as u32 == p) {
        return Ok(term.operand);
    }
    b.builder
        .transpose_with_options(
            term.operand,
            MLTransposeOptions {
                label: label.to_string(),
                permutation,
            },
        )
        .map_err(map_op_error)
}

fn contract_pair(
    b: &mut OnnxBuilder<'_, '_, '_>,
    _context: &ConversionContext,
    parsed: &ParsedEinsum,
    terms: &[Term],
    label: &str,
) -> Result<MLOperand, OnnxError> {
    let (a, bt) = (&terms[0], &terms[1]);

    // Batch labels keep output order so the final transpose is often a no-op.
    let batch: Vec<char> = parsed
        .output_spec
        .iter()
        .copied()
        .filter(|l| a.labels.contains(l) && bt.labels.contains(l))
        .collect();
    let contracted: Vec<char> = a
        .labels
        .iter()
        .copied()
        .filter(|l| bt.labels.contains(l) && !parsed.output_spec.contains(l))
        .collect();
    let free_a: Vec<char> = a
        .labels
        .iter()
        .copied()
        .filter(|l| !batch.contains(l) && !contracted.contains(l))
        .collect();
    let free_b: Vec<char> = bt
        .labels
        .iter()
        .copied()
        .filter(|l| !batch.contains(l) && !contracted.contains(l))
        .collect();

    let dim_product = |term: &Term, labels: &[char]| -> i64 {
        labels.iter().map(|&l| term.dim_of(l)).product::<i64>()
    };
    let batch_dims: Vec<i64> = batch.iter().map(|&l| a.dim_of(l)).collect();
    let (nb, fa, fb, c) = (
        batch_dims.iter().product::<i64>(),
        dim_product(a, &free_a),
        dim_product(bt, &free_b),
        dim_product(a, &contracted),
    );

    // a -> [batch, free_a, contracted] -> [NB, FA, C]
    let a_order: Vec<char> = batch
        .iter()
        .chain(&free_a)
        .chain(&contracted)
        .copied()
        .collect();
    let a_t = transpose_to(b, a, &a_order, &format!("{label}__a_perm"))?;
    let a_3d = reshape_with_shape(
        b,
        a_t,
        &format!("{label}__a_3d"),
        i64_slice_to_mldim(&[nb, fa, c])?,
    )?;

    // b -> [batch, contracted, free_b] -> [NB, C, FB]
    let b_order: Vec<char> = batch
        .iter()
        .chain(&contracted)
        .chain(&free_b)
        .copied()
        .collect();
    let b_t = transpose_to(b, bt, &b_order, &format!("{label}__b_perm"))?;
    let b_3d = reshape_with_shape(
        b,
        b_t,
        &format!("{label}__b_3d"),
        i64_slice_to_mldim(&[nb, c, fb])?,
    )?;

    let product = b
        .builder
        .matmul_with_options(
            a_3d,
            b_3d,
            OnnxBuilder::labeled_options(&format!("{label}__matmul")),
        )
        .map_err(map_op_error)?;

    // [NB, FA, FB] -> [batch..., free_a..., free_b...] -> output order.
    let mut unpacked_dims = batch_dims;
    unpacked_dims.extend(free_a.iter().map(|&l| a.dim_of(l)));
    unpacked_dims.extend(free_b.iter().map(|&l| bt.dim_of(l)));
    let unpacked = reshape_with_shape(
        b,
        product,
        &format!("{label}__unpacked"),
        i64_slice_to_mldim(&unpacked_dims)?,
    )?;

    let unpacked_labels: Vec<char> = batch
        .iter()
        .chain(&free_a)
        .chain(&free_b)
        .copied()
        .collect();
    let unpacked_term = Term {
        operand: unpacked,
        labels: unpacked_labels,
        dims: unpacked_dims,
    };
    transpose_to(
        b,
        &unpacked_term,
        &parsed.output_spec,
        &format!("{label}__out_perm"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::AttributeProto;
    use rustnn::DataType;
    use std::collections::HashMap;

    fn einsum_node(equation: &str, inputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: "Einsum".to_string(),
            name: "test_einsum".to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec!["y".to_string()],
            attribute: vec![AttributeProto {
                name: "equation".to_string(),
                s: equation.as_bytes().to_vec(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn convert_with_shapes(
        equation: &str,
        shapes: &[(&str, Vec<i64>)],
    ) -> Result<ConversionResult, OnnxError> {
        let node = einsum_node(equation, shapes.iter().map(|(n, _)| *n).collect());
        let value_shapes: HashMap<String, Vec<i64>> = shapes
            .iter()
            .map(|(n, s)| (n.to_string(), s.clone()))
            .collect();
        let value_types: HashMap<String, DataType> = shapes
            .iter()
            .map(|(n, _)| (n.to_string(), DataType::Float32))
            .collect();
        let initializers = HashMap::new();
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
        crate::onnx::ops::convert_handler_with_context(&EinsumHandler, &node, &context)
    }

    #[test]
    fn parses_explicit_and_implicit_output() {
        let parsed = parse_einsum_equation("ij,jk->ik", 2, "t").unwrap();
        assert_eq!(parsed.output_spec, vec!['i', 'k']);
        let implicit = parse_einsum_equation("ij,jk", 2, "t").unwrap();
        assert_eq!(implicit.output_spec, vec!['i', 'k']);
    }

    #[test]
    fn rejects_ellipsis_and_diagonals() {
        assert!(matches!(
            parse_einsum_equation("...ij,jk->...ik", 2, "t").unwrap_err(),
            OnnxError::UnsupportedOps(_)
        ));
        assert!(matches!(
            parse_einsum_equation("ii->i", 1, "t").unwrap_err(),
            OnnxError::UnsupportedOps(_)
        ));
    }

    #[test]
    fn infers_output_shapes() {
        assert_eq!(
            einsum_output_shape("i,j->ij", &[vec![5], vec![3]]),
            Some(vec![5, 3])
        );
        assert_eq!(
            einsum_output_shape("bid,bjd->bij", &[vec![2, 4, 8], vec![2, 6, 8]]),
            Some(vec![2, 4, 6])
        );
        assert_eq!(
            einsum_output_shape("ij->ji", &[vec![2, 3]]),
            Some(vec![3, 2])
        );
        // Mismatched shared dimension.
        assert_eq!(
            einsum_output_shape("ij,jk->ik", &[vec![2, 3], vec![4, 5]]),
            None
        );
    }

    #[test]
    fn converts_outer_product() {
        convert_with_shapes("i,j->ij", &[("a", vec![4]), ("b", vec![6])]).unwrap();
    }

    #[test]
    fn converts_matmul_equation() {
        convert_with_shapes("ij,jk->ik", &[("a", vec![2, 3]), ("b", vec![3, 5])]).unwrap();
    }

    #[test]
    fn converts_batched_attention_equation() {
        convert_with_shapes(
            "bhid,bhjd->bhij",
            &[("q", vec![2, 4, 7, 16]), ("k", vec![2, 4, 9, 16])],
        )
        .unwrap();
    }

    #[test]
    fn converts_transpose_equation() {
        convert_with_shapes("ij->ji", &[("a", vec![2, 3])]).unwrap();
    }

    #[test]
    fn converts_single_input_reduction() {
        convert_with_shapes("ij->i", &[("a", vec![2, 3])]).unwrap();
    }

    #[test]
    fn rejects_three_inputs() {
        let err = convert_with_shapes(
            "i,j,k->ijk",
            &[("a", vec![2]), ("b", vec![3]), ("c", vec![4])],
        )
        .unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedOps(_)));
    }
}
