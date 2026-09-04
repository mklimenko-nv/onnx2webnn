# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom ONNX op test graph builders for operators the generic builder cannot satisfy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from onnx import ModelProto, TensorProto, checker, helper, numpy_helper

DEFAULT_VECTOR_SHAPE = [1, 2]
SPATIAL_SHAPE = [1, 1, 4, 4]
SEQ_FLOAT = helper.make_sequence_type_proto(
    helper.make_tensor_type_proto(TensorProto.FLOAT, DEFAULT_VECTOR_SHAPE)
)


def _model(graph, opset: int) -> ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    checker.check_model(model)
    return model


def _f32(name: str, shape: list[int]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _i32(name: str, shape: list[int]):
    return helper.make_tensor_value_info(name, TensorProto.INT32, shape)


def _i64(name: str, shape: list[int]):
    return helper.make_tensor_value_info(name, TensorProto.INT64, shape)


def _u8(name: str, shape: list[int]):
    return helper.make_tensor_value_info(name, TensorProto.UINT8, shape)


def _build_batch_normalization(opset: int) -> ModelProto:
    x = _f32("X", SPATIAL_SHAPE)
    scale = numpy_helper.from_array(np.ones(1, dtype=np.float32), "scale")
    bias = numpy_helper.from_array(np.zeros(1, dtype=np.float32), "B")
    mean = numpy_helper.from_array(np.zeros(1, dtype=np.float32), "input_mean")
    var = numpy_helper.from_array(np.ones(1, dtype=np.float32), "input_var")
    node = helper.make_node(
        "BatchNormalization",
        ["X", "scale", "B", "input_mean", "input_var"],
        ["Y"],
        name="test_BatchNormalization",
        epsilon=1e-5,
        training_mode=0,
    )
    graph = helper.make_graph(
        [node],
        "test_BatchNormalization_graph",
        [x],
        [_f32("Y", SPATIAL_SHAPE)],
        [scale, bias, mean, var],
    )
    return _model(graph, opset)


def _build_bitwise(op_type: str, onnx_op: str, opset: int) -> ModelProto:
    a = numpy_helper.from_array(np.array([[1, 0]], dtype=np.int32), "A")
    if onnx_op == "BitwiseNot":
        node = helper.make_node(onnx_op, ["A"], ["C"], name=f"test_{op_type}")
        inits = [a]
    else:
        b = numpy_helper.from_array(np.array([[0, 1]], dtype=np.int32), "B")
        node = helper.make_node(onnx_op, ["A", "B"], ["C"], name=f"test_{op_type}")
        inits = [a, b]
    graph = helper.make_graph(
        [node],
        f"test_{op_type}_graph",
        [],
        [_i32("C", DEFAULT_VECTOR_SHAPE)],
        inits,
    )
    return _model(graph, opset)


def _build_bitshift(opset: int) -> ModelProto:
    x = numpy_helper.from_array(np.array([1, 2], dtype=np.uint8), "X")
    y = numpy_helper.from_array(np.array(1, dtype=np.uint8), "Y")
    node = helper.make_node("BitShift", ["X", "Y"], ["Z"], direction="LEFT", name="test_BitShift")
    graph = helper.make_graph(
        [node],
        "test_BitShift_graph",
        [],
        [helper.make_tensor_value_info("Z", TensorProto.UINT8, DEFAULT_VECTOR_SHAPE)],
        [x, y],
    )
    return _model(graph, opset)


def _build_window(op_type: str, opset: int) -> ModelProto:
    size = numpy_helper.from_array(np.array(8, dtype=np.int64), "size")
    node = helper.make_node(op_type, ["size"], ["output"], name=f"test_{op_type}")
    graph = helper.make_graph(
        [node],
        f"test_{op_type}_graph",
        [],
        [_f32("output", [8])],
        [size],
    )
    return _model(graph, opset)


def _build_affine_grid(opset: int) -> ModelProto:
    theta = _f32("theta", [1, 2, 3])
    size = numpy_helper.from_array(np.array([1, 3, 4], dtype=np.int64), "size")
    node = helper.make_node("AffineGrid", ["theta", "size"], ["output"], align_corners=0, name="test")
    graph = helper.make_graph([node], "test_AffineGrid_graph", [theta], [_f32("output", [1, 3, 4, 2])], [size])
    return _model(graph, opset)


def _build_attention(opset: int) -> ModelProto:
    q = _f32("Q", [1, 2, 4, 8])
    k = _f32("K", [1, 2, 4, 8])
    v = _f32("V", [1, 2, 4, 8])
    node = helper.make_node(
        "Attention",
        ["Q", "K", "V"],
        ["Y"],
        name="test_Attention",
        q_num_heads=2,
        kv_num_heads=2,
    )
    graph = helper.make_graph([node], "test_Attention_graph", [q, k, v], [_f32("Y", [1, 2, 4, 8])])
    return _model(graph, opset)


def _build_cum_axis(op_type: str, opset: int) -> ModelProto:
    x = _f32("x", DEFAULT_VECTOR_SHAPE)
    axis = numpy_helper.from_array(np.array(0, dtype=np.int64), "axis")
    node = helper.make_node(op_type, ["x", "axis"], ["y"], name=f"test_{op_type}")
    graph = helper.make_graph([node], f"test_{op_type}_graph", [x], [_f32("y", DEFAULT_VECTOR_SHAPE)], [axis])
    return _model(graph, opset)


def _build_col2im(opset: int) -> ModelProto:
    # Minimal valid Col2Im (opset 18+): input, image_shape, block_shape.
    input_data = numpy_helper.from_array(
        np.array(
            [
                [
                    [1.0, 6.0, 11.0, 16.0, 21.0],
                    [2.0, 7.0, 12.0, 17.0, 22.0],
                    [3.0, 8.0, 13.0, 18.0, 23.0],
                    [4.0, 9.0, 14.0, 19.0, 24.0],
                    [5.0, 0.0, 15.0, 20.0, 25.0],
                ]
            ],
            dtype=np.float32,
        ),
        "input",
    )
    image_shape = numpy_helper.from_array(np.array([5, 5], dtype=np.int64), "image_shape")
    block_shape = numpy_helper.from_array(np.array([1, 5], dtype=np.int64), "block_shape")
    node = helper.make_node(
        "Col2Im",
        ["input", "image_shape", "block_shape"],
        ["output"],
        name="test_Col2Im",
    )
    graph = helper.make_graph(
        [node],
        "test_Col2Im_graph",
        [],
        [_f32("output", [1, 1, 5, 5])],
        [input_data, image_shape, block_shape],
    )
    return _model(graph, opset)


def _build_compress(opset: int) -> ModelProto:
    data = _f32("data", DEFAULT_VECTOR_SHAPE)
    cond = numpy_helper.from_array(np.array([1, 0], dtype=np.bool_), "condition")
    node = helper.make_node("Compress", ["data", "condition"], ["output"], axis=0, name="test")
    graph = helper.make_graph(
        [node],
        "test_Compress_graph",
        [data],
        [_f32("output", [1])],
        [cond],
    )
    return _model(graph, opset)


def _build_constant_of_shape(opset: int) -> ModelProto:
    shape_in = numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "input")
    node = helper.make_node(
        "ConstantOfShape",
        ["input"],
        ["output"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [1], [1.0]),
        name="test",
    )
    graph = helper.make_graph(
        [node],
        "test_ConstantOfShape_graph",
        [],
        [_f32("output", DEFAULT_VECTOR_SHAPE)],
        [shape_in],
    )
    return _model(graph, opset)


def _build_det(opset: int) -> ModelProto:
    x = _f32("X", [2, 2])
    node = helper.make_node("Det", ["X"], ["Y"], name="test_Det")
    graph = helper.make_graph([node], "test_Det_graph", [x], [_f32("Y", [])])
    return _model(graph, opset)


def _build_einsum(opset: int) -> ModelProto:
    a = _f32("a", [2, 2])
    b = _f32("b", [2, 2])
    node = helper.make_node("Einsum", ["a", "b"], ["c"], equation="ij,jk->ik", name="test")
    graph = helper.make_graph([node], "test_Einsum_graph", [a, b], [_f32("c", [2, 2])])
    return _model(graph, opset)


def _build_gru(opset: int) -> ModelProto:
    x = _f32("X", [1, 3, 4])
    w = _f32("W", [1, 12, 4])
    r = _f32("R", [1, 12, 4])
    b = _f32("B", [1, 24])
    node = helper.make_node("GRU", ["X", "W", "R", "B"], ["Y"], name="test_GRU", hidden_size=4)
    graph = helper.make_graph([node], "test_GRU_graph", [x, w, r, b], [_f32("Y", [1, 1, 4])])
    return _model(graph, opset)


def _build_lstm(opset: int) -> ModelProto:
    x = _f32("X", [1, 3, 4])
    w = _f32("W", [1, 16, 4])
    r = _f32("R", [1, 16, 4])
    b = _f32("B", [1, 32])
    node = helper.make_node("LSTM", ["X", "W", "R", "B"], ["Y"], name="test_LSTM", hidden_size=4)
    graph = helper.make_graph([node], "test_LSTM_graph", [x, w, r, b], [_f32("Y", [1, 1, 4])])
    return _model(graph, opset)


def _build_rnn(opset: int) -> ModelProto:
    x = _f32("X", [1, 3, 4])
    w = _f32("W", [1, 4, 4])
    r = _f32("R", [1, 4, 4])
    b = _f32("B", [1, 8])
    node = helper.make_node("RNN", ["X", "W", "R", "B"], ["Y"], name="test_RNN", hidden_size=4)
    graph = helper.make_graph([node], "test_RNN_graph", [x, w, r, b], [_f32("Y", [1, 1, 4])])
    return _model(graph, opset)


def _build_grid_sample(opset: int) -> ModelProto:
    x = _f32("X", [1, 1, 4, 4])
    grid = _f32("grid", [1, 4, 4, 2])
    node = helper.make_node("GridSample", ["X", "grid"], ["Y"], name="test", mode="bilinear")
    graph = helper.make_graph([node], "test_GridSample_graph", [x, grid], [_f32("Y", [1, 1, 4, 4])])
    return _model(graph, opset)


def _build_if(opset: int) -> ModelProto:
    then_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["then_out"], value=helper.make_tensor("v", TensorProto.FLOAT, [], [1.0]))],
        "then",
        [],
        [_f32("then_out", DEFAULT_VECTOR_SHAPE)],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["else_out"], value=helper.make_tensor("v", TensorProto.FLOAT, [], [2.0]))],
        "else",
        [],
        [_f32("else_out", DEFAULT_VECTOR_SHAPE)],
    )
    cond = numpy_helper.from_array(np.array(True), "cond")
    node = helper.make_node("If", ["cond"], ["out"], then_branch=then_graph, else_branch=else_graph, name="test_If")
    graph = helper.make_graph(
        [node],
        "test_If_graph",
        [],
        [_f32("out", DEFAULT_VECTOR_SHAPE)],
        [cond],
    )
    return _model(graph, opset)


def _build_loop(opset: int) -> ModelProto:
    false_val = helper.make_tensor("false", TensorProto.BOOL, [], [False])
    body = helper.make_graph(
        [
            helper.make_node("Constant", [], ["cond_out"], value=false_val, name="set_false"),
            helper.make_node("Identity", ["v_in"], ["v_out"], name="pass"),
        ],
        "body",
        [
            helper.make_tensor_value_info("iteration_num", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_in", TensorProto.INT64, []),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_out", TensorProto.INT64, []),
        ],
    )
    trip_count = numpy_helper.from_array(np.array(1, dtype=np.int64), "M")
    cond_init = numpy_helper.from_array(np.array(True, dtype=np.bool_), "cond")
    v_init = numpy_helper.from_array(np.array(0, dtype=np.int64), "v_init")
    node = helper.make_node("Loop", ["M", "cond", "v_init"], ["v_final"], body=body, name="test_Loop")
    graph = helper.make_graph(
        [node],
        "test_Loop_graph",
        [],
        [_i64("v_final", [])],
        [trip_count, cond_init, v_init],
    )
    return _model(graph, opset)


def _build_scan(opset: int) -> ModelProto:
    body = helper.make_graph(
        [helper.make_node("Identity", ["scan_in"], ["scan_out"], name="id")],
        "body",
        [_f32("scan_in", DEFAULT_VECTOR_SHAPE)],
        [_f32("scan_out", DEFAULT_VECTOR_SHAPE)],
    )
    init = numpy_helper.from_array(np.zeros((1, 2), dtype=np.float32), "init")
    scan = numpy_helper.from_array(np.zeros((1, 1, 2), dtype=np.float32), "scan")
    node = helper.make_node("Scan", ["init", "scan"], ["out"], body=body, num_scan_inputs=1, name="test_Scan")
    graph = helper.make_graph(
        [node],
        "test_Scan_graph",
        [],
        [_f32("out", [1, 1, 2])],
        [init, scan],
    )
    return _model(graph, opset)


def _build_negative_log_likelihood_loss(opset: int) -> ModelProto:
    x = _f32("input", [2, 3])
    target = _i32("target", [2])
    node = helper.make_node(
        "NegativeLogLikelihoodLoss",
        ["input", "target"],
        ["output"],
        name="test",
        reduction="mean",
    )
    graph = helper.make_graph(
        [node],
        "test_NegativeLogLikelihoodLoss_graph",
        [x, target],
        [_f32("output", [])],
    )
    return _model(graph, opset)


def _build_non_max_suppression(opset: int) -> ModelProto:
    boxes = _f32("boxes", [1, 4, 4])
    scores = _f32("scores", [1, 1, 4])
    node = helper.make_node("NonMaxSuppression", ["boxes", "scores"], ["selected_indices"], name="test")
    graph = helper.make_graph(
        [node],
        "test_NonMaxSuppression_graph",
        [boxes, scores],
        [_i64("selected_indices", [0, 3])],
    )
    return _model(graph, opset)


def _build_one_hot(opset: int) -> ModelProto:
    indices = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "indices")
    depth = numpy_helper.from_array(np.array(3, dtype=np.int64), "depth")
    values = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), "values")
    node = helper.make_node("OneHot", ["indices", "depth", "values"], ["output"], axis=-1, name="test")
    graph = helper.make_graph(
        [node],
        "test_OneHot_graph",
        [],
        [_f32("output", [2, 3])],
        [indices, depth, values],
    )
    return _model(graph, opset)


def _build_optional(opset: int) -> ModelProto:
    x = _f32("input", DEFAULT_VECTOR_SHAPE)
    node = helper.make_node("Optional", ["input"], ["output"], name="test")
    graph = helper.make_graph(
        [node],
        "test_Optional_graph",
        [x],
        [helper.make_value_info("output", helper.make_optional_type_proto(helper.make_tensor_type_proto(TensorProto.FLOAT, DEFAULT_VECTOR_SHAPE)))],
    )
    return _model(graph, opset)


def _build_quantize_linear(opset: int) -> ModelProto:
    x = _f32("x", DEFAULT_VECTOR_SHAPE)
    scale = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "y_scale")
    zp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "y_zero_point")
    node = helper.make_node("QuantizeLinear", ["x", "y_scale", "y_zero_point"], ["y"], name="test")
    graph = helper.make_graph([node], "test_QuantizeLinear_graph", [x], [_u8("y", DEFAULT_VECTOR_SHAPE)], [scale, zp])
    return _model(graph, opset)


def _build_dynamic_quantize_linear(opset: int) -> ModelProto:
    x = _f32("x", DEFAULT_VECTOR_SHAPE)
    node = helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "y_scale", "y_zero_point"], name="test")
    graph = helper.make_graph(
        [node],
        "test_DynamicQuantizeLinear_graph",
        [x],
        [_u8("y", DEFAULT_VECTOR_SHAPE), _f32("y_scale", []), _u8("y_zero_point", [])],
    )
    return _model(graph, opset)


def _build_range(opset: int) -> ModelProto:
    # Fractional float range exercises the dtype-aware lowering (start=0.5, limit=2.0, delta=0.5
    # -> [0.5, 1.0, 1.5]). Integer-valued floats would hide truncation bugs.
    start = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "start")
    limit = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "limit")
    delta = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "delta")
    node = helper.make_node("Range", ["start", "limit", "delta"], ["output"], name="test")
    graph = helper.make_graph(
        [node],
        "test_Range_graph",
        [],
        [_f32("output", [3])],
        [start, limit, delta],
    )
    return _model(graph, opset)


def _build_resize(opset: int) -> ModelProto:
    x = _f32("X", [1, 1, 4, 4])
    sizes = numpy_helper.from_array(np.array([1, 1, 6, 6], dtype=np.int64), "sizes")
    node = helper.make_node("Resize", ["X", "", "", "sizes"], ["Y"], mode="nearest", name="test")
    graph = helper.make_graph([node], "test_Resize_graph", [x], [_f32("Y", [1, 1, 6, 6])], [sizes])
    return _model(graph, opset)


def _build_reverse_sequence(opset: int) -> ModelProto:
    x = _f32("input", [3, 2, 4])
    lens = numpy_helper.from_array(np.array([2, 2, 2], dtype=np.int64), "sequence_lens")
    node = helper.make_node(
        "ReverseSequence",
        ["input", "sequence_lens"],
        ["Y"],
        name="test",
        batch_axis=0,
        time_axis=1,
    )
    graph = helper.make_graph([node], "test_ReverseSequence_graph", [x], [_f32("Y", [3, 2, 4])], [lens])
    return _model(graph, opset)


def _build_roi_align(opset: int) -> ModelProto:
    x = _f32("X", SPATIAL_SHAPE)
    rois = _f32("rois", [2, 4])
    bi = numpy_helper.from_array(np.array([0, 0], dtype=np.int64), "batch_indices")
    node = helper.make_node(
        "RoiAlign",
        ["X", "rois", "batch_indices"],
        ["Y"],
        name="test",
        output_height=2,
        output_width=2,
    )
    graph = helper.make_graph([node], "test_RoiAlign_graph", [x, rois], [_f32("Y", [2, 1, 2, 2])], [bi])
    return _model(graph, opset)


def _build_rotary_embedding(opset: int) -> ModelProto:
    x = _f32("x", [1, 4, 2, 2])
    cos_cache = numpy_helper.from_array(np.ones((1, 2, 1), dtype=np.float32), "cos_cache")
    sin_cache = numpy_helper.from_array(np.zeros((1, 2, 1), dtype=np.float32), "sin_cache")
    node = helper.make_node(
        "RotaryEmbedding",
        ["x", "cos_cache", "sin_cache"],
        ["out"],
        name="test",
        num_heads=4,
        rotary_embedding_dim=2,
    )
    graph = helper.make_graph(
        [node],
        "test_RotaryEmbedding_graph",
        [x],
        [_f32("out", [1, 4, 2, 2])],
        [cos_cache, sin_cache],
    )
    return _model(graph, opset)


def _build_split(opset: int) -> ModelProto:
    x = _f32("input", DEFAULT_VECTOR_SHAPE)
    # Opset 18+ uses `num_outputs`; older schemas use the `split` attribute.
    if opset >= 18:
        node = helper.make_node(
            "Split",
            ["input"],
            ["out0", "out1"],
            axis=1,
            num_outputs=2,
            name="test",
        )
    else:
        node = helper.make_node(
            "Split",
            ["input"],
            ["out0", "out1"],
            axis=1,
            split=[1, 1],
            name="test",
        )
    graph = helper.make_graph(
        [node],
        "test_Split_graph",
        [x],
        [_f32("out0", [1, 1]), _f32("out1", [1, 1])],
    )
    return _model(graph, opset)


def _build_sequence_construct(opset: int) -> ModelProto:
    t = numpy_helper.from_array(np.zeros((1, 2), dtype=np.float32), "t0")
    node = helper.make_node("SequenceConstruct", ["t0"], ["output_sequence"], name="test")
    graph = helper.make_graph(
        [node],
        "test_SequenceConstruct_graph",
        [],
        [helper.make_value_info("output_sequence", SEQ_FLOAT)],
        [t],
    )
    return _model(graph, opset)


def _build_sequence_empty(opset: int) -> ModelProto:
    node = helper.make_node("SequenceEmpty", [], ["output"], dtype=TensorProto.FLOAT, name="test")
    graph = helper.make_graph(
        [node],
        "test_SequenceEmpty_graph",
        [],
        [helper.make_value_info("output", SEQ_FLOAT)],
    )
    return _model(graph, opset)


def _build_sequence_at(opset: int) -> ModelProto:
    t = numpy_helper.from_array(np.zeros((1, 2), dtype=np.float32), "t0")
    pos = numpy_helper.from_array(np.array(0, dtype=np.int64), "position")
    seq = helper.make_node("SequenceConstruct", ["t0"], ["seq"], name="mk")
    at = helper.make_node("SequenceAt", ["seq", "position"], ["output"], name="test")
    graph = helper.make_graph(
        [seq, at],
        "test_SequenceAt_graph",
        [],
        [_f32("output", DEFAULT_VECTOR_SHAPE)],
        [t, pos],
    )
    return _model(graph, opset)


def _build_concat_from_sequence(opset: int) -> ModelProto:
    t0 = numpy_helper.from_array(np.zeros((1, 2), dtype=np.float32), "t0")
    t1 = numpy_helper.from_array(np.ones((1, 2), dtype=np.float32), "t1")
    seq = helper.make_node("SequenceConstruct", ["t0", "t1"], ["input_sequence"], name="mk")
    cat = helper.make_node("ConcatFromSequence", ["input_sequence"], ["output"], axis=0, name="test")
    graph = helper.make_graph(
        [seq, cat],
        "test_ConcatFromSequence_graph",
        [],
        [_f32("output", [2, 2])],
        [t0, t1],
    )
    return _model(graph, opset)


def _build_split_to_sequence(opset: int) -> ModelProto:
    x = _f32("input", [2, 2])
    node = helper.make_node("SplitToSequence", ["input"], ["output_sequence"], axis=0, name="test")
    graph = helper.make_graph(
        [node],
        "test_SplitToSequence_graph",
        [x],
        [helper.make_value_info("output_sequence", SEQ_FLOAT)],
    )
    return _model(graph, opset)


def _build_tile(opset: int) -> ModelProto:
    x = _f32("input", DEFAULT_VECTOR_SHAPE)
    repeats = numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "repeats")
    node = helper.make_node("Tile", ["input", "repeats"], ["output"], name="test")
    graph = helper.make_graph([node], "test_Tile_graph", [x], [_f32("output", DEFAULT_VECTOR_SHAPE)], [repeats])
    return _model(graph, opset)


def _build_topk(opset: int) -> ModelProto:
    x = _f32("X", DEFAULT_VECTOR_SHAPE)
    k = numpy_helper.from_array(np.array(1, dtype=np.int64), "K")
    node = helper.make_node("TopK", ["X", "K"], ["Values", "Indices"], axis=0, name="test")
    graph = helper.make_graph(
        [node],
        "test_TopK_graph",
        [x],
        [_f32("Values", [1]), _i64("Indices", [1])],
        [k],
    )
    return _model(graph, opset)


def _build_tensor_scatter(opset: int) -> ModelProto:
    data = _f32("data", DEFAULT_VECTOR_SHAPE)
    indices = numpy_helper.from_array(np.array(0, dtype=np.int64), "indices")
    updates = _f32("updates", DEFAULT_VECTOR_SHAPE)
    node = helper.make_node("TensorScatter", ["data", "indices", "updates"], ["output"], name="test")
    graph = helper.make_graph(
        [node],
        "test_TensorScatter_graph",
        [data, updates],
        [_f32("output", DEFAULT_VECTOR_SHAPE)],
        [indices],
    )
    return _model(graph, opset)


def _build_center_crop_pad(opset: int) -> ModelProto:
    x = _f32("input", SPATIAL_SHAPE)
    shape = numpy_helper.from_array(np.array([1, 1, 4, 4], dtype=np.int64), "shape")
    node = helper.make_node("CenterCropPad", ["input", "shape"], ["output"], name="test")
    graph = helper.make_graph([node], "test_CenterCropPad_graph", [x], [_f32("output", SPATIAL_SHAPE)], [shape])
    return _model(graph, opset)


def _build_max_roi_pool(opset: int) -> ModelProto:
    x = _f32("X", SPATIAL_SHAPE)
    rois = _f32("rois", [2, 4])
    node = helper.make_node(
        "MaxRoiPool",
        ["X", "rois"],
        ["Y"],
        name="test",
        pooled_shape=[2, 2],
        spatial_scale=1.0,
    )
    graph = helper.make_graph([node], "test_MaxRoiPool_graph", [x, rois], [_f32("Y", [2, 1, 2, 2])])
    return _model(graph, opset)


def _build_max_unpool(opset: int) -> ModelProto:
    x = _f32("X", [1, 1, 2, 2])
    indices = numpy_helper.from_array(
        np.array([[[[0, 1], [4, 5]]]], dtype=np.int64),
        "I",
    )
    node = helper.make_node(
        "MaxUnpool",
        ["X", "I"],
        ["Y"],
        name="test",
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    graph = helper.make_graph(
        [node],
        "test_MaxUnpool_graph",
        [x],
        [_f32("Y", [1, 1, 4, 4])],
        [indices],
    )
    return _model(graph, opset)


def _build_multinomial(opset: int) -> ModelProto:
    x = _f32("input", DEFAULT_VECTOR_SHAPE)
    node = helper.make_node("Multinomial", ["input"], ["output"], sample_size=2, dtype=TensorProto.INT32, name="test")
    graph = helper.make_graph([node], "test_Multinomial_graph", [x], [_i32("output", [2])])
    return _model(graph, opset)


def _build_mel_weight_matrix(opset: int) -> ModelProto:
    num = numpy_helper.from_array(np.array(8, dtype=np.int64), "num_mel_bins")
    dft = numpy_helper.from_array(np.array(16, dtype=np.int64), "dft_length")
    sr = numpy_helper.from_array(np.array(16000, dtype=np.int64), "sample_rate")
    sl = numpy_helper.from_array(np.array(0, dtype=np.int64), "stft_lower_bound")
    su = numpy_helper.from_array(np.array(8000, dtype=np.int64), "stft_upper_bound")
    node = helper.make_node(
        "MelWeightMatrix",
        ["num_mel_bins", "dft_length", "sample_rate", "stft_lower_bound", "stft_upper_bound"],
        ["output"],
        name="test",
    )
    graph = helper.make_graph(
        [node],
        "test_MelWeight_matrix_graph",
        [],
        [_f32("output", [8, 9])],
        [num, dft, sr, sl, su],
    )
    return _model(graph, opset)


def _build_stft(opset: int) -> ModelProto:
    signal = _f32("signal", [8])
    frame_step = numpy_helper.from_array(np.array(4, dtype=np.int64), "frame_step")
    frame_length = numpy_helper.from_array(np.array(4, dtype=np.int64), "frame_length")
    node = helper.make_node("STFT", ["signal", "frame_step", "frame_length"], ["output"], name="test")
    graph = helper.make_graph(
        [node],
        "test_STFT_graph",
        [signal],
        [_f32("output", [5, 2, 2])],
        [frame_step, frame_length],
    )
    return _model(graph, opset)


def _build_string_concat(opset: int) -> ModelProto:
    a = numpy_helper.from_array(np.array(["a", "b"], dtype=object), "X")
    b = numpy_helper.from_array(np.array(["c", "d"], dtype=object), "Y")
    node = helper.make_node("StringConcat", ["X", "Y"], ["Z"], name="test")
    graph = helper.make_graph(
        [node],
        "test_StringConcat_graph",
        [],
        [helper.make_tensor_value_info("Z", TensorProto.STRING, [4])],
        [a, b],
    )
    return _model(graph, opset)


def _build_string_split(opset: int) -> ModelProto:
    x = numpy_helper.from_array(np.array(["a,b", "c,d"], dtype=object), "X")
    node = helper.make_node("StringSplit", ["X"], ["Y", "Z"], name="test")
    graph = helper.make_graph(
        [node],
        "test_StringSplit_graph",
        [],
        [
            helper.make_tensor_value_info("Y", TensorProto.STRING, [2, 2]),
            helper.make_tensor_value_info("Z", TensorProto.INT64, [2, 2]),
        ],
        [x],
    )
    return _model(graph, opset)


def _build_regex_full_match(opset: int) -> ModelProto:
    x = numpy_helper.from_array(np.array(["abc", "def"], dtype=object), "X")
    node = helper.make_node("RegexFullMatch", ["X"], ["Y"], name="test", pattern="a.*")
    graph = helper.make_graph(
        [node],
        "test_RegexFullMatch_graph",
        [],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [2])],
        [x],
    )
    return _model(graph, opset)


def _build_tfidf_vectorizer(opset: int) -> ModelProto:
    x = numpy_helper.from_array(np.array(["a b", "b c"], dtype=object), "X")
    node = helper.make_node(
        "TfIdfVectorizer",
        ["X"],
        ["Y"],
        name="test",
        mode="TFIDF",
        min_gram_length=1,
        max_gram_length=1,
        max_skip_count=0,
        ngram_counts=[1, 1, 1],
        ngram_indexes=[0, 1, 2],
        pool_int64s=[0, 1, 2],
    )
    graph = helper.make_graph([node], "test_TfIdfVectorizer_graph", [], [_f32("Y", [2, 3])], [x])
    return _model(graph, opset)


def _build_dequantize_linear(opset: int) -> ModelProto:
    x = numpy_helper.from_array(np.array([1, 2], dtype=np.uint8), "x")
    scale = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "x_scale")
    zp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "x_zero_point")
    node = helper.make_node("DequantizeLinear", ["x", "x_scale", "x_zero_point"], ["y"], name="test")
    graph = helper.make_graph([node], "test_DequantizeLinear_graph", [], [_f32("y", DEFAULT_VECTOR_SHAPE)], [x, scale, zp])
    return _model(graph, opset)


def _build_conv_integer(opset: int) -> ModelProto:
    x = numpy_helper.from_array(np.arange(1, 17, dtype=np.uint8).reshape(1, 1, 4, 4), "x")
    w = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.uint8), "w")
    xz = numpy_helper.from_array(np.array(0, dtype=np.uint8), "x_zero_point")
    wz = numpy_helper.from_array(np.array(0, dtype=np.uint8), "w_zero_point")
    node = helper.make_node("ConvInteger", ["x", "w", "x_zero_point", "w_zero_point"], ["y"], name="test")
    graph = helper.make_graph([node], "test_ConvInteger_graph", [], [_i32("y", [1, 1, 4, 4])], [x, w, xz, wz])
    return _model(graph, opset)


def _build_matmul_integer(opset: int) -> ModelProto:
    a = numpy_helper.from_array(np.array([[1, 2]], dtype=np.uint8), "A")
    b = numpy_helper.from_array(np.array([[3], [4]], dtype=np.uint8), "B")
    node = helper.make_node("MatMulInteger", ["A", "B"], ["Y"], name="test")
    graph = helper.make_graph([node], "test_MatMulInteger_graph", [], [_i32("Y", [1, 1])], [a, b])
    return _model(graph, opset)


def _build_qlinear_conv(opset: int) -> ModelProto:
    x = numpy_helper.from_array(np.arange(1, 17, dtype=np.uint8).reshape(1, 1, 4, 4), "x")
    x_scale = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "x_scale")
    x_zp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "x_zero_point")
    w = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.uint8), "w")
    w_scale = numpy_helper.from_array(np.array(0.25, dtype=np.float32), "w_scale")
    w_zp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "w_zero_point")
    y_scale = numpy_helper.from_array(np.array(0.125, dtype=np.float32), "y_scale")
    y_zp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "y_zero_point")
    node = helper.make_node(
        "QLinearConv",
        ["x", "x_scale", "x_zero_point", "w", "w_scale", "w_zero_point", "y_scale", "y_zero_point"],
        ["y"],
        name="test",
    )
    graph = helper.make_graph(
        [node],
        "test_QLinearConv_graph",
        [],
        [_u8("y", [1, 1, 4, 4])],
        [x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp],
    )
    return _model(graph, opset)


def _build_qlinear_matmul(opset: int) -> ModelProto:
    a = numpy_helper.from_array(np.array([[1, 2]], dtype=np.uint8), "a")
    a_scale = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "a_scale")
    a_zp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "a_zero_point")
    b = numpy_helper.from_array(np.array([[3], [4]], dtype=np.uint8), "b")
    b_scale = numpy_helper.from_array(np.array(0.25, dtype=np.float32), "b_scale")
    b_zp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "b_zero_point")
    y_scale = numpy_helper.from_array(np.array(0.125, dtype=np.float32), "y_scale")
    y_zp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "y_zero_point")
    node = helper.make_node(
        "QLinearMatMul",
        ["a", "a_scale", "a_zero_point", "b", "b_scale", "b_zero_point", "y_scale", "y_zero_point"],
        ["y"],
        name="test",
    )
    graph = helper.make_graph(
        [node],
        "test_QLinearMatMul_graph",
        [],
        [_u8("y", [1, 1])],
        [a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp],
    )
    return _model(graph, opset)


def _build_softmax_cross_entropy(opset: int) -> ModelProto:
    return _build_negative_log_likelihood_loss(opset)


def _build_bernoulli(opset: int) -> ModelProto:
    x = _f32("input", DEFAULT_VECTOR_SHAPE)
    node = helper.make_node("Bernoulli", ["input"], ["output"], dtype=TensorProto.FLOAT, seed=1.0, name="test")
    graph = helper.make_graph([node], "test_Bernoulli_graph", [x], [_f32("output", DEFAULT_VECTOR_SHAPE)])
    return _model(graph, opset)


def _build_random(op_type: str, opset: int) -> ModelProto:
    node = helper.make_node(
        op_type,
        [],
        ["output"],
        dtype=TensorProto.FLOAT,
        shape=DEFAULT_VECTOR_SHAPE,
        name=f"test_{op_type}",
    )
    graph = helper.make_graph([node], f"test_{op_type}_graph", [], [_f32("output", DEFAULT_VECTOR_SHAPE)])
    return _model(graph, opset)


def _build_random_like(op_type: str, opset: int) -> ModelProto:
    x = _f32("input", DEFAULT_VECTOR_SHAPE)
    node = helper.make_node(op_type, ["input"], ["output"], dtype=TensorProto.FLOAT, name=f"test_{op_type}")
    graph = helper.make_graph([node], f"test_{op_type}_graph", [x], [_f32("output", DEFAULT_VECTOR_SHAPE)])
    return _model(graph, opset)


def _build_global_lp_pool(opset: int) -> ModelProto:
    x = _f32("X", SPATIAL_SHAPE)
    node = helper.make_node("GlobalLpPool", ["X"], ["Y"], p=2, name="test")
    graph = helper.make_graph([node], "test_GlobalLpPool_graph", [x], [_f32("Y", [1, 1, 1, 1])])
    return _model(graph, opset)


def _build_swish(opset: int) -> ModelProto:
    x = _f32("X", DEFAULT_VECTOR_SHAPE)
    node = helper.make_node("Swish", ["X"], ["Y"], name="test")
    graph = helper.make_graph([node], "test_Swish_graph", [x], [_f32("Y", DEFAULT_VECTOR_SHAPE)])
    return _model(graph, opset)


def _build_image_decoder(opset: int) -> ModelProto:
    stream = numpy_helper.from_array(np.array([0, 0, 0], dtype=np.uint8), "encoded_stream")
    node = helper.make_node("ImageDecoder", ["encoded_stream"], ["rgb"], pixel_format="RGB", name="test")
    graph = helper.make_graph(
        [node],
        "test_ImageDecoder_graph",
        [],
        [_u8("rgb", [1, 1, 1])],
        [stream],
    )
    return _model(graph, opset)


def _build_string_normalizer(opset: int) -> ModelProto:
    x = numpy_helper.from_array(np.array(["Hello", "World"], dtype=object), "X")
    node = helper.make_node(
        "StringNormalizer",
        ["X"],
        ["Y"],
        name="test",
        case_change_action="UPPER",
        is_case_sensitive=0,
        locale="en_US",
    )
    graph = helper.make_graph(
        [node],
        "test_StringNormalizer_graph",
        [],
        [helper.make_tensor_value_info("Y", TensorProto.STRING, [2])],
        [x],
    )
    return _model(graph, opset)


def _build_sequence_erase(opset: int) -> ModelProto:
    t = numpy_helper.from_array(np.zeros((1, 2), dtype=np.float32), "t0")
    seq = helper.make_node("SequenceConstruct", ["t0"], ["seq"], name="mk")
    pos = numpy_helper.from_array(np.array(0, dtype=np.int64), "position")
    erase = helper.make_node("SequenceErase", ["seq", "position"], ["output_sequence"], name="test")
    graph = helper.make_graph(
        [seq, erase],
        "test_SequenceErase_graph",
        [],
        [helper.make_value_info("output_sequence", SEQ_FLOAT)],
        [t, pos],
    )
    return _model(graph, opset)


def _build_sequence_insert(opset: int) -> ModelProto:
    t0 = numpy_helper.from_array(np.zeros((1, 2), dtype=np.float32), "t0")
    t1 = numpy_helper.from_array(np.ones((1, 2), dtype=np.float32), "t1")
    seq = helper.make_node("SequenceConstruct", ["t0"], ["seq"], name="mk")
    ins = helper.make_node("SequenceInsert", ["seq", "t1"], ["output_sequence"], name="test")
    graph = helper.make_graph(
        [seq, ins],
        "test_SequenceInsert_graph",
        [],
        [helper.make_value_info("output_sequence", SEQ_FLOAT)],
        [t0, t1],
    )
    return _model(graph, opset)


def _build_sequence_length(opset: int) -> ModelProto:
    t = numpy_helper.from_array(np.zeros((1, 2), dtype=np.float32), "t0")
    seq = helper.make_node("SequenceConstruct", ["t0"], ["seq"], name="mk")
    ln = helper.make_node("SequenceLength", ["seq"], ["length"], name="test")
    graph = helper.make_graph(
        [seq, ln],
        "test_SequenceLength_graph",
        [],
        [_i64("length", [])],
        [t],
    )
    return _model(graph, opset)


CUSTOM_BUILDERS: dict[str, Callable[[int], ModelProto]] = {
    "AffineGrid": _build_affine_grid,
    "Attention": _build_attention,
    "BatchNormalization": _build_batch_normalization,
    "Bernoulli": _build_bernoulli,
    "BitShift": _build_bitshift,
    "BitwiseAnd": lambda o: _build_bitwise("BitwiseAnd", "BitwiseAnd", o),
    "BitwiseNot": lambda o: _build_bitwise("BitwiseNot", "BitwiseNot", o),
    "BitwiseOr": lambda o: _build_bitwise("BitwiseOr", "BitwiseOr", o),
    "BitwiseXor": lambda o: _build_bitwise("BitwiseXor", "BitwiseXor", o),
    "BlackmanWindow": lambda o: _build_window("BlackmanWindow", o),
    "HammingWindow": lambda o: _build_window("HammingWindow", o),
    "HannWindow": lambda o: _build_window("HannWindow", o),
    "CenterCropPad": _build_center_crop_pad,
    "Col2Im": _build_col2im,
    "Compress": _build_compress,
    "ConcatFromSequence": _build_concat_from_sequence,
    "ConstantOfShape": _build_constant_of_shape,
    "ConvInteger": _build_conv_integer,
    "CumProd": lambda o: _build_cum_axis("CumProd", o),
    "CumSum": lambda o: _build_cum_axis("CumSum", o),
    "DequantizeLinear": _build_dequantize_linear,
    "Det": _build_det,
    "DynamicQuantizeLinear": _build_dynamic_quantize_linear,
    "Einsum": _build_einsum,
    "GRU": _build_gru,
    "GlobalLpPool": _build_global_lp_pool,
    "GridSample": _build_grid_sample,
    "If": _build_if,
    "ImageDecoder": _build_image_decoder,
    "LSTM": _build_lstm,
    "Loop": _build_loop,
    "MatMulInteger": _build_matmul_integer,
    "MaxRoiPool": _build_max_roi_pool,
    "MaxUnpool": _build_max_unpool,
    "MelWeightMatrix": _build_mel_weight_matrix,
    "Multinomial": _build_multinomial,
    "NegativeLogLikelihoodLoss": _build_negative_log_likelihood_loss,
    "NonMaxSuppression": _build_non_max_suppression,
    "OneHot": _build_one_hot,
    "Optional": _build_optional,
    "QLinearConv": _build_qlinear_conv,
    "QLinearMatMul": _build_qlinear_matmul,
    "QuantizeLinear": _build_quantize_linear,
    "Range": _build_range,
    "RandomNormal": lambda o: _build_random("RandomNormal", o),
    "RandomNormalLike": lambda o: _build_random_like("RandomNormalLike", o),
    "RandomUniform": lambda o: _build_random("RandomUniform", o),
    "RandomUniformLike": lambda o: _build_random_like("RandomUniformLike", o),
    "RegexFullMatch": _build_regex_full_match,
    "Resize": _build_resize,
    "ReverseSequence": _build_reverse_sequence,
    "RoiAlign": _build_roi_align,
    "RNN": _build_rnn,
    "RotaryEmbedding": _build_rotary_embedding,
    "Scan": _build_scan,
    "SequenceAt": _build_sequence_at,
    "SequenceConstruct": _build_sequence_construct,
    "SequenceEmpty": _build_sequence_empty,
    "SequenceErase": _build_sequence_erase,
    "SequenceInsert": _build_sequence_insert,
    "SequenceLength": _build_sequence_length,
    "SoftmaxCrossEntropyLoss": _build_softmax_cross_entropy,
    "Split": _build_split,
    "SplitToSequence": _build_split_to_sequence,
    "STFT": _build_stft,
    "StringConcat": _build_string_concat,
    "StringNormalizer": _build_string_normalizer,
    "StringSplit": _build_string_split,
    "Swish": _build_swish,
    "TensorScatter": _build_tensor_scatter,
    "TfIdfVectorizer": _build_tfidf_vectorizer,
    "Tile": _build_tile,
    "TopK": _build_topk,
}
