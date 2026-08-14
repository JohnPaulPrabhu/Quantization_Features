#!/usr/bin/env python3
import onnx
from onnx import helper, TensorProto

# Exactly 75 GiB per FP32 tensor.
H = 131072
W = 153600
MODEL_NAME = "conv_225gib_total_with_im2col.onnx"

shape = [1, 1, H, W]

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)

weight = helper.make_tensor(
    name="W",
    data_type=TensorProto.FLOAT,
    dims=[1, 1, 1, 1],
    vals=[1.0],
)

conv = helper.make_node(
    "Conv",
    inputs=["X", "W"],
    outputs=["Y"],
    name="Conv_0",
)

graph = helper.make_graph(
    [conv],
    "single_conv_im2col_ram_test",
    [X],
    [Y],
    initializer=[weight],
)

model = helper.make_model(
    graph,
    producer_name="conv-im2col-ram-test",
    opset_imports=[helper.make_opsetid("", 13)],
)
model.ir_version = 8

onnx.checker.check_model(model)
onnx.save(model, MODEL_NAME)

bytes_per_tensor = H * W * 4
print("Saved:", MODEL_NAME)
print(f"Input:     {bytes_per_tensor / 1024**3:.2f} GiB")
print(f"Unfolded:  {bytes_per_tensor / 1024**3:.2f} GiB  (1x1 Conv)")
print(f"Output:    {bytes_per_tensor / 1024**3:.2f} GiB")
print(f"Total:     {3 * bytes_per_tensor / 1024**3:.2f} GiB")
