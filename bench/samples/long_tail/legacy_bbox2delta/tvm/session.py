# Copyright (c) OpenComputeLab. All Rights Reserved.
import onnx
import onnxruntime
import time
from gen_data import gen_np_args


onnx_model = onnx.load_model("./legacy_bbox2delta.onnx")
session = onnxruntime.InferenceSession(onnx_model.SerializeToString())

x, y = gen_np_args(3000)
input = {session.get_inputs()[0].name:x, session.get_inputs()[1].name:y}

time_start = time.time()
for i in range(1000):
    output = session.run(None, input)
time_end = time.time()
print("onnxruntime time cost: ", time_end - time_start)