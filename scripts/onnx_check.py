# import onnx
#
# # 加载原始模型
# model = onnx.load("/data/16t/wxh/LitDetect/lightning_logs/version_79/ckpts/faster_rcnn-epoch=167-map_50=0.82812.onnx")
#
# # 遍历图中的所有节点，检查是否有空输入
# for node in model.graph.node:
#     for i, inp in enumerate(node.input):
#         if inp == "":
#             print(f"Found empty input at node: {node.name} ({node.op_type}), input index: {i}")
#             # 替换为空占位符或者根据上下文合理命名
#             node.input[i] = f"__empty_input_{node.name}_{i}__"
#
# # 可选：重新初始化模型的元数据
# onnx.checker.check_model(model)
#
# # 保存修复后的模型
# onnx.save(model, "repaired_model.onnx")
# print("Model saved as repaired_model.onnx")

import onnxruntime
import numpy as np

# # ONNX model
# onnx_model = onnxruntime.InferenceSession('simplified_model.onnx')
# ipt = np.random.rand(3, 512, 512).astype(np.float32)
# output = onnx_model.run(None, {'input': ipt})
# print(output[0].shape)
import onnx

# 加载模型
model = onnx.load('/data/16t/wxh/LitDetect/lightning_logs/version_46/ckpts/faster_rcnn-epoch=014-map_50=0.67197_dynamic.onnx')

# 运行形状推断
# from onnx import shape_inference
# inferred_model = shape_inference.infer_shapes(model)
#
# # 保存固定形状的模型
# onnx.save(inferred_model, "model_fixed.onnx")
# 获取输入信息
# for ipt in model.graph.input:
#     print("Input name:", ipt.name)
#     print("Input shape:", [dim.dim_value for dim in ipt.type.tensor_type.shape.dim])
#
# # 获取输出信息
# for output in model.graph.output:
#     print("Output name:", output.name)
#     print("Output shape:", [dim.dim_value for dim in output.type.tensor_type.shape.dim])
for node in model.graph.node:
    if node.op_type == "ScatterND":
        print(f"ScatterND node found: {node.name}")
# from onnx import version, IR_VERSION
# from onnx.defs import onnx_opset_version
#
# print(f"onnx.__version__ = {version.version!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")
