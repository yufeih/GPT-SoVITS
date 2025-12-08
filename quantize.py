import onnx
from onnxconverter_common import float16
from onnxconverter_common import auto_convert_mixed_precision
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'D:/gptsovitsmodels/g2pw/g2pW.onnx'
model_quant = 'D:/gptsovitsmodels/g2pw/g2pW.quant.onnx'
#quantize_dynamic(model_fp32, model_quant)

g2pw_node_block_list = [
    "Cast_2",
    "Cast_20",
    "Cast_90",
    "Cast_114",
    "Cast_184",
    "Cast_208",
    "Cast_278",
    "Cast_302",
    "Cast_372",
    "Cast_396",
    "Cast_466",
    "Cast_490",
    "Cast_560",
    "Cast_584",
    "Cast_654",
    "Cast_678",
    "Cast_748",
    "Cast_772",
    "Cast_842",
    "Cast_866",
    "Cast_936",
    "Cast_960",
    "Cast_1030",
    "Cast_1054",
    "Cast_1124",
    "Cast_1148",
    "Cast_1160",
    "Cast_1164",
    "Cast_1165",
]

model = onnx.load("D:/gptsovitsmodels/g2pw/g2pW.onnx")
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True, node_block_list=g2pw_node_block_list)
onnx.save(model_fp16, "D:/gptsovitsmodels/g2pw/g2pW_fp16.onnx")