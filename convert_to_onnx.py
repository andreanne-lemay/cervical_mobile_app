import onnxruntime
import onnxruntime.tools.convert_onnx_models_to_ort
import monai
import torch
import os

densenet = getattr(monai.networks.nets, 'densenet121')
model = densenet(spatial_dims=2,
                 in_channels=3,
                 out_channels=3,
                 dropout_prob=float(0.1),
                 pretrained=True)
device = 'cpu'
dummy_input = torch.randn(1, 3, 96, 96, device=device)
model.eval()
dynamic_axes = {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width'}

torch.onnx.export(model, dummy_input, "dummy_model.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': dynamic_axes, 'output': dynamic_axes})

# nnapi
# os.system("python -m onnxruntime.tools.convert_onnx_models_to_ort . --use_nnapi --optimization_level basic")
os.system("python -m onnxruntime.tools.convert_onnx_models_to_ort . --optimization_level basic")
