import torchvision
import torch
import monai
from torch.utils.mobile_optimizer import optimize_for_mobile
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
import matplotlib.image as mpimg

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    Transpose,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    NormalizeIntensity,
    Resize,
    AddChannel,
    CenterSpatialCrop,
    RandShiftIntensity,
    RandStdShiftIntensity,
    RandScaleIntensity,
    RandGaussianNoise,
    AddChannel,
    ToTensor,
    RepeatChannel
)

import dropout_resnet
from pytorch_quantization import quant_modules
# quant_modules.initialize()
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
# model.eval()
# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model, example)
# optimized_traced_model = optimize_for_mobile(traced_script_module)
# torchscript_model = torch.jit.script(model)
# optimize_for_mobile(torchscript_model)._save_for_lite_interpreter("model_custom.pt")


# model_path = "/home/andreanne/Documents/dataset/cervix/model_36.pth"
#
# densenet = getattr(monai.networks.nets, "densenet121")
# model = densenet(spatial_dims=2,
#                  in_channels=3,
#                  out_channels=3,
#                  dropout_prob=float(0.1),
#                  pretrained=True)

model_path = "/home/andreanne/Documents/dataset/cervix/model_36.pth"
# model_path = "/home/andreanne/Documents/dataset/cervix/model_61.pth"
# #
# resnet = getattr(dropout_resnet, 'resnet50')
# model = resnet()
# model.fc = torch.nn.Linear(model.fc.in_features, 3)

# quant_desc_input = QuantDescriptor(calib_method='histogram')
# quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
# quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
# quant_nn.QuantB.set_default_quant_desc_input(quant_desc_input)

# model = torchvision.models.resnet18(pretrained=True)

densenet = getattr(monai.networks.nets, "densenet121")
model = densenet(spatial_dims=2,
                 in_channels=3,
                 out_channels=3,
                 dropout_prob=float(0.1),
                 pretrained=True)
#
device = "cpu"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

import os
import pandas as pd
base_path = "/run/user/1000/gvfs/sftp:host=glacier.nmr.mgh.harvard.edu,user=ai347/home/ai347/mnt/2015P002510/cervix_dataset/full_dataset"
img_path = os.path.join(base_path, "I312374_C2.jpg")

def load_transforms(transforms_dict: dict):
    """Converts dictionary into python transforms"""
    transform_lst = []
    for tr in transforms_dict:
        transform_lst.append(globals()[tr](**transforms_dict[tr]))
#     print(transform_lst)
    return Compose(transform_lst)


val_transforms = {
    "Transpose": {"indices": [2, 0, 1]},
    "ScaleIntensity": {},
    "Resize": {"spatial_size": [256, 256]},
}

tr = load_transforms(val_transforms)
img = mpimg.imread(img_path)
df = pd.read_csv("/run/user/1000/gvfs/sftp:host=glacier.nmr.mgh.harvard.edu,user=ai347/home/ai347/mnt/2015P002510/cervix_dataset/full_dataset_df_5labels.csv")
idx_data = df[df.MASKED_IMG_ID == img_path.split("/")[-1]]
img = img[int(idx_data['y1']): int(idx_data['y2']), int(idx_data['x1']): int(idx_data['x2']), :]
output_img = tr(img).swapaxes(0,2).swapaxes(0,1)
mpimg.imsave('I312374_C2.jpg', tr(img).swapaxes(0,2).swapaxes(0,1))
im = mpimg.imread('I312374_C2.jpg')
output = torch.nn.Softmax(dim=1)(model(ToTensor()(tr(im)[None, ] * 0 + 0.5)))
print(output)

# model.to(device)
# model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
# model_static_quantized = torch.quantization.prepare(model, inplace=True)
# model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=True)
#
# example = torch.rand(1, 3, 256, 256)
# traced_script_module = torch.jit.trace(model_static_quantized, example)
# optimized_traced_model = optimize_for_mobile(traced_script_module)
# optimized_traced_model._save_for_lite_interpreter("model_36.ptl")
# torchscript_model = torch.jit.script(model)
# # torchscript_model_optimized = optimize_for_mobile(torchscript_model)
# # torch.jit.save(torchscript_model_optimized, "model.pt")
# optimize_for_mobile(torchscript_model)._save_for_lite_interpreter("model_36.ptl")
#
# print()

