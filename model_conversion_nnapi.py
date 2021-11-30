import torch
import monai
from densenet import Densenet121
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.utils.bundled_inputs
import torch.utils.mobile_optimizer
import torch.backends._nnapi.prepare
import torchvision
import dropout_resnet

# Download custom model weights
# model_path = "path/to/model.pth"

# Densenet121
densenet = Densenet121
# densenet = getattr(monai.networks.nets, 'densenet121')
model = densenet(spatial_dims=2,
                  in_channels=3,
                  out_channels=3,
                  dropout_prob=float(0.1),
                  pretrained=True)

# Resnet50
# model = getattr(dropout_resnet, "resnet50")(pretrained=True)
# model.fc = torch.nn.Linear(model.fc.in_features, 3)

# Load custom path into model
# device = "cpu"
# model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

output = model(torch.ones((1, 3, 256, 256)))

for name, m in model.named_modules():
    if hasattr(m, 'inplace'):
        m.inplace = False

input_float = torch.zeros(1, 3, 256, 256)
input_tensor = input_float
input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
input_tensor.nnapi_nhwc = True
with torch.no_grad():
    traced = torch.jit.trace(model, input_tensor)

nnapi_model = torch.backends._nnapi.prepare.convert_model_to_nnapi(traced, input_tensor)

class BundleWrapper(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, arg):
        return self.mod(arg)

nnapi_model = torch.jit.script(BundleWrapper(nnapi_model))
torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
    nnapi_model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])

nnapi_model._save_for_lite_interpreter("dummy_model_densenet_nnapi.ptl")
optimize_for_mobile(traced)._save_for_lite_interpreter("dummy_model_densenet.ptl")
