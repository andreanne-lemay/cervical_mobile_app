import torch
import monai
from torch.utils.mobile_optimizer import optimize_for_mobile

## Download custom model weights
# model_path = "path/to/model.pth"

densenet = getattr(monai.networks.nets, "densenet121")
model = densenet(spatial_dims=2,
                 in_channels=3,
                 out_channels=3,
                 dropout_prob=float(0.1),
                 pretrained=True)

## Load custom path into model
device = "cpu"
# model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

torchscript_model = torch.jit.script(model)
optimize_for_mobile(torchscript_model)._save_for_lite_interpreter("dummy_model.ptl")
