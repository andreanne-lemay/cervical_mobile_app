import torch
from monai.networks.nets.densenet import Densenet121
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.utils.bundled_inputs
import torch.utils.mobile_optimizer
import torch.backends._nnapi.prepare

## Download custom model weights
# model_path = "path/to/model.pth"

class DenseNet(Densenet121):
    def __init__(self, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.class_layers(x)
        return self.dequant(x)

    def fuse_model(self):
        for name, m in self.named_modules():
            if name == 'features':
                torch.quantization.fuse_modules(m, ['norm0', 'relu0'], inplace=True)
            if name.endswith('layers') and name != "class_layers":
                torch.quantization.fuse_modules(m, ['norm1',  'relu1'], inplace=True)
                torch.quantization.fuse_modules(m, ['norm2',  'relu2'], inplace=True)
            if name[:-1].endswith('transition'):
                torch.quantization.fuse_modules(m, ['norm', 'relu'], inplace=True)

densenet = DenseNet
model = densenet(spatial_dims=2,
                 in_channels=3,
                 out_channels=3,
                 dropout_prob=float(0.1),
                 pretrained=True)

## Load custom path into model
# device = "cpu"
# model.load_state_dict(torch.load(model_path, map_location=device))

# Model quantization
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model.fuse_model()
model = torch.quantization.prepare(model, inplace=True)
model = torch.quantization.convert(model, inplace=True)

torchscript_model = torch.jit.script(model)
optimize_for_mobile(torchscript_model)._save_for_lite_interpreter("dummy_model_quantized.ptl")
