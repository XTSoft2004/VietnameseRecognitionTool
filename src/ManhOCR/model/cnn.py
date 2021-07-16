from torch import nn

from ..model.efficientnet import efficient_net
from ..model.repvgg import create_RepVGG_A0


class CNN(nn.Module):
    def __init__(self, model_type, deploy=False):
        super().__init__()
        self.model_type = model_type
        self.deploy = deploy

        if model_type == "eff":
            self.model = efficient_net(image_size=None)
        elif model_type == "repvgg":
            self.model = create_RepVGG_A0(deploy=deploy)
        else:
            raise ValueError("Model type not existed")

    def forward(self, x):
        if self.model_type == "repvgg" and self.deploy:
            for module in self.model.modules():
                if hasattr(module, "switch_to_deploy"):
                    module.switch_to_deploy()

        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != "last_conv_1x1":
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
