import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision.models as models




class ResNet(nn.Module):
    def __init__(self, num_class:int, model_name:str, trainable_layers = None, wgth = None):
        super().__init__()
        available_models = {
            'ResNet50' : models.resnet50,
            'ResNet34' : models.resnet34,
            'ResNet18' : models.resnet18,
            'ResNet101' : models.resnet101
        }
        if model_name not in available_models.keys():
            raise ValueError(f' ** ERROR ** Chose model name between : {available_models.keys()}')
        else :
            self.model = available_models[model_name](weights="IMAGENET1K_V1")

        # modifying fc layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_class)

        if wgth :
            self.model.load_statedict(torch.load(wgth))
            print(f' ** Weights loadded from {wgth} **')

    # TODO : add verify input channels == input channel model !!

    def forward(self,x):
        return self.model(x)


class EfficienceNet(nn.Module):
    def __init__(self, num_class : int, model_name:str, wgth = None):
        super().__init__()
        available_models = {
            'effb0':models.efficientnet_b0,
            'effv2s': models.efficientnet_v2_s
        }
        if model_name not in available_models.keys():
            raise ValueError(f' ** ERROR ** Chose model name between : {available_models.keys()}')
        else :
            self.model = available_models[model_name](weights="IMAGENET1K_V1")

        # modifying classifier layer
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_class)

        if wgth:
            self.model.load_state_dict(torch.load(wgth))
            print(f' ** Weights loadded from {wgth} **')

    def forward(self, x):
        return self.model(x)