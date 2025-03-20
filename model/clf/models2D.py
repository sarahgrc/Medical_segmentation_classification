import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision.models as models




class ResNet(nn.Module):
    def __init__(self, num_class:int, model_name:str, trainable_layers = None):
        """
        Fine-tuned ResNet
        :param num_class (int): Number of classes
        :param model_name (str) : name of the resnet model to implement
        :param trainable_layers (List): list of layers to fine tune
        """
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

        # freeze -- unfreeze trainable layers
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in trainable_layers):
                param.requires_grad = True
            else :
                param.requires_grad = False


    def forward(self,x):
        return self.model(x)

    def load_wgts(self, wghts_path : str):
        """
        Load trained weights
        :param wghts_path (str) : path of the finetuned weights
        :return: loadded model
        """
        return self.model.load_state_dict(torch.load(wghts_path))


class EfficienceNet(nn.Module):
    def __init__(self, num_class : int, model_name:str, trainable_layers: list):
        """
        EfficientNet model for fine tuning
        :param num_class (int) : number of classes for the classification
        :param model_name (str) : name of the efficient net to implement
        :param trainable_layers (List(str)) :
        """
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

        for name, param in self.model.named_parameters :
            if any(layer in name for layer in trainable_layers) :
                param.requires_grad = True
            else :
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def load_wgts(self, wghts_path : str):
        """
        Load trained weights
        :param wghts_path (str) : path of the finetuned weights
        :return: loaded model
        """
        return self.model.load_state_dict(torch.load(wghts_path))