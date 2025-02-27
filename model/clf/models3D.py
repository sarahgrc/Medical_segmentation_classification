""" Implementation of 3D pretrained classification models"""
import torch
import torch.nn as nn
import torchvision.models.video as models


class Clf3D(nn.Module):
    def __init__(self, model_name:str, num_class : int, train_layers = ['layer3', 'layer4', 'fc'] ):
        """
        Implementation of 3D pretrained classification models
        :param model_name: name of the model to implement
        :param num_class: number of classes
        :param train_layers: List of layers to train
        """
        super().__init__()
        available_models = {'r3d_18':models.r3d_18,
                            'mc3_18': models.mc3_18,
                            'r2plus1d_18': models.r2plus1d_18}
        if model_name not in available_models :
            raise ValueError(f'Wrong model name, chose from : {available_models.keys()}')
        else :
            self.model = available_models[model_name](pretrained = True)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_class)
        self._freeze_layers(train_layers)

    def _freeze_layers(self, train_layers):
        """ Unfreeze trainable layers"""
        for name, param in self.model.named_parameters() :
            layer_name = name.split('.')[0] # extract layers names
            param.requires_grad = layer_name in train_layers

    def forward(self, x):
        return self.model(x)





