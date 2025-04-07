import torch
import torch.nn as nn


class CustomPreTrained(nn.Module):
    def __init__(self, pretrained_model, pretrained_weights):
        super(CustomPreTrained, self).__init__()
        self.pretrained_model = pretrained_model(weights=pretrained_weights)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained_model(x)
        return x


class ResnetPreTrained(CustomPreTrained):
    def __init__(self, pretrained_model, pretrained_weights):
        super(ResnetPreTrained, self).__init__(pretrained_model, pretrained_weights)
        self.linear = nn.Linear(self.pretrained_model.fc.in_features, 38)
        self.pretrained_model.fc = self.linear


class TransformerPreTrained(CustomPreTrained):
    def __init__(self, pretrained_model, pretrained_weights):
        super(TransformerPreTrained, self).__init__(pretrained_model, pretrained_weights)
        self.linear = nn.Linear(self.pretrained_model.heads.head.in_features, 38)
        self.pretrained_model.heads.head = self.linear


class UnetPreTrained(CustomPreTrained):
    def __init__(self, pretrained_model, pretrained_weights):
        super(UnetPreTrained, self).__init__(pretrained_model, pretrained_weights)
