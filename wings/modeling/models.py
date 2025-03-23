import torch.nn as nn


class ResnetPreTrained(nn.Module):
    def __init__(self, pretrained_model, pretrained_weights):
        super(ResnetPreTrained, self).__init__()

        self.pretrained_model = pretrained_model(weights=pretrained_weights)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(self.pretrained_model.fc.in_features, 38)
        self.pretrained_model.fc = self.linear

    def forward(self, x):
        x = self.pretrained_model(x)
        return x


class TransformerPreTrained(nn.Module):
    def __init__(self, pretrained_model, pretrained_weights):
        super(TransformerPreTrained, self).__init__()
        self.pretrained_model = pretrained_model(weights=pretrained_weights)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(self.pretrained_model.heads.head.in_features, 38)
        self.pretrained_model.heads.head = self.linear


    def forward(self, x):
        x = self.pretrained_model(x)
        return x
