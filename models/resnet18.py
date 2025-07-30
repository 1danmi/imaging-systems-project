import torch
import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        if in_channels != 3:
            conv1 = self.model.conv1
            new_conv = nn.Conv2d(
                in_channels,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.copy_(conv1.weight.sum(dim=1, keepdim=True))

            self.model.conv1 = new_conv

        # Freeze the pretrained model parameters
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        # Add one trainable layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
