import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    squared_norm = (x**2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)


class DigitCapsLayer(nn.Module):
    def __init__(
        self,
        num_capsules: int,
        num_routes: int,
        in_dim: int,
        out_dim: int,
        num_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.num_iterations = num_iterations
        self.W = nn.Parameter(0.01 * torch.randn(num_routes, num_capsules, in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_routes, in_dim)
        batch_size = x.size(0)
        # (batch, num_routes, num_capsules, out_dim)
        u_hat = torch.einsum("bri,rijo->brjo", x, self.W)
        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, device=x.device)
        for i in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            v_j = squash(s_j)
            if i < self.num_iterations - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)
        return v_j


class CapsNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.primary_caps = nn.Conv2d(256, 32 * 8, kernel_size=1)

        # input image assumed to be 224x224 -> output of conv layers is 14x14
        self.num_routes = 32 * 14 * 14
        self.digit_caps = DigitCapsLayer(num_classes, self.num_routes, 8, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.primary_caps(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 32 * 14 * 14, 8)
        digit_caps = self.digit_caps(x)
        logits = digit_caps.norm(dim=-1)
        return logits
