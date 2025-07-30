import torch
import torch.nn as nn


def squash(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    scale = (norm**2) / (1 + norm**2)
    return scale * tensor / (norm + 1e-8)


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels: int, num_capsules: int, capsule_dim: int, kernel_size: int, stride: int):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        b, _, h, w = out.shape
        out = out.view(b, self.num_capsules, self.capsule_dim, h * w)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(b, -1, self.capsule_dim)
        return squash(out)


class DigitCapsules(nn.Module):
    def __init__(self, num_capsules: int, num_routes: int, in_dim: int, out_dim: int, num_iterations: int = 3):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_iterations = num_iterations
        self.W = nn.Parameter(0.01 * torch.randn(1, num_routes, num_capsules, out_dim, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x[:, :, None, :, None]
        u_hat = torch.matmul(self.W, x).squeeze(-1)
        b = torch.zeros(batch_size, self.num_routes, self.num_capsules, device=x.device)
        v = None
        for i in range(self.num_iterations):
            c = torch.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s)
            if i < self.num_iterations - 1:
                b = b + (u_hat * v[:, None, :, :]).sum(dim=-1)
        return v


class CapsNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.primary_caps = PrimaryCapsules(128, num_capsules=8, capsule_dim=16, kernel_size=3, stride=1)
        self.digit_caps = DigitCapsules(num_capsules=num_classes, num_routes=1152, in_dim=16, out_dim=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        return torch.norm(x, dim=-1)
