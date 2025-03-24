import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return (x1, x2, x3, x4, x5)


def spatial_average(in_feat, keepdim=True):
    return in_feat.mean([2, 3], keepdim=keepdim)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


class MyLPIPS(nn.Module):
    def __init__(self, weight_path):
        super().__init__()
        self.alex = AlexNet()

        state_dict = torch.load(weight_path, weights_only=True)
        self.alex.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, img0, img1):
        z0 = self.alex(img0)
        z1 = self.alex(img1)

        dist = []
        for i in range(len(z0)):
            feat0 = normalize_tensor(z0[i])
            feat1 = normalize_tensor(z1[i])
            dist.append(spatial_average(torch.sum((feat0 - feat1) ** 2, dim=1, keepdim=True)))

        val = 0
        for i in range(len(dist)):
            val += dist[i]

        return val.squeeze(-1).squeeze(-1).squeeze(-1)


if __name__ == "__main__":
    lpips = MyLPIPS("alexnet.pth")

    img0 = torch.randn(4, 1, 256, 256)
    img1 = torch.randn(4, 1, 256, 256)

    res = lpips(img0, img1)
    print(res)
