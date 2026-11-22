import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Sequential):
    def __init__(self, width):
        super().__init__(nn.Flatten(), nn.Linear(784, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 10))


class BasicBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.downsample = None
        if stride != 1 or incoming != outgoing:
            self.downsample = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 1, stride, bias=False),
                nn.BatchNorm2d(outgoing))

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + residual)


class CifarResNet(nn.Module):
    def __init__(self, depth, classes):
        super().__init__()
        blocks = (depth - 2) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.stage(16, 16, blocks, 1)
        self.layer2 = self.stage(16, 32, blocks, 2)
        self.layer3 = self.stage(32, 64, blocks, 2)
        self.fc = nn.Linear(64, classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def stage(incoming, outgoing, count, stride):
        return nn.Sequential(BasicBlock(incoming, outgoing, stride),
                             *(BasicBlock(outgoing, outgoing) for _ in range(count - 1)))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def make_model(name, classes):
    if name in ("mlp1200", "mlp800") and classes == 10:
        return MLP(int(name[3:]))
    if name == "mlp1200d" and classes == 10:
        flatten, fc1, relu1, fc2, relu2, fc3 = MLP(1200)
        return nn.Sequential(flatten, nn.Dropout(0.2), fc1, relu1, nn.Dropout(0.5),
                             fc2, relu2, nn.Dropout(0.5), fc3)
    if name in ("resnet20", "resnet56"):
        return CifarResNet(int(name[6:]), classes)
    raise ValueError(f"Unsupported model/classes: {name}/{classes}")


@torch.no_grad()
def complexity(model, shape):
    macs = 0

    def count(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        else:
            macs += output.numel() * module.in_features

    handles = [m.register_forward_hook(count) for m in model.modules()
               if isinstance(m, (nn.Linear, nn.Conv2d))]
    training = model.training
    try:
        model.eval()(torch.zeros(1, *shape, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "macs": macs, "flops": 2 * macs,
            "flops_convention": "2 per MAC; Conv/Linear only; one image"}
