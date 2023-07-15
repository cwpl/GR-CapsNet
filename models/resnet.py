import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *
from utils import weights_init,squash


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, planes, num_caps, caps_size, depth, cfg_data, mode):
        super(ResNet, self).__init__()
        self.in_planes = planes
        print("planes", planes)
        channels, classes = cfg_data['channels'], cfg_data['classes']
        self.num_caps = num_caps
        self.caps_size = caps_size

        self.depth = depth

        self.conv1 = nn.Conv2d(channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, planes, num_blocks[0], stride=1)  # 3
        self.layer2 = self._make_layer(block, 2 * planes, num_blocks[1], stride=2)  # 3
        self.layer3 = self._make_layer(block, 4 * planes, num_blocks[2], stride=2)  # 3

        self.mode = mode

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for d in range(1, depth):
            stride = 2 if d == 1 else 1
            if self.mode == 'EM':
                self.conv_layers.append(
                    EmRouting2d(num_caps, num_caps, caps_size, kernel_size=3, stride=stride, padding=1))
                self.norm_layers.append(nn.BatchNorm2d(caps_size * num_caps))
            elif self.mode == 'GR':
                self.conv_layers.append(
                    GlobalRouting(num_caps, num_caps, caps_size, caps_size, kernel_size=3, stride=stride, padding=1,
                                  pose_out=True))
                self.norm_layers.append(nn.BatchNorm2d(caps_size * num_caps))
            else:
                break

        final_shape = 8 if depth == 1 else 4

        if self.mode == 'EM':
            self.conv_a = nn.Conv2d(4 * planes, num_caps, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_pose = nn.Conv2d(4 * planes, num_caps * caps_size, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_a = nn.BatchNorm2d(num_caps)
            self.bn_pose = nn.BatchNorm2d(num_caps * caps_size)
            self.fc = EmRouting2d(num_caps, classes, caps_size, kernel_size=final_shape, padding=0)

        elif self.mode == 'GR':
            self.conv_a = nn.Conv2d(4 * planes, num_caps, kernel_size=3, stride=1, padding=1, bias=False)  # planes16
            self.conv_pose = nn.Conv2d(4 * planes, num_caps * caps_size, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_a = nn.BatchNorm2d(num_caps)
            self.bn_pose = nn.BatchNorm2d(num_caps * caps_size)
            self.fc = GlobalRouting(num_caps, classes, caps_size, 1, kernel_size=final_shape, padding=0, pose_out=False)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.mode == 'EM':
            a, pose = self.conv_a(out), self.conv_pose(out)
            a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

            for m, bn in zip(self.conv_layers, self.norm_layers):
                a, pose = m(a, pose)
                pose = bn(pose)

            a, _ = self.fc(a, pose)
            out = a.view(a.size(0), -1)
            out = out / out.sum(dim=1, keepdim=True)
            out = out.log()

        elif self.mode == 'GR':
            a, pose = self.conv_a(out), self.conv_pose(out)
            a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)
            for m, bn in zip(self.conv_layers, self.norm_layers):
                a, pose = m(a, pose)
                pose = bn(pose)
            a, _ = self.fc(a, pose)
            out = a.view(a.size(0), -1)
            out = out.log()

        return out

    def forward_activations(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.mode == 'EM':
            a = torch.sigmoid(self.bn_a(self.conv_a(out)))

        elif self.mode == 'GR':
            a = torch.sigmoid(self.bn_a(self.conv_a(out)))

        else:
            raise NotImplementedError

        return a

def resnet20(planes, cfg_data, num_caps, caps_size, depth, mode):
    return ResNet(BasicBlock, [3, 3, 3], planes, num_caps, caps_size, depth, cfg_data, mode)


