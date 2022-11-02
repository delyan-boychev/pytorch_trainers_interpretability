# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.1 for the model architecture on ImageNet                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #
import torch.nn as  nn
import torch.nn.functional as F

from ..trainers.tools import FakeReLU, SequentialWithArgs


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, shortcut=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.shortcut = shortcut
        self.stride = stride
        
    def forward(self, x, fake_relu=False):
        identity = x.clone()
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out += identity
        if fake_relu:
            return FakeReLU.apply(out)
        out = F.relu(out)
        return out
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, shortcut=None, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut
        self.stride = stride
        
    def forward(self, x):
        identity = x.clone()
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out += identity
        out = F.relu(out)
        
        return out
        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, stride=1)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        self.linear = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x, fake_relu=False, with_latent=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.reshape(out.shape[0], -1)
        outf = self.linear(out)
        if with_latent is True:
            return outf, out
        return outf
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        shortcut = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, shortcut=shortcut, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return SequentialWithArgs(*layers)


def ResNet18(num_classes, channels=3):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, channels)

def ResNet34(num_classes, channels=3):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, channels)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)