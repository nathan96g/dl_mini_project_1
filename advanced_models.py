from torch import nn
from torch.nn import functional as F

################################################
#                    LeNet                     #
################################################

class LeNet(nn.Module):

    def __init__(self, hidden_units):
        super().__init__()
        self.hidden = hidden_units
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(6)
        self.MaxPool2d1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 32, kernel_size=5,padding=0)
        self.bn2   = nn.BatchNorm2d(32)
        self.MaxPool2d2 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(32, hidden_units)
        self.bn3 = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

        self.batch_normalization = True

    def forward(self, x):
        out = self.bn1(F.relu(self.conv1(x)))
        out = self.MaxPool2d1(out)
        out = self.bn2(F.relu(self.conv2(out)))
        out = self.MaxPool2d2(out)
        
        out = out.view(-1, 32)

        out = self.bn3(F.relu(self.fc1(out)))
        out = self.bn4(F.relu(self.fc2(out)))
        out = self.fc3(out)

        return out

################################################
#                   ResNet                     #
################################################

class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y = y + x
        y = F.relu(y)
        
        return y

class ResNet(nn.Module):
    def __init__(self, nb_residual_blocks, nb_channels,
                 kernel_size = 3, nb_classes = 10, img_size=(14,14)):
        super().__init__()
        self.conv = nn.Conv2d(1, nb_channels, kernel_size = 1)
                              
        self.bn = nn.BatchNorm2d(nb_channels)
        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(nb_channels, kernel_size)
              for _ in range(nb_residual_blocks))
        )
        self.avg = nn.AvgPool2d(kernel_size = img_size[0])
        self.fc = nn.Linear(nb_channels, nb_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.resnet_blocks(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

################################################
#                  ResNeXt                     #
################################################

class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, width, cardinality):
        super(ResNeXtBlock, self).__init__()

        d = width * cardinality
        self.conv1 = nn.Conv2d(in_channels, d, 1, padding=0)
        self.bn1   = nn.BatchNorm2d(d)

        self.conv2 = nn.Conv2d(d, d, 3, padding=1, groups=cardinality)
        self.bn2   = nn.BatchNorm2d(d)

        self.conv3 = nn.Conv2d(d, in_channels, 1, padding=0)
        self.bn3   = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y = F.relu(y)
        y = self.bn3(self.conv3(y))
        y = F.relu(x + y)
        return y

class ResNeXt(nn.Module):
    def __init__(self, in_channels=1, img_size=(14,14), n_classes=10, filters=100, nb_blocks=10, width=4, cardinality=32):
        super(ResNeXt, self).__init__()
        # Convolution initial
        self.conv0 = nn.Conv2d(in_channels, filters, 3, padding=1)
        self.bn0   = nn.BatchNorm2d(filters)
        # ResNeXt blocks
        self.resnext_blocks = nn.Sequential(
            *[ResNeXtBlock(filters, width, cardinality) for _ in range(nb_blocks)]
        )
        # Classifier
        self.conv1 = nn.Conv2d(filters, 1, 1, padding=0)
        self.bn1   = nn.BatchNorm2d(1)
        self.fc1   = nn.Linear(img_size[0]*img_size[1], n_classes)

    def forward(self, x):
        # Convolution initial
        y = self.bn0(self.conv0(x))
        y = F.relu(y)
        # ResNeXt blocks
        y = self.resnext_blocks(y)
        # Classifier
        y = self.bn1(self.conv1(y))
        y = F.relu(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        return y