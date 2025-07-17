import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_p=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2d = nn.Dropout2d(dropout_p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout2d(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=100, num_layers=2, kernel_size=3, input_size=(224, 224)):
        super().__init__()
        self.num_layers = num_layers
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size, 1, padding)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size, 1, padding)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size, 1, padding)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size, 1, padding)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size, 1, padding)
        self.bn6 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

        #размер feature map после forward pass dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            x = self._forward_conv(dummy)
            self.flattened_size = x.numel()

        self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        if self.num_layers == 2:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
        elif self.num_layers == 4:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.dropout(x)
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            x = self.dropout(x)
        elif self.num_layers == 6:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            x = self.pool(F.relu(self.bn5(self.conv5(x))))
            x = self.pool(F.relu(self.bn6(self.conv6(x))))
    
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def _forward_conv(self, x):
        if self.num_layers == 2:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
        elif self.num_layers == 4:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
        elif self.num_layers == 6:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            x = self.pool(F.relu(self.bn5(self.conv5(x))))
            x = self.pool(F.relu(self.bn6(self.conv6(x))))
        return x

class CNNWithResidual(nn.Module):
    def __init__(self, input_channels=3, num_classes=100, kernel_size=3, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size, stride=2, padding=padding)
        self.bn1 = nn.BatchNorm2d(32)

        self.res1 = ResidualBlock(32, 32, kernel_size=kernel_size, stride=1)
        self.res2 = ResidualBlock(32, 64, kernel_size=kernel_size, stride=2)
        self.res3 = ResidualBlock(64, 128, kernel_size=kernel_size, stride=2)
        self.res4 = ResidualBlock(128, 256, kernel_size=kernel_size, stride=2)
        self.res5 = ResidualBlock(256, 256, kernel_size=kernel_size, stride=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        if self.num_layers == 1:
            x = self.res1(x)
            x = self.pool(x).flatten(1)
            x = self.dropout(x)
            x = self.fc1(x)

        elif self.num_layers == 3:
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.pool(x).flatten(1)
            x = self.dropout(x)
            x = self.fc2(x)

        elif self.num_layers == 5:
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            x = self.res5(x)
            x = self.pool(x).flatten(1)
            x = self.dropout(x)
            x = self.fc3(x)

        return x


