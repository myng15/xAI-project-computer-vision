import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import print_model_summary, get_available_device, move_to_device

# get device
device = get_available_device()
print(f"Device: {device}\n")


class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Implementation of basic ConvNets

class SimpleConvNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleConvNet, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # # linear layer (64 * 4 * 4 -> 500)
        # self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=500)
        # # dropout layer (p=0.25)
        # self.dropout = nn.Dropout(0.25)
        # # linear layer (500 -> num_classes)
        # self.fc2 = nn.Linear(in_features=500, out_features=num_classes)

        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=768)
        self.dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)

        # add dropout layer
        # x = self.dropout(x)
        # # add 1st hidden layer, with relu activation function
        # x = F.relu(self.fc1(x))
        # # add dropout layer
        # x = self.dropout(x)
        # # add 2nd hidden layer, with relu activation function
        # x = self.fc2(x)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def get_embeddings(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)

        # # add dropout layer
        # x = self.dropout(x)
        # # add 1st hidden layer, with relu activation function
        # x = F.relu(self.fc1(x))
        # # add dropout layer
        # x = self.dropout(x)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return x


class SimpleConvNetV2(nn.Module):
    def __init__(self, num_classes):
        super(SimpleConvNetV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # self.fc1 = nn.Linear(in_features=8 * 8 * 256, out_features=512)
        # self.fc2 = nn.Linear(in_features=512, out_features=64)
        # self.dropout = nn.Dropout(0.25)
        # self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

        self.fc1 = nn.Linear(in_features=8 * 8 * 256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=768)
        self.dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32*32*48
        x = F.relu(self.conv2(x))  # 32*32*96
        x = self.pool(x)  # 16*16*96
        x = self.dropout(x)
        x = F.relu(self.conv3(x))  # 16*16*192
        x = F.relu(self.conv4(x))  # 16*16*256
        x = self.pool(x)  # 8*8*256
        x = self.dropout(x)
        x = x.view(-1, 8 * 8 * 256)  # reshape x

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        x = self.fc3(x)
        return x

    def get_embeddings(self, x):
        x = F.relu(self.conv1(x))  # 32*32*48
        x = F.relu(self.conv2(x))  # 32*32*96
        x = self.pool(x)  # 16*16*96
        x = self.dropout(x)
        x = F.relu(self.conv3(x))  # 16*16*192
        x = F.relu(self.conv4(x))  # 16*16*256
        x = self.pool(x)  # 8*8*256
        x = self.dropout(x)
        x = x.view(-1, 8 * 8 * 256)  # reshape x

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return x


# Implementation of a custom simplified version of ResNet

class ResnetCustomSimplified(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.res_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.res_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pre_fc = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Flatten())

        #self.fc = nn.Linear(512 * 2 * 2, num_classes)

        self.fc1 = nn.Linear(in_features=512 * 2 * 2, out_features=768)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)
        out = self.res_layer1(out) + out
        out = self.conv_layer_3(out)
        out = self.conv_layer_4(out)
        out = self.res_layer2(out) + out
        out = self.pre_fc(out)

        #out = self.fc(out)

        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        return out

    def get_embeddings(self, x):
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)
        out = self.res_layer1(out) + out
        out = self.conv_layer_3(out)
        out = self.conv_layer_4(out)
        out = self.res_layer2(out) + out
        out = self.pre_fc(out)

        out = self.dropout(self.fc1(out)) #New

        return out


# Implementation of ResNet

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.fc1 = nn.Linear(512 * block.expansion, 768)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(768, num_classes)

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
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        #out = self.fc(out)

        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        return out

    def get_embeddings(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.dropout(self.fc1(out)) #New

        return out


def resnet18_custom(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet50_custom(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def get_model(model_name, batch_size, num_classes):
    if model_name == 'simple_convnet':
        model = SimpleConvNet(num_classes).to(device)
    elif model_name == 'simple_convnet_v2':
        model = SimpleConvNetV2(num_classes).to(device)
    elif model_name == 'resnet_custom_simplified':
        model = ResnetCustomSimplified(num_classes).to(device)
    elif model_name == 'resnet18_custom':
        model = resnet18_custom(num_classes).to(device)
    elif model_name == 'resnet50_custom':
        model = resnet50_custom(num_classes).to(device)
    elif model_name == 'resnet50_fine_tuned':
        resnet50_fine_tuned = torchvision.models.resnet50(weights=None)
        # Modify the last fully connected layer for CIFAR-10
        num_ftrs = resnet50_fine_tuned.fc.in_features
        resnet50_fine_tuned.fc = nn.Linear(num_ftrs, num_classes)
        model = resnet50_fine_tuned.to(device)
    elif model_name == 'resnet50_fine_tuned_pretrained_weights':
        resnet50_fine_tuned_pretrained_weights = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Modify the last fully connected layer for CIFAR-10
        num_ftrs = resnet50_fine_tuned_pretrained_weights.fc.in_features
        resnet50_fine_tuned_pretrained_weights.fc = nn.Linear(num_ftrs, num_classes)
        model = resnet50_fine_tuned_pretrained_weights.to(device)

    model = move_to_device(model, device)
    print_model_summary(model, batch_size)

    return model


#def get_loss_func(loss_func_name):
#    if loss_func_name == 'cross_entropy':
#        return nn.CrossEntropyLoss()
