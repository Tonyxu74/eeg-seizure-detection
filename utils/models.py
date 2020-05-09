import pretrainedmodels
from torch import nn
from myargs import args
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # adjust resnet to fit our data
        self.resnet_model = pretrainedmodels.__dict__[args.model_name](num_classes=1000, pretrained='imagenet')
        del self.resnet_model.last_linear
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(args.num_electrodes, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # final fc layers that fit our data
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        resnet_features = self.resnet_model.features(x)
        y = self.avgpool(resnet_features)
        y = torch.flatten(y, start_dim=1)  # don't flatten batch
        y = self.fc(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.softmax(y)
        return y
