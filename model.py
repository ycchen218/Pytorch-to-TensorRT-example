import torch.nn as nn
from torchvision import models

class MNIST_ResNet34(nn.Module):
    def __init__(self):
        super(MNIST_ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=True)

        #for param in self.model.parameters():
        #    param.requires_grad = False
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        logits = self.model(x)
        return logits
