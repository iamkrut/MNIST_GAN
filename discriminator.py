import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5)
        )
        self.out = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = x.reshape(x.size(0), -1)
        x = self.hidden2(x)
        x = self.out(x)

        return x
