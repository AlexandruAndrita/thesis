import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.list_instructions1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )

        self.list_instructions2 = nn.Sequential(
            nn.ConvTranspose2d(32,16,2,2),
            nn.Sigmoid(),
            nn.ConvTranspose2d(16,1,2,2),
            nn.ReflectionPad2d((1,1,0,0))
        )

    def forward(self, x):
        x = self.list_instructions1(x)
        x = self.list_instructions2(x)
        return x