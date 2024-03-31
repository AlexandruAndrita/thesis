import torch
import torch.nn as nn
import torch.nn.functional as F


# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()

#         self.list_instructions1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(),

#             nn.MaxPool2d(2,2),

#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),

#             nn.MaxPool2d(2, 2)
#         )

#         self.list_instructions2 = nn.Sequential(
#             nn.ConvTranspose2d(32,16,2,2),
#             nn.Sigmoid(),
#             nn.ConvTranspose2d(16,1,2,2),
#             nn.ReflectionPad2d((1,1,0,0))
#         )

#         self.activation = nn.Sigmoid()  # Add a final activation function

#     def forward(self, x):
#         x = self.list_instructions1(x)
#         x = self.list_instructions2(x)
#         x = self.activation(x)  # Apply the final activation function
#         return x
# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#
#         self.list_instructions = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.MaxPool2d(2, 2),
#
#             nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#
#             nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#
#             nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.list_instructions(x)
#         #x = F.interpolate(x, size=(128, 170), mode='bilinear', align_corners=False)
#         return x
    
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.list_instructions = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.list_instructions(x)
        return x