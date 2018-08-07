import torch.nn as nn
import torch


class ANet(nn.Module):
    def __init__(self):
        super(ANet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.decv2_1 = nn.ConvTranspose2d(256, 256, 8, stride=4, padding=2)
        self.convs2_1 = nn.Conv2d(3, 96, 9, padding=4)
        self.convs2_2 = nn.Conv2d(256, 64, 1)
        self.convs2_3 = nn.Conv2d(160, 64, 5, padding=2)
        self.convs2_4 = nn.Conv2d(64, 64, 5, padding=2)
        self.decv2_2 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)
        
    def forward(self, input):
        conv2_3 = self.features(input)
        decv2_1 = self.decv2_1(conv2_3)
        convs2_1 = self.convs2_1(self.pool1(input))
        convs2_2 = self.convs2_2(decv2_1)
        concat2 = torch.cat((convs2_1, convs2_2), 1)
        convs2_3 = self.convs2_3(concat2)
        convs2_4 = self.convs2_4(convs2_3)
        convs2_5 = self.convs2_4(convs2_4)
        convs2_6 = self.convs2_4(convs2_5)
        decv2_2 = self.decv2_2(convs2_6)
        return decv2_2        
