import torch
import torch.nn as nn

class ReductionA(nn.Module):
    def __init__(self, in_channels) -> None:
        super(ReductionA, self).__init__()
        k = 256
        l = 256
        m = 384
        n = 384
        self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.conv_layer1_1 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=n, kernel_size=(3, 3), stride=2),
                nn.ReLU())
        self.conv_layer1_2 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=k, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer2 = nn.Sequential( 
                nn.Conv2d(in_channels=k, out_channels=l, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU())
        self.conv_layer3 = nn.Sequential( 
                nn.Conv2d(in_channels=l, out_channels=m, kernel_size=(3, 3), stride=2),
                nn.ReLU())

        self.activation = nn.ReLU()

    def forward(self, x):
        x_1 = self.max_pool(x)
        x_2 = self.conv_layer1_1(x)
        x_3 = self.conv_layer1_2(x)
        x_3 = self.conv_layer2(x_3)
        x_3 = self.conv_layer3(x_3)
        filter_concat = torch.cat((x_1, x_2, x_3), dim=1)
        x = self.activation(filter_concat)
        return x

class ReductionB(nn.Module):
    def __init__(self, in_channels) -> None:
        super(ReductionB, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.conv_layer1_1 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer1_2 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer1_3 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer2_1 = nn.Sequential( 
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=2),
                nn.ReLU())
        self.conv_layer2_2 = nn.Sequential( 
                nn.Conv2d(in_channels=256, out_channels=288, kernel_size=(3, 3), stride=2),
                nn.ReLU())
        self.conv_layer2_3 = nn.Sequential( 
                nn.Conv2d(in_channels=256, out_channels=288, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU())
        self.conv_layer3 = nn.Sequential( 
                nn.Conv2d(in_channels=288, out_channels=320, kernel_size=(3, 3), stride=2),
                nn.ReLU())

        self.activation = nn.ReLU()

    def forward(self, x):
        x_1 = self.max_pool(x)
        x_2 = self.conv_layer1_1(x)
        x_2 = self.conv_layer2_1(x_2)
        x_3 = self.conv_layer1_2(x)
        x_3 = self.conv_layer2_2(x_3)
        x_4 = self.conv_layer1_3(x)
        x_4 = self.conv_layer2_3(x_4)
        x_4 = self.conv_layer3(x_4)
        filter_concat = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        x = self.activation(filter_concat)
        return x


