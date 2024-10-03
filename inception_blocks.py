import torch
import torch.nn as nn

class InceptionA_Block(nn.Module):
    def __init__(self, in_channels) -> None:
        super(InceptionA_Block, self).__init__()
        self.conv_layer1_1 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer1_2 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer1_3 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer2_1 = nn.Sequential( 
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
                nn.ReLU())
        self.conv_layer2_2 = nn.Sequential( 
                nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1),
                nn.ReLU())
        self.conv_layer3 = nn.Sequential( 
                nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), padding=1),
                nn.ReLU())
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=384, kernel_size=(1, 1)),
                nn.ReLU())

        self.activation = nn.ReLU()

    def forward(self, x):
        x_1 = self.conv_layer1_1(x)
        x_2 = self.conv_layer1_2(x)
        x_2 = self.conv_layer2_1(x_2)
        x_3 = self.conv_layer1_3(x)
        x_3 = self.conv_layer2_2(x_3)
        x_3 = self.conv_layer3(x_3)
        filter_concat = torch.cat((x_1, x_2, x_3), dim=1)
        x = x + self.conv_layer4(filter_concat)
        x = self.activation(x)
        return x

class InceptionB_Block(nn.Module):
    def __init__(self, in_channels) -> None:
        super(InceptionB_Block, self).__init__()
        self.conv_layer1_1 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=192, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer1_2 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer2 = nn.Sequential( 
                nn.Conv2d(in_channels=128, out_channels=160, kernel_size=(1, 7), padding=(0, 3)),
                nn.ReLU())
        self.conv_layer3 = nn.Sequential( 
                nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(7, 1), padding=(3, 0)),
                nn.ReLU())
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels=384, out_channels=1152, kernel_size=(1, 1)),
                nn.ReLU())

        self.activation = nn.ReLU()

    def forward(self, x):
        x_1 = self.conv_layer1_1(x)
        x_2 = self.conv_layer1_2(x)
        x_2 = self.conv_layer2(x_2)
        x_2 = self.conv_layer3(x_2)
        filter_concat = torch.cat((x_1, x_2), dim=1)
        x = x + self.conv_layer4(filter_concat)
        x = self.activation(x)
        return x

class InceptionC_Block(nn.Module):
    def __init__(self, in_channels) -> None:
        super(InceptionC_Block, self).__init__()
        self.conv_layer1_1 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=192, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer1_2 = nn.Sequential( 
                nn.Conv2d(in_channels=in_channels, out_channels=192, kernel_size=(1, 1)),
                nn.ReLU())
        self.conv_layer2 = nn.Sequential( 
                nn.Conv2d(in_channels=192, out_channels=224, kernel_size=(1, 3), padding=(0, 1)),
                nn.ReLU())
        self.conv_layer3 = nn.Sequential( 
                nn.Conv2d(in_channels=224, out_channels=256, kernel_size=(3, 1), padding=(1, 0)),
                nn.ReLU())
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels=448, out_channels=2144, kernel_size=(1, 1)),
                nn.ReLU())

        self.activation = nn.ReLU()

    def forward(self, x):
        x_1 = self.conv_layer1_1(x)
        x_2 = self.conv_layer1_2(x)
        x_2 = self.conv_layer2(x_2)
        x_2 = self.conv_layer3(x_2)
        filter_concat = torch.cat((x_1, x_2), dim=1)
        x = x + self.conv_layer4(filter_concat)
        x = self.activation(x)
        return x


