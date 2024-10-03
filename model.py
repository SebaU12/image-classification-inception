import torch
import torch.nn as nn
from inception_blocks import InceptionA_Block, InceptionB_Block, InceptionC_Block
from reduction_blocks import ReductionA, ReductionB

class StemBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super(StemBlock, self).__init__()
        self.conv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3,3), stride=2),
                nn.ReLU())
        self.conv_layer2 = nn.Sequential( 
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3)),
                nn.ReLU())
        self.conv_layer3 = nn.Sequential( 
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1),
                nn.ReLU())
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3,3), stride=2),
                nn.ReLU())
        self.conv_layer5_1 = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,1)),
                nn.ReLU())
        self.conv_layer5_2 = nn.Sequential( 
                nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,1)),
                nn.ReLU())
        self.conv_layer6_1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3,3)),
                nn.ReLU())
        self.conv_layer6_2 = nn.Sequential( 
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7,1), padding=(3, 0)),
                nn.ReLU())
        self.conv_layer7 = nn.Sequential( 
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,7), padding=(0, 3)),
                nn.ReLU())
        self.conv_layer8 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3,3)),
                nn.ReLU())
        self.conv_layer9 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3,3), stride=2),
                nn.ReLU())
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x_1 = self.max_pool1(x)
        x_2 = self.conv_layer4(x)
        filter_concat1 = torch.cat((x_1, x_2), dim=1)
        x_1 = self.conv_layer5_1(filter_concat1)
        x_1 = self.conv_layer6_1(x_1)
        x_2 = self.conv_layer5_2(filter_concat1)
        x_2 = self.conv_layer6_2(x_2)
        x_2 = self.conv_layer7(x_2)
        x_2 = self.conv_layer8(x_2)
        filter_concat2 = torch.cat((x_1, x_2), dim=1)
        x_1 = self.conv_layer9(filter_concat2)
        x_2 = self.max_pool2(filter_concat2)
        filter_concat3 = torch.cat((x_1, x_2), dim=1)
        x =  self.activation(filter_concat3)
        return x

class Inception_ResNet_v2(nn.Module):
    def __init__(self, in_channels) -> None:
        super(Inception_ResNet_v2, self).__init__()
        self.stem = StemBlock(in_channels)
        self.inceptionA_list = nn.ModuleList()
        self.inceptionB_list = nn.ModuleList()
        self.inceptionC_list = nn.ModuleList()
        self.reductionA = ReductionA(384)
        self.reductionB = ReductionB(1152)
        self.avarage = nn.Sequential(
                nn.AvgPool2d(kernel_size=(8,8)),
                nn.Dropout(0.8))
        self.fc = nn.Linear(in_features=2144, out_features=100)
        self.softmax = nn.Softmax(dim=1)

        for _ in range(5):
            inceptionA = InceptionA_Block(384)
            self.inceptionA_list.append(inceptionA)

        for _ in range(10):
            inceptionB = InceptionB_Block(1152)
            self.inceptionB_list.append(inceptionB)
        self.pro = InceptionB_Block(1152)

        for _ in range(5):
            inceptionC = InceptionC_Block(2144)
            self.inceptionC_list.append(inceptionC)


    def forward(self, x):
        x = self.stem(x)
        #print(f'Output STEM: {x.size()}')
        for inceptionA in self.inceptionA_list:
            x = inceptionA(x)
        #print(f'Output InceptionA: {x.size()}')
        x = self.reductionA(x)
        #print(f'Output RedutionA: {x.size()}')
        for inceptionB in self.inceptionB_list:
            x = inceptionB(x)
        #print(f'Output InceptionB: {x.size()}')
        x = self.reductionB(x)
        #print(f'Output RedutionB: {x.size()}')
        for inceptionC in self.inceptionC_list:
            x = inceptionC(x)
        #print(f'Output InceptionC: {x.size()}')
        x = self.avarage(x)
        #print(f'Output AVG POOL: {x.size()}')
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        #print(f'Output fully connected: {x.size()}')
        x = self.softmax(x)
        return x

