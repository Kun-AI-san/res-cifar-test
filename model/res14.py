import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3, stride=1, padding=1, to_downsample=False):
        super().__init__()
        self.channels = channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.to_downsample=to_downsample
        if self.to_downsample:
            self.block = nn.Sequential(
                nn.Conv2d(self.channels//2, self.channels, self.kernel_size, self.stride*2, self.padding),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, self.kernel_size, self.stride, self.padding),
                nn.BatchNorm2d(self.channels),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, self.kernel_size, self.stride, self.padding),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, self.kernel_size, self.stride, self.padding),
                nn.BatchNorm2d(self.channels),
            )
        self.out_relu = nn.ReLU()
        self.input_sampler = nn.Sequential(
            nn.Conv2d(in_channels=self.channels//2, out_channels=self.channels, kernel_size=self.kernel_size, stride=self.stride*2, padding=self.padding),
            nn.BatchNorm2d(self.channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.to_downsample:
            out+=self.input_sampler(x)
        else:
            out+=x
        return self.out_relu(out)

    
class Res14(nn.Module):
    def __init__(self, channels=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.channels=channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.starter_conv = nn.Sequential(nn.Conv2d(
            in_channels=3, out_channels=self.channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride
        ),
        nn.BatchNorm2d(self.channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride*2)
        )
        self.layer1=nn.ModuleList([ResidualBlock() for _ in range(2)])
        self.layer2=nn.ModuleList([ResidualBlock(channels=self.channels*2,to_downsample=True) if i%2==0 else ResidualBlock(channels=self.channels*2) for i in range(2)])
        self.layer3=nn.ModuleList([ResidualBlock(channels=self.channels*4,to_downsample=True) if i%2==0 else ResidualBlock(channels=self.channels*4) for i in range(2)])
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_linear = nn.Linear(self.channels*4, 10)
        self.out_relu = nn.ReLU() #optional - did not use it since adding this was causing worse results (mostly overfitting).


    def forward(self, x):
        out = self.starter_conv(x)
        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)
        out = self.avg_pool(out)
        return self.out_linear(out.flatten(1))
