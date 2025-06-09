import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """ [(Conv2d) => (BN) => (ReLu)] * 2 """
    
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,3,padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels,out_channels,3,padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()      
            )

    def forward(self,x):
        return self.double_conv(x)
    



# class DoubleConv(nn.Module):    
#     """
#         depthwise separable convolution
#     """
#     def __init__(self,in_channels,out_channels) -> None:
#         super().__init__()
#         self.double_conv = nn.Sequential(
#                 # depthwise convolution
#                 nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
#                 nn.BatchNorm2d(in_channels),
#                 nn.ReLU6(inplace=True),
#                 # pointwise convolution
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU6(inplace=True),

#                 # depthwise convolution
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU6(inplace=True),
#                 # pointwise convolution
#                 nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU6(inplace=True),
#             )

#     def forward(self,x):
#         return self.double_conv(x)
    



class DownSample(nn.Module):
    """ MaxPool => DoubleConv """
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self,x):
        x  = self.down_sample(x)
        return x




class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,c:int) -> None:
        """ UpSample input tensor by a factor of `c`
                - the value of base 2 log c defines the number of upsample 
                layers that will be applied
        """
        super().__init__()
        n = 0 if c == 0 else int(math.log(c,2))

        self.upsample = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels,in_channels,2,2) for i in range(n)]
        )
        self.conv_3 = nn.Conv2d(in_channels,out_channels,3,padding=1,stride=1)

    def forward(self,x):
        for layer in self.upsample:
            x = layer(x)
        return self.conv_3(x)        




class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)