import torch
import torch.nn as nn
from thop import profile
import numpy as np


class fire_module(nn.Module):
    def __init__(self, c_in, c_out_p, c_out, s, transpose=False):
        super(fire_module, self).__init__()

        self.conv_1 = nn.Sequential( nn.Conv2d(c_in, c_out_p, kernel_size=1, bias=False), nn.BatchNorm2d(c_out_p), nn.ReLU6(inplace=True) )
        
        self.conv_2 = nn.Sequential( 
                                    nn.Conv2d(c_out_p, c_out_p, kernel_size=3, stride=s, padding=1, groups=c_out_p, bias=False), 
                                    nn.BatchNorm2d(c_out_p), 
                                    nn.ReLU6(inplace=True),
                                    nn.Conv2d(c_out_p,int(c_out/2), kernel_size=1, bias=False), 
                                    nn.BatchNorm2d(int(c_out/2)), 
                                    nn.ReLU6(inplace=True),
                                    )
        self.conv_3 = nn.Sequential( nn.Conv2d(c_out_p, int(c_out/2), kernel_size=1, stride=s, bias=False), nn.BatchNorm2d(int(c_out/2)), nn.ReLU6(inplace=True) )

        self.conv_2_T = nn.Sequential(  nn.ConvTranspose2d(c_out_p, c_out_p, kernel_size=2, stride=2, groups=c_out_p, bias=False), 
                                        nn.BatchNorm2d(c_out_p), 
                                        nn.ReLU6(inplace=True),
                                        nn.Conv2d(c_out_p, int(c_out/2), kernel_size=1, bias=False),
                                        nn.BatchNorm2d(int(c_out/2)), 
                                        nn.ReLU6(inplace=True),
                                    )
        self.conv_3_T = nn.Sequential( nn.ConvTranspose2d(c_out_p, int(c_out/2), kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(int(c_out/2)), nn.ReLU6(inplace=True) )

        self.trnaspose = transpose


    def forward(self,x):
        if self.trnaspose:
            x = self.conv_1(x)
            x_1 = self.conv_2_T(x)
            x_2 = self.conv_3_T(x)
            x_2 = nn.functional.pad(x_2, (0,1,0,1)) 
            x = torch.cat([x_1, x_2], dim=1)
        else:
            x = self.conv_1(x)
            x_1 = self.conv_2(x)
            x_2 = self.conv_3(x)
            x = torch.cat([x_1, x_2], dim=1)
        
        return x
    



class convolution_block(nn.Module):
    def __init__(self, c_in, c_out):
        super(convolution_block, self).__init__()
        self.conv_1 = nn.Sequential(    nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False), 
                                        nn.BatchNorm2d(c_in), 
                                        nn.ReLU6(inplace=True), 
                                        nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(c_out),
                                        nn.ReLU6(inplace=True), 
                                    )
        self.conv_2 = nn.Sequential(    nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, groups=c_out, bias=False), 
                                        nn.BatchNorm2d(c_out), 
                                        nn.ReLU6(inplace=True), 
                                        nn.Conv2d(c_out, c_out, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(c_out),
                                        nn.ReLU6(inplace=True), 
                                    )    
    def forward(self,x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x 
    



class Down_Sample(nn.Module):
    def __init__(self, c_in, c_out_p, c_out):
        super(Down_Sample, self).__init__()
        self.fire_module_1 =fire_module(c_in, c_out_p, c_out, 2)
        self.fire_module_2 = fire_module(c_out, c_out_p, c_out, 1)
    def forward(self,x):
        x = self.fire_module_1(x)
        x = self.fire_module_2(x)
        return x 




class Up_Sample(nn.Module):
    def __init__(self, c_in, c_out_p1, c_out_p2, c_out):
        super(Up_Sample, self).__init__()
        self.fire_module_T = fire_module(c_in, c_out_p1, c_out, 2 , True)
        self.fire_module_1 = fire_module(c_out*2, c_out_p2, c_out, 1)
        self.fire_module_2 = fire_module(c_out, c_out_p2, c_out, 1)
    def forward(self,x,y):
        x = self.fire_module_T(x)
        x = torch.cat([x,y], dim=1)
        x = self.fire_module_1(x)
        x = self.fire_module_2(x)
        return x




class squeeze_unet_M(nn.Module):
    def __init__(self, c_in, c_out):
        super(squeeze_unet_M, self).__init__()
        self.convblock_1 = convolution_block(c_in, 64)
        
        self.DS1 = Down_Sample(64, 32, 128)
        self.DS2 = Down_Sample(128, 48, 256)
        self.DS3 = Down_Sample(256, 64, 512)
        self.DS4 = Down_Sample(512, 80, 1024)

        self.US1 = Up_Sample(1024, 80, 64, 512)
        self.US2 = Up_Sample(512, 64, 48, 256)
        self.US3 = Up_Sample(256, 48, 32, 128)

        self.convT = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2, bias=False), nn.BatchNorm2d(64), nn.ReLU6(inplace=True) )
        self.convblock_2 = convolution_block(128, 64)
        self.conv_out = nn.Conv2d(64, c_out, 1)


    def forward(self,x):
        x0 = self.convblock_1(x)
        x1 = self.DS1(x0)
        x2 = self.DS2(x1)
        x3 = self.DS3(x2)
        x4 = self.DS4(x3)

        # print(f'x0 = {x0.size()}')
        # print(f'x1 = {x1.size()}')
        # print(f'x2 = {x2.size()}')
        # print(f'x3 = {x3.size()}')
        # print(f'x4 = {x4.size()}')

        x = self.US1(x4, x3)
        #print(x.size())
        x = self.US2(x, x2)
        #print(x.size())
        x = self.US3(x, x1)
        #print(x.size())
        x = self.convT(x)
        #print(x.size())

        x = torch.cat([x,x0], dim=1)
        x = self.convblock_2(x)
        x = self.conv_out(x)

        return x 




if __name__ == "__main__":
    from mobile_unet import get_inference_time
    print("Squeeze UNet M")

    device = torch.device("cuda:0")
    dummy_input = torch.rand(1, 1, 480, 640).to(device)
    model = squeeze_unet_M(1, 2).to(device)

    # get inference time 
    mean_inference_time = get_inference_time(model, dummy_input)
    print(f'inference time = {mean_inference_time/1000} s')


    # get output shape
    predict = model(dummy_input)
    predict = predict.detach().cpu()
    print(f'the output size is {predict.size()}')
    print(f'the predicted seg/dist map size is {predict[0][0].size()}')\
    



    """
        check the total number of parameters in a pytorch model 
        https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    pytorch_total_params = sum(p.numel() for p in model.parameters() )
    print(f'the total number of parameters = {pytorch_total_params / 1000000} millions')




    """ 
        use profile from thop to get the total number of parameters and flops 
        https://github.com/Lyken17/pytorch-OpCounter
    """
    macs, params = profile(model, inputs=(dummy_input,))
    """
        FLOPs:floating point operations (FLOPS = floating point operation per second)
        MACs:multiply-accumulate operations
        1 MACs = 2 FLOPs approximately (https://medium.com/ching-i/cnn-parameters-flops-macs-cio-%E8%A8%88%E7%AE%97-9575d61765cc)
        MFLOPS = 10**6 FLOPS, GFLOPS = 10**9 FLOPS
        FLOPS is determined by the hardware
        inference time is roughly FLOPs / FLOPS (https://www.thinkautonomous.ai/blog/deep-learning-optimization/#calculating-the-inference-time)
    """
    print('MACs = ' + str(macs/10**9) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')


