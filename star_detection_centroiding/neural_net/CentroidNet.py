from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from thop import profile
from .mobile_unet import get_inference_time

class Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv_1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(c_out) 
        self.conv_2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)

        return x




class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-2)])
        self.last_block = Block(chs[-2], chs[-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        encoder_features = []
        for block in self.enc_blocks:
            x = block(x)
            encoder_features.append(x)
            x = self.pool(x)
        x = self.last_block(x)
        return x, encoder_features 




class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=2, stride=2, bias=True) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            x = F.interpolate( x, size=(encoder_features[i].size()[2], encoder_features[i].size()[3]), mode='bilinear', align_corners=True )
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x




class CentroidNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.conv_last = nn.Conv2d(dec_chs[-1], num_class, kernel_size=1, bias=True)
        
    def forward(self, x):
        x, encoder_features  = self.encoder(x)
        out = self.decoder(x, encoder_features[::-1]) # reverse encoder_features
        out = self.conv_last(out)
        #out = torch.sigmoid(out) # for bceloss, sigmoid should be used at the last layer
        return out




if __name__ == "__main__":
    
    device = torch.device("cuda:0")
    model = CentroidNet( enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class = 2 ).to(device)
    input = torch.randn(1, 1, 480, 640).to(device)





    # get inference time
    inference_time = get_inference_time(model, input)
    print(f'inference time = {inference_time/1000} s')




    # get output size
    predict = model(input)
    predict = predict.detach().cpu()
    print(f'the output size is {predict.size()}')
    print(f'the predicted seg/dist map size is {predict[0][0].size()}')




    """
        check the total number of parameters in a pytorch model 
        https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    pytorch_total_params = sum(p.numel() for p in model.parameters() )
    print(f'the total number of parameters = {pytorch_total_params / 1000000} millions')

    # # to export bilinear interploation to onnx correctly, opset_version=11
    # print("output ONNX model for Netron")
    # torch.onnx.export(model, input, 'CentroidNet.onnx', opset_version=11)




    """ 
        use profile from thop to get the total number of parameters and flops 
        https://github.com/Lyken17/pytorch-OpCounter
    """
    macs, params = profile(model, inputs=(input,))
    """
        FLOPs:floating point operations (FLOPS = floating point operation per second)
        MACs:multiply-accumulate operations
        1 MACs = 2 FLOPs approximately (https://medium.com/ching-i/cnn-parameters-flops-macs-cio-%E8%A8%88%E7%AE%97-9575d61765cc)
        MFLOPS = 10**6 FLOPS, GFLOPS = 10**9 FLOPS
    """
    print( 'MACs = ' + str(macs/10**9) + 'G' ) 
    print('Params = ' + str(params/1000**2) + 'M')


