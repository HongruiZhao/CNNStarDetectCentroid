import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile




class InvertedResidualBlock(nn.Module):
    """
    inverted residual block used in MobileNetV2
    """
    def __init__(self, in_c, out_c, stride, expansion_factor=6):
        super(InvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if stride == 1 else False

        # expansion factor or t as mentioned in the paper
        ex_c = int(in_c * expansion_factor)
        self.conv = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(in_c, ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ex_c),
            nn.ReLU6(inplace=True),
            # depthwise convolution
            nn.Conv2d(ex_c, ex_c, 3, stride, 1, groups=ex_c, bias=False),
            nn.BatchNorm2d(ex_c),
            nn.ReLU6(inplace=True),
            # pointwise convolution
            nn.Conv2d(ex_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.use_conv1x1 =  False if in_c == out_c else True
        self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.use_skip_connection:
            out = self.conv(x)
            if self.use_conv1x1:
                x = self.conv1x1(x)
            return x+out
        else:
            return self.conv(x)

    


class DoubleConv(nn.Module):    
    def __init__(self,c_in, t, c_out, s):
        super(DoubleConv, self).__init__()
        self.IR_1 =InvertedResidualBlock(c_in, c_out, s, t)
        self.IR_2 = InvertedResidualBlock(c_out, c_out, 1, t)
    def forward(self,x):
        x = self.IR_1(x)
        x = self.IR_2(x)
        return x

     


class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,c:int):
        """ UpSample input tensor by a factor of `c`
                - the value of base 2 log c defines the number of upsample 
                layers that will be applied
        """
        super(UpSample, self).__init__()
        self.scale_factor = c
        self.conv_3 = nn.Conv2d(in_channels,out_channels,3,padding=1,stride=1)

    def forward(self,x):
        if self.scale_factor != 0:
            x = F.interpolate( x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True )
        x= self.conv_3(x)      
        return x      




class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)




class ELUnet_Mobile(nn.Module):
    def __init__(self,in_channels,out_channels,n:int = 8):
        """ 
        Construct the Elu-net model.
        Args:
            in_channels: The number of color channels of the input image. 0:for binary 3: for RGB
            out_channels: The number of color channels of the input mask, corresponds to the number
                            of classes.Includes the background
            n: Channels size of the first CNN in the encoder layer. The bigger this value the bigger 
                the number of parameters of the model. Defaults to n = 8, which is recommended by the 
                authors of the paper.
        """
        super(ELUnet_Mobile, self).__init__()
        # ------ Input convolution --------------
        self.in_conv = DoubleConv(in_channels, 3, n, 1)
        # -------- Encoder ----------------------
        self.down_1 = DoubleConv(n, 2, 2*n, 2)
        self.down_2 = DoubleConv(2*n, 2, 4*n, 2)
        self.down_3 = DoubleConv(4*n, 2, 8*n, 2)
        self.down_4 = DoubleConv(8*n, 2, 16*n, 2)
        
        # -------- Upsampling ------------------
        self.up_16n_8n = UpSample(16*n,8*n,2)

        self.up_8n_n = UpSample(8*n,n,8)
        self.up_8n_2n = UpSample(8*n,2*n,4)
        self.up_8n_4n = UpSample(8*n,4*n,2)
        self.up_8n_4n_2 = UpSample(8*n,4*n,2)
        self.up_8n_8n = UpSample(8*n,8*n,0)

        self.up_4n_n = UpSample(4*n,n,4)
        self.up_4n_2n = UpSample(4*n,2*n,2)
        self.up_4n_2n_2 = UpSample(4*n,2*n,2)
        self.up_4n_4n = UpSample(4*n,4*n,0)

        self.up_2n_n = UpSample(2*n,n,2)
        self.up_2n_n_2 = UpSample(2*n,n,2)
        self.up_2n_2n = UpSample(2*n,2*n,0)
        

        self.up_n_n = UpSample(n,n,0)
     
        # ------ Decoder block ---------------
        self.dec_4 = DoubleConv(2*8*n, 2, 8*n, 1)
        self.dec_3 = DoubleConv(3*4*n, 2, 4*n, 1)
        self.dec_2 = DoubleConv(4*2*n, 2, 2*n, 1)
        self.dec_1 = DoubleConv(5*n, 2, n, 1)
        # ------ Output convolution

        self.out_conv = OutConv(n,out_channels)

    def forward(self,x):
        x = self.in_conv(x) # ch output = n
        # ---- Encoder outputs
        x_enc_1 = self.down_1(x) # 2n
        x_enc_2 = self.down_2(x_enc_1) # 4n
        x_enc_3 = self.down_3(x_enc_2) # 8n
        x_enc_4 = self.down_4(x_enc_3) # 16n
    
        # ------ decoder outputs
        x_up_1 = self.up_16n_8n(x_enc_4)
        x_dec_4 = self.dec_4(torch.cat([x_up_1,self.up_8n_8n(x_enc_3)],dim=1)) # 8n

        x_up_2 = self.up_8n_4n(x_dec_4)
        x_dec_3 = self.dec_3(torch.cat([x_up_2,
            self.up_8n_4n_2(x_enc_3),
            self.up_4n_4n(x_enc_2)
            ],
        dim=1)) # 4n

        x_up_3 = self.up_4n_2n(x_dec_3)
        x_dec_2 = self.dec_2(torch.cat([
            x_up_3,
            self.up_8n_2n(x_enc_3),
            self.up_4n_2n_2(x_enc_2),
            self.up_2n_2n(x_enc_1)
        ],dim=1)) # 2n

        x_up_4 = self.up_2n_n(x_dec_2)
        x_dec_1 = self.dec_1(torch.cat([
            x_up_4,
            self.up_8n_n(x_enc_3),
            self.up_4n_n(x_enc_2),
            self.up_2n_n_2(x_enc_1),
            self.up_n_n(x)
        ],dim=1)) # n


        return self.out_conv(x_dec_1)
    



if __name__ == "__main__":
    from mobile_unet import get_inference_time

    print("[ELUnet_Mobile]")

    device = torch.device("cuda:0")
    dummy_input = torch.rand(1, 1, 480, 640).to(device)
    model = ELUnet_Mobile(1,2,8).to(device)

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