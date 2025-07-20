import torch
import torch.nn as nn
from thop import profile
import numpy as np
from neural_net.mobile_unet import get_inference_time

device = torch.device("cuda:0")

model = torch.load("./saved_models/straylight.pt").to(device)
dummy_input = torch.randn(1,1,480,640).to(device)


# get inference time 
mean_inference_time = get_inference_time(model, dummy_input)
print(f'inference time = {mean_inference_time/1000} s')


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
