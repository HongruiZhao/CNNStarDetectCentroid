import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.functional import norm
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import math

from neural_net.CentroidNet import CentroidNet
from loss_func import mse_loss, bce_loss
from data_load import StarDataSet


# global variable
device = torch.device("cuda:0")


def training(model, criterion_mse, criterion_bce, optimizer, input, target_seg, target_dist, epoch, writer):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    optimizer.zero_grad()
    prediction = model(input)
    seg_prediction = prediction[0][0]
    dist_prediction = prediction[0][1]

    loss_dist = criterion_mse(torch.mul(dist_prediction,target_seg), torch.mul(target_dist, target_seg)) 
    loss_seg =  criterion_bce(torch.sigmoid(seg_prediction), target_seg )
 
    loss =  0.6*loss_dist + 0.4*loss_seg

    loss.backward()
    optimizer.step()
    
    # save running loss 
    running_loss = loss.item()

    # clean GPU cache
    torch.cuda.empty_cache()

    # print result
    print( 'epoch = {},  train_loss = {:.4f}. {:.2f} s per epoch'.format(epoch, running_loss, time.time() - start_time) )

    # tensorboard
    writer.add_scalar('Loss/training', running_loss, epoch) 

    return running_loss





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
        python overfitting.py --w 0 --ep 1000 --trial 1 --lr 0.005
    """
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w", type=float, default=1) # weight decay = w * 1e-5
    parser.add_argument("--ep", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    # set manual seed
    torch.manual_seed(args.seed)

    # create model 
    model = CentroidNet( enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=2 ).to(device)
    print("model {} is created!".format(model.__class__.__name__))

    # for tensorboard
    writer = SummaryWriter('runs/{}_{}'.format(model.__class__.__name__, args.trial))


    #---------------------- Set up datasets ----------------------#
    input = torch.randn(1, 1, 48, 48).to(device) # zero mean normal distribution with standard deviation = 1
    target_seg = torch.rand(48, 48).to(device) # uniform distribution, [0,1)
    target_seg = 1.0 * ( target_seg > 0.5 ) # turn it into a binary seg map
    target_dist = torch.randn(48, 48).to(device) # zero mean normal distribution with standard deviation = 1


    #---------------------- Set up optimizer ----------------------#
    w_input = args.w * 1e-5
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=w_input)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma= 0.5)


    #---------------------- Training Loop ----------------------#
    criterion_mse = mse_loss('sum').to(device)
    criterion_bce = bce_loss().to(device)

    for epoch in range(args.ep):
        running_loss = training(model, criterion_mse, criterion_bce, optimizer, input, target_seg, target_dist, epoch, writer)
        scheduler.step() 

    print("training finish")

    model.eval()
    prediction = model(input)
    seg_prediction = 1.0 * (torch.sigmoid(prediction[0][0]) > 0.5)
    dist_prediction = prediction[0][1]

    print(f'seg map error = { torch.mean(seg_prediction - target_seg)}' )
    print(f'dist map error = { torch.mean(dist_prediction - target_dist)}' )







