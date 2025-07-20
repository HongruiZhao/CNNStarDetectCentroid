import torch
from torch.functional import norm
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import os 

from neural_net.CentroidNet import CentroidNet
from neural_net.mobile_unet import MobileUNet
from neural_net.squeezeunet import squeeze_unet
from neural_net.squeezeunet_M import squeeze_unet_M
from neural_net.elunet import ELUnet
from neural_net.elunet_squeeze import ELUnet_squeeze
from neural_net.elunet_inter import ELUnet_Inter
from neural_net.elunet_resnet34 import ELUnet_ResNet34

from loss_func import mse_loss, bce_loss
from data_load import StarDataSet

import numpy as np

# global variable
device = torch.device("cuda:0")


#---------------------- Training iteration and validation  ----------------------#
def validation(model, criterion_mse, criterion_bce, val_dataloader):
    model.eval()
    running_dist_loss = 0.0
    running_seg_loss = 0.0

    for counter, batch in enumerate(val_dataloader, 0):
        images, dist_map, seg_map, centroid_real  = batch
        images = images.to(device)
        dist_map = dist_map.to(device)
        seg_map = seg_map.to(device)

        prediction = model(images)
        seg_prediction = prediction[:,0]
        dist_prediction = prediction[:,1]

        loss_dist = criterion_mse(torch.mul(dist_prediction, seg_map), torch.mul(dist_map, seg_map)) 
        loss_seg =  criterion_bce(torch.sigmoid(seg_prediction), seg_map)
                
        # save running loss 
        running_dist_loss += loss_dist.item()
        running_seg_loss += loss_seg.item()

        # clean GPU cache
        torch.cuda.empty_cache()
    
    # normalize the loss by the total number of train batches 
    running_dist_loss /= len(val_dataloader)
    running_seg_loss /= len(val_dataloader)

    return running_dist_loss, running_seg_loss




def training(model, criterion_mse, criterion_bce, optimizer, train_dataloader, val_dataloader, epoch, writer):
    model.train()
    running_dist_loss = 0.0
    running_seg_loss = 0.0

    start_time = time.time()

    for counter, batch in enumerate(train_dataloader, 0):
        images, dist_map, seg_map, centroid_real  = batch
        images = images.to(device)
        dist_map = dist_map.to(device)
        seg_map = seg_map.to(device)

        optimizer.zero_grad()
        prediction = model(images)
        seg_prediction = prediction[:,0]
        dist_prediction = prediction[:,1]

        loss_dist = criterion_mse(torch.mul(dist_prediction, seg_map), torch.mul(dist_map, seg_map)) 
        loss_seg =  criterion_bce(torch.sigmoid(seg_prediction), seg_map)
        
        loss =  2.5*loss_dist + loss_seg

        loss.backward()
        optimizer.step()
        
        # save running loss 
        running_dist_loss += loss_dist.item()
        running_seg_loss += loss_seg.item()

        # clean GPU cache
        torch.cuda.empty_cache()
    
    # normalize the loss by the total number of train batches 
    running_dist_loss /= len(train_dataloader)
    running_seg_loss /= len(train_dataloader)

    # get validation loss
    with torch.no_grad():
        val_dist_loss, val_seg_loss = validation(model, criterion_mse, criterion_bce, val_dataloader)    

    # print result
    print( f'epoch = {epoch}, train_dist_loss = {running_dist_loss}, train_seg_loss = {running_seg_loss}, val_dist_loss = {val_dist_loss}, val_seg_loss = {val_seg_loss}, lr = {optimizer.param_groups[0]["lr"]}, {time.time() - start_time} s per epoch' )

    # saving training info
    writer['epoch'].append(epoch)
    writer['train_dist'].append(running_dist_loss)
    writer['train_seg'].append(running_seg_loss)
    writer['val_dist'].append(val_dist_loss)
    writer['val_seg'].append(val_seg_loss)

    return writer




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
        python .\training_stepLR.py --trial 1 
    """
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w", type=float, default=5) # weight decay = w * 1e-4
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--ep", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--create", type=int, default=1)
    parser.add_argument("--load", type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    writer = SummaryWriter()

    # set manual seed
    torch.manual_seed(args.seed)


    #---------------------- Set/load model ----------------------#
    if args.create == 1:
        #model = CentroidNet( enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=2 ).to(device)
        #model = MobileUNet(ch_in=1, ch_out=2).to(device)
        model = ELUnet(1,2,8).to(device)
        #model = squeeze_unet(1, 2).to(device)
        #model = squeeze_unet_M(1,2).to(device)
        #model = ELUnet_Inter(1, 2, 8).to(device)
        #model = ELUnet_ResNet34(1, 2, 8).to(device)
        print("model {} is created!".format(model.__class__.__name__))
    else: 
        model = torch.load(args.load).to(device)
        print("model {} is loaded!".format(args.load))
    # for saving training information 
    writer_info = {'epoch':[], 'train_dist':[], 'train_seg':[], 'val_dist':[], 'val_seg':[]}


    #---------------------- Set up datasets ----------------------#
    mean = [25.36114133]
    std =  [44.31162568]

    # training dataset 
    train_dataset = StarDataSet(split='train', norm=True, mean=mean, std=std)
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=2, 
                                    drop_last=False) 
    # validation dataset
    val_dataset = StarDataSet(split="val", norm=True, mean=mean, std=std)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=2, 
                                    drop_last=False)


    #---------------------- Set up optimizer ----------------------#
    w_input = args.w * 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=w_input)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    #---------------------- Training Loop ----------------------#
    criterion_mse = mse_loss('mean').to(device)
    criterion_bce = bce_loss().to(device)

    for epoch in range(args.ep):
        writer_info = training(model, criterion_mse, criterion_bce, optimizer, train_dataloader, val_dataloader, epoch, writer_info) 
        scheduler.step() 

        writer.add_scalar('train_dist', writer_info['train_dist'][-1], epoch)
        writer.add_scalar('train_seg', writer_info['train_seg'][-1], epoch)
        writer.add_scalar('val_dist', writer_info['val_dist'][-1], epoch)
        writer.add_scalar('val_seg', writer_info['val_seg'][-1], epoch)


        if (epoch) % 10 == 0:
            # get module name
            # https://discuss.pytorch.org/t/any-way-to-get-model-name/12877
            torch.save(model, "./saved_models/{}_{}_{}.pt".format(model.__class__.__name__, args.trial, epoch+1))
            print("model saved")

    print("training finish")


