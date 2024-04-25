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
from loss_func import mse_loss
from data_load import StarDataSet


# global variable
device = torch.device("cuda:0")


#---------------------- Training iteration and validation  ----------------------#
def validation(model, criterion, val_dataloader):
    model.eval()
    running_loss = 0.0

    for counter, batch in enumerate(val_dataloader, 0):
        img, target, centroid = batch
        img = img.to(device)
        target = target.to(device)

        prediction = model(img)
        loss = criterion(prediction, target)
       
        # save running loss 
        running_loss += loss.item()

        # clean GPU cache
        torch.cuda.empty_cache()
    
    # normalize the loss by the total number of train batches 
    running_loss /= len(val_dataloader)

    return running_loss


def training(model, criterion, optimizer, train_dataloader, val_dataloader, epoch, writer):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for counter, batch in enumerate(train_dataloader, 0):
        img, target, centroid = batch
        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        prediction = model(img)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        
        # save running loss 
        running_loss += loss.item()

         # clean GPU cache
        torch.cuda.empty_cache()
    
    # normalize the loss by the total number of train batches 
    running_loss /= len(train_dataloader)

    # get validation loss
    val_loss = validation(model, criterion, val_dataloader)    

    # print result
    print( 'epoch = {},  train_loss = {:.4f}, val_loss = {:.4f}. {:.2f} s per epoch'.format(epoch, running_loss, val_loss, time.time() - start_time) )

    # tensorboard
    writer.add_scalar('Loss/training', running_loss, epoch) 
    writer.add_scalar('Loss/validation', val_loss, epoch)


    return running_loss, val_loss





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
        python .\training.py --create 1 --batch 1 --LR_range 1 --base_lr 0.01 --max_lr 0.04 --w 0 --ep 30 --save 1 --trial 1
    """
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w", type=float, default=1) # weight decay = w * 1e-5
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--max_lr", type=float, default=0.04)
    parser.add_argument("--step_size", type=int, default=5) # step_size_up = step_size * len(dataloader)
    parser.add_argument("--ep", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--create", type=int, default=1)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--LR_range", type=int, default=0)

    args = parser.parse_args()


    # set manual seed
    torch.manual_seed(args.seed)

    #---------------------- Set/load model ----------------------#

    if args.create == 1:
        model = CentroidNet( enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1 ).to(device)
        print("model {} is created!".format(model.__class__.__name__))
    else: 
        model = torch.load(args.load).to(device)
        print("model {} is loaded!".format(args.load))
    # for tensorboard
    writer = SummaryWriter('runs/{}_{}'.format(model.__class__.__name__, args.trial))




    #---------------------- Set up datasets ----------------------#
    mean = [4.78753542]
    std =  [2.47975301]
    # training dataset 
    train_dataset = StarDataSet(split='train', norm=True, mean=mean, std=std)
    train_dataloader = data.DataLoader( train_dataset, batch_size=args.batch, 
                                        shuffle=True, num_workers=2, 
                                        drop_last=True)
    # validation dataset
    val_dataset = StarDataSet(split="val", norm=True, mean=mean, std=std)
    val_dataloader = data.DataLoader( val_dataset, batch_size=args.batch, 
                                        shuffle=True, num_workers=2, 
                                        drop_last=True)





    #---------------------- Set up optimizer ----------------------#
    w_input = args.w * 1e-5
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=w_input)
    if args.LR_range == 1:
        step_size = args.ep * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr, step_size_up=step_size, cycle_momentum=False)
    else:
        step_size =args.step_size * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr, step_size_up=step_size, cycle_momentum=False)




    #---------------------- Training Loop ----------------------#
    criterion = mse_loss('mean').to(device)
    val_loss_list = []

    for epoch in range(args.ep):
        running_loss, val_loss = training(model, criterion, optimizer, train_dataloader, val_dataloader, epoch, writer)
        scheduler.step() # TODO: DEBUG should step every batch iteration. https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
        val_loss_list.append( val_loss )

    print("training finish")

    if args.save == 1:
        # get module name
        # https://discuss.pytorch.org/t/any-way-to-get-model-name/12877
        torch.save(model, "./saved_models/{}_{}.pt".format(model.__class__.__name__, args.trial))
        print("model saved")




    #---------------------- LR range test ----------------------#
    if args.LR_range == 1:
        step = (args.max_lr - args.base_lr) / args.ep
        lr_list = np.arange(args.base_lr, args.max_lr, step)
        plt.figure(1)
        plt.plot(lr_list, val_loss_list, 'b')
        plt.xlabel("Learning rate", fontsize=12)
        plt.ylabel("Validation loss", fontsize=12)
        plt.show()
