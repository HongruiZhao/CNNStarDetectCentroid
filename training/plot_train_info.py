import torch
import numpy as np
import matplotlib.pyplot as plt


def visualization(writer, fig_num):
    plt.figure(fig_num)
    plt.subplot(2,2,1)
    plt.plot(writer['epoch'], writer['train_dist'], 'r')
    plt.grid(True)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Train Dist Loss', fontsize = 12)

    plt.subplot(2,2,2)
    plt.plot(writer['epoch'], writer['train_seg'], 'g')
    plt.grid(True)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Train Seg Loss', fontsize = 12)

    plt.subplot(2,2,3)
    plt.plot(writer['epoch'], writer['val_dist'], 'b')
    plt.grid(True)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Val Dist Loss', fontsize = 12)

    plt.subplot(2,2,4)
    plt.plot(writer['epoch'], writer['val_seg'], 'y')
    plt.grid(True)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Val Seg Loss', fontsize = 12)


def visualization_comparison(writer_1, writer_2, label_1, label_2, fig_num):
    plt.figure(fig_num)
    plt.subplot(2,2,1)
    plt.plot(writer_1['train_dist'], 'r', label=label_1)
    plt.plot(writer_2['train_dist'], 'g', label=label_2)
    plt.grid(True)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Train Dist Loss', fontsize = 12)
    plt.legend(loc='upper right', fontsize = 15)

    plt.subplot(2,2,2)
    plt.plot(writer_1['train_seg'], 'r', label=label_1)
    plt.plot(writer_2['train_seg'], 'g', label=label_2)
    plt.grid(True)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Train Seg Loss', fontsize = 12)
    plt.legend(loc='upper right', fontsize = 15)

    plt.subplot(2,2,3)
    plt.plot(writer_1['val_dist'], 'r', label=label_1)
    plt.plot(writer_2['val_dist'], 'g', label=label_2)
    plt.grid(True)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Val Dist Loss', fontsize = 12)
    plt.legend(loc='upper right', fontsize = 15)

    plt.subplot(2,2,4)
    plt.plot(writer_1['val_seg'], 'r', label=label_1)
    plt.plot(writer_2['val_seg'], 'g', label=label_2)
    plt.grid(True)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Val Seg Loss', fontsize = 12)
    plt.legend(loc='upper right', fontsize = 15)



def main():
    try:
  
        writer_1 = torch.load("./runs/ELUnet_B10.pt")
        writer_2 = torch.load("./runs/UNet_B10.pt")
        #writer_2 = torch.load("./runs/ELUnet_Inter_2.pt")



        # visualization(writer_1, 1)
        # visualization(writer_2, 2)
        label_1 = 'ELU-Net B10'
        label_2 = 'UNet B10'
        visualization_comparison(writer_1, writer_2, label_1, label_2, 1)
        plt.show()

    except IOError:
            print('file is unavailable for now')




if __name__ == '__main__':
    main()