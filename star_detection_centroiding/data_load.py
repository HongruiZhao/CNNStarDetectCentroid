import os
import glob
from torch.utils import data
from torchvision.transforms import Normalize
import torch
import numpy as np
import argparse
from visualization import visualization


class StarDataSet(data.Dataset):
    """
    A custom pytorch dataset.  
    need to have: __init__, __len__, and __getitem__.  
    """
    def __init__(self, split="train", data_dir="../data_generation/data_Oct19", norm=False, mean=None, std=None):
    #def __init__(self, split="train", data_dir="../data_generation/data_Feb12_2024", norm=False, mean=None, std=None):
        assert(split in ["train", "val", "test"])

        self.img_dir = os.path.join(data_dir, split + "_raw")
        self.dist_map_dir = os.path.join(data_dir, split + "_dist_map")
        self.seg_map_dir = os.path.join(data_dir, split + "_seg_map")
        self.centroid_dir = os.path.join(data_dir, split + "_centroid")
        
        self.filenames = [
            os.path.splitext(os.path.basename(l))[0] for l in glob.glob(self.img_dir + "/*.npy")
        ]

        self.norm = norm
        self.mean = mean 
        self.std = std

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        img = np.load(os.path.join(self.img_dir, filename) + ".npy")
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0) # add one fake channel dimension

        dist_map = np.load(os.path.join(self.dist_map_dir, filename.replace("raw_image", "dist_map")) + ".npy")
        dist_map = torch.tensor(dist_map, dtype=torch.float32)
        #dist_map = dist_map.unsqueeze(0) # add one fake channel dimension

        seg_map = np.load(os.path.join(self.seg_map_dir, filename.replace("raw_image", "seg_map")) + ".npy")
        seg_map = torch.tensor(seg_map, dtype=torch.float32)
        #seg_map = seg_map.unsqueeze(0) # add one fake channel dimension

        centroid = np.load(os.path.join(self.centroid_dir, filename.replace("raw_image", "centroid")) + ".npy")
        centroid = torch.tensor(centroid, dtype=torch.float32)

        if self.norm: 
            # Normalize([mean[0], ..., mean[n]], [std[1],..,std[n]])
            # https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Normalize
            # mean and std for each channel
            # output[channel] = (input[channel] - mean[channel]) / std[channel]
            # test data should use the same data normalization parameters as train data 
            img = Normalize(self.mean, self.std)(img)


        # https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do
        # contiguous()
        return img.contiguous(), dist_map, seg_map, centroid




def compute_mean_std(train_dataloader):

    data_len = len(train_dataloader)
    batch = next(iter(train_dataloader)) 
    batch_size = batch[0].size()[0]
    num_channel = batch[0].size()[1]
    height = batch[0].size()[2]
    width = batch[0].size()[3]
    num_of_pixels = data_len * batch_size * height * width

    # compute mean for each channel
    mean = np.zeros(num_channel)

    for counter, batch in enumerate(train_dataloader, 0):
        print("Calculating mean {} out of {}.".format(counter, data_len)) 
        for i in range(batch_size):
            for j in range(num_channel):
                mean[j] += batch[0][i][j].sum()

    mean = mean / num_of_pixels

    # compute standard deviation for each channel 
    std = np.zeros(num_channel)
    for counter, batch in enumerate(train_dataloader, 0):
        print("Calculating std {} out of {}.".format(counter, data_len)) 
        for i in range(batch_size):
            for j in range(num_channel):
                std[j] += ((batch[0][i][j] - mean[j]).pow(2)).sum()
    std = np.sqrt(std / num_of_pixels)

    return mean, std





if __name__ == '__main__':
    # python .\data_load.py --cal_mean_std 0 --vis 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_mean_std", type=int, default=1)
    parser.add_argument("--vis", type=int, default=1)
    args = parser.parse_args()

    # calculate mean and std of the trainning set
    if args.cal_mean_std == 1:
        train_dataset = StarDataSet(split='train')
        train_dataloader = data.DataLoader(train_dataset, batch_size=1,
                                            shuffle=False, num_workers=2, 
                                            drop_last=False)
        mean, std = compute_mean_std(train_dataloader)
    else:
        mean = [44.1619381]
        std =  [60.98225565]
    print("means of training set are {}".format(mean))
    print("standard deviations of training set are {}".format(std))


    # normalize data
    train_dataset = StarDataSet(split='train', norm=True, mean=mean, std=std)
    train_dataloader = data.DataLoader(train_dataset, batch_size=1,
                                        shuffle=True, num_workers=2, 
                                        drop_last=True)


    # also load val and test set. use same parameters as training set for normalization
    val_dataset = StarDataSet(split='val', norm=True, mean=mean, std=std)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1,
                                        shuffle=True, num_workers=2, 
                                        drop_last=True)
    test_dataset = StarDataSet(split='test', norm=True, mean=mean, std=std)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1,
                                        shuffle=True, num_workers=2, 
                                        drop_last=True)
    batch_val = next(iter(val_dataloader))
    batch_test = next(iter(test_dataloader))          
    print( "the dimension of a single raw images batch of validation set is {}".format(batch_val[0].size()) )
    print( "the dimension of a single dist map batch of test set is {}".format(batch_test[1].size()) )          
    print( "the dimension of a single seg map batch of test set is {}".format(batch_test[2].size()) )                          
                

    # get a single batch from dataloader 
    # iter() returns an iterator
    # next() get the first iteration. Running next() again will get the second item of the iterator
    # https://stackoverflow.com/questions/62549990/what-does-next-and-iter-do-in-pytorchs-dataloader
    batch = next(iter(train_dataloader)) 
    images, dist_map, seg_map, centroid_real  = batch
    print( "the dimension of a single centroid batch of training set is {}".format(batch[3].size()) )


    if args.vis == 1:
        # get a fake groundtruth prediction 
        _, _, _, centroid_est = batch_val 
        # visualization
        visualization(None, dist_map[0], seg_map[0], centroid_real[0], centroid_est[0], centroid_est[0], images[0][0], 6/1000)
