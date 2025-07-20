import torch
import torch.nn as nn
from data_load import StarDataSet
from torch.utils import data
import torchvision


# global variable
device = torch.device("cuda:0")

#--------------------- loss function --------------------- #
class mse_loss(nn.Module):
    def __init__(self, mode):
        """
            @param mode: 'mean' or 'sum'
        """
        super(mse_loss, self).__init__()
        self.mode = mode
        if self.mode == 'sum':
            self.mse = nn.MSELoss(reduction='sum')
        else:
            self.mse = nn.MSELoss(reduction='mean')

    def forward(self, prediction, target):
        loss = self.mse(prediction, target)
        if self.mode == 'sum':
            loss /= (len(prediction))
        return loss


class bce_loss(nn.Module): # binary cross entropy
    def __init__(self):
        super(bce_loss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')
    
    def forward(self, prediction, target):
        loss = self.bce_loss(prediction, target)
        return loss


if __name__ == '__main__':

    # sanity check 
    predict = torch.randn(5, 1, 480, 640)
    target = torch.randn(5, 1, 480, 640)

    criterion = mse_loss()

    loss = criterion(torch.sigmoid(predict), torch.sigmoid(target))

    print(loss)