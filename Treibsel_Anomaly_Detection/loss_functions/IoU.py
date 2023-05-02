import torch
import torch.nn as nn
import torch.nn.functional as F
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):        
        #comment out if model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs) 
        inputs = inputs.view(-1)
        targets = targets.view(-1) 
        IoU = torch.logical_and(inputs, targets).sum() / torch.logical_or(inputs, targets).sum()     
                
        return 1-IoU # Alternatively 1-...