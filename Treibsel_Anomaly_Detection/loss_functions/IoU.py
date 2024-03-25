#!/usr/bin/env python
#######################################################
### IoU.py: What is the task of the code. ( in Short )
# General Text and Notice
#######################################################

"""
NumberList holds a sequence of numbers, and defines several statistical
operations (mean, stdev, etc.) FrequencyDistribution
holds a mapping from items (not necessarily numbers)
to counts, and defines operations such as Shannon
entropy and frequency normalization.
"""

__date__ = "18.03.2024"
__author__ = "Tjark Ziehm"
__copyright__ = "Copyright 2024, BalticMaterials"
__credits__ = ["Tjark Ziehm"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Tjark Ziehm"
__email__ = "kontakt@balticmaterials.de"
__status__ = "Development"

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