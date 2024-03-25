#!/usr/bin/env python
#######################################################
### random_val_spli.py: What is the task of the code. ( in Short )
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

import random
import os

root = "./Treibsel_Anomaly_Detection/PyTorch_Playground/UNET_example/data/"
train_im = "train_images/"
train_masks = "train_masks/"
val_im = "val_images/"
val_masks = "val_masks/"
mask_end = "_Maske"
end = ".jpg"

val_num = random.sample(range(1, 110), 28) # 75:25 split
for num in val_num:
    #image
    old = root + train_im + str(num) + end
    new = root + val_im + str(num) + end
    os.rename(old, new)

    #mask
    old = root + train_masks + str(num) + mask_end + end
    new = root + val_masks + str(num) + mask_end + end
    os.rename(old, new)
