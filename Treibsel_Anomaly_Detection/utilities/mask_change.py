#!/usr/bin/env python
#######################################################
### mask_change.py: What is the task of the code. ( in Short )
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

import os
from PIL import Image
import numpy as np

folder = "./Treibsel_Anomaly_Detection/PyTorch_Playground/UNET_example/data/val_masks/"
mask_end = "_Maske"
end = ".jpg"


for file in os.listdir(folder):
    with Image.open(folder + file).convert("L") as f:
        print(file)
        im = np.array(f)
        im[im > 200] = 0.0
        im[im > 90] = 255.0
        Image.fromarray(im).save(folder + file)