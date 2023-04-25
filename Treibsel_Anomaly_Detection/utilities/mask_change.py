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