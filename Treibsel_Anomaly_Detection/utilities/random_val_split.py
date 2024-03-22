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
