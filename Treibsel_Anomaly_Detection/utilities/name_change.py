import os

folder = "./Treibsel_Anomaly_Detection/PyTorch_Playground/UNET_example/data/"
mask_end = "_Maske"
end = ".jpg"
index = 1


files = sorted(os.listdir(folder), key=len)
for file in files:
    if end in file:
           old = folder + file
           new = folder + "masks/" + file
           os.rename(old, new)