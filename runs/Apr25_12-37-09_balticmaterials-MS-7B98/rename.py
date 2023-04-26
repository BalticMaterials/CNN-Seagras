import os

path = "./runs/Apr25_12-37-09_balticmaterials-MS-7B98/"
for x in os.listdir(path):
    if " " in x:
        os.rename(path + x, path + x.replace(" ", "_"))
        # print(F"FILE TO REPLACE{x}")
    else:
        print(x)