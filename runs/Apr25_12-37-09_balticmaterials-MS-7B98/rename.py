import os

for x in os.listdir("./"):
    if ":" in x:
        os.rename(x, x.replace(":", "_"))
    else:
        print(x)