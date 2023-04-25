import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as TF

class DoubleConv(nn.Module): # On every steps of the unet two convolutions are applied, before moving down
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            # padding=1 makes it a same convolution -> Input height and width is the same after convolution. Padding adds a 0-pixel to the edge of input
            # Orig. Paper has zero padding
            nn.BatchNorm2d(out_channels), # bias=False, because BatchNorm2d is used. The bias would be canceled out by the batch norm
            # Normalizes the batches so that mean=0, variance=1, which makes the bias redundant
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
       
class UNET(nn.Module):
    def __init__(self,
                  in_channels=3, #RGB, original paper was 1
                    out_channels=1, # Original Paper specifies 2 Channels, but for binary segmentation only 1 is necessary!
                      features=[64,128,256,512],): # features are the different features in the convolutional layer?
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # downward pooling 2x2, stride = 2 for greater downsampling
        # Problem: 161 x 161 -> 80 x 80,  upsample output later: 160 x 160 !

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d( # Up conv 2x2
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) # Bottom line from 512 (features[-1]) to 1024
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) # Last convolution to decrease the feature/channel size to num_classes, not the image size

    def forward(self, x):
        skip_connections = []

        for down in self.downs: # Everything down, except bottom line
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1] # list reverse

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] # because of the "double" step

            if x.shape != skip_connection.shape: # Addresses shape bias from pooling 
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # concatination of the skip connection tensor with the upwards tensor
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((30, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()