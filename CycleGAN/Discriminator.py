import torch.nn as nn

class Conv4(nn.Module):
    
    def __init__(self, in_channels, out_channels, norm=True):
        
        super(Conv4, self).__init__()
        self.norm = norm
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=5, stride=2, padding=2)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, t):
        
        t = t
        
        t = self.conv(t)
        if self.norm:
            t = self.instance_norm(t)
        t = self.leaky_relu(t)
        
        return t


class Discriminator(nn.Module):
    
    def __init__(self):
        
        super(Discriminator, self).__init__()
        
        self.conv1 = Conv4(3, 32, norm=False)
        self.conv2 = Conv4(32, 64)
        self.conv3 = Conv4(64, 128)
        self.conv4 = Conv4(128, 256)
        
        self.out = nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=2)
        
    def forward(self, t):
        
        t = t
        
        t = self.conv1(t)
        t = self.conv2(t)
        t = self.conv3(t)
        t = self.conv4(t)
        
        t = self.out(t)

        return t
        