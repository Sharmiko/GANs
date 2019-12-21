import torch
import torch.nn as nn

class DownSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(DownSample, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=4, stride=2, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, t):
        
        t = t
        
        t = self.conv1(t)
        t = self.instance_norm(t)
        t = self.relu(t)
        
        return t
    
    
class UpSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(UpSample, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=5, stride=1, padding=2)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        
    def forward(self, t, skip_layer):
        
        t = t
        
        t = self.upsample(t)        
        t = self.conv1(t)
        t = self.instance_norm(t)
        t = self.relu(t)
        
        t = torch.cat((t, skip_layer), dim=0)
        
        return t
    
    
class UnetGenerator(nn.Module):
    
    def __init__(self):
        
        super(UnetGenerator, self).__init__()
        
        self.downsample1 = DownSample(3, 32)
        self.downsample2 = DownSample(32, 64)
        self.downsample3 = DownSample(64, 128)
        self.downsample4 = DownSample(128, 256)
        
        self.upsample1 = UpSample(256, 128)
        self.upsample2 = UpSample(128, 64)
        self.upsample3 = UpSample(64, 32)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True)
        
        self.out = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                             padding=2)
        
        
    def forward(self, t):
        
        t = t

        d1 = self.downsample1(t)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
    
        u1 = self.upsample1(d4, d3)
        u2 = self.upsample2(u1, d2)
        u3 = self.upsample3(u2, d1)
        u4 = self.upsample4(u3)
        
        t = self.out(u4)

        return t
        