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
        
        # (1) Input Layer
        t = t
        
        # (2) Hidden Conv Layer
        t = self.conv1(t)
        
        # (3) Instance Normalization
        t = self.instance_norm(t)
        
        # (4) ReLU Activaion Function
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
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        
    def forward(self, t, skip_layer):
        
        # (1) Input Layer
        t = t
 
        # (2) Upsample Layer
        t = self.upsample(t)
                
        # (3) Hidden Conv Layer
        t = self.conv1(t)
         
        # (4) Instance Normalization Layer
        t = self.instance_norm(t)
         
        # (5) ReLU Activation Function
        t = self.relu(t)

        # (6) Concatenation Layer -> (Skip Connection)
        t = torch.cat((t, skip_layer), dim=1)
        
        # (7) Hidden Conv Layer
        t = self.conv2(t)

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
        
        self.out = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5,
                             padding=2)
        
        
    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) First Downsample Layer
        d1 = self.downsample1(t)
        
        # (3) Second Downsample Layer
        d2 = self.downsample2(d1)

        # (4) Third Downsample Layer
        d3 = self.downsample3(d2)
        
        # (5) Forth Downsample Layer
        d4 = self.downsample4(d3)

        # (6) First Upsample Layer
        u1 = self.upsample1(d4, d3)
        
        # (7) Second Upsample Layer
        u2 = self.upsample2(u1, d2)
        
        # (8) Third Upsample Layer
        u3 = self.upsample3(u2, d1)

        # (9) Forth Upsample Layer
        u4 = self.upsample4(u3)
        
        # (10) Hidden Output Conv Layer
        t = self.out(u4)

        return t
        
    
class ResNetGenerator(nn.Module):
    pass


class Residual(nn.Module):
    
    def __init__(self):
        super(Residual, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3,
                               padding=1)
        self.instance_norm = nn.InstanceNorm1d(3)
        self.relu = nn.ReLU()
        
    def forward(self, t, skip_layer):
        pass





