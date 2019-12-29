import torch.nn as nn

class DownSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(DownSample, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=2, padding=2)
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
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=4, stride=2,
                                                padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        
    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) Hidden Conv Transpose Layer
        t = self.conv_transpose(t)

        # (3) Instance Normalization
        t = self.instance_norm(t)
        
        # (4) ReLU Activation Function
        t = self.relu(t)
        
        return t
    
class ResidualBlock(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        
        self.padding = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_features, 
                               out_channels=out_features, 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_features, 
                               out_channels=out_features,
                               kernel_size=3, padding=1)
        self.instance_norm = nn.InstanceNorm2d(32)
        self.relu = nn.ReLU()
        
    def forward(self, t):
        
        # (1) Input Layer
        input_t = t
        
        # (2) First Hidden Conv Layer
        t = self.conv1(t)
        t = self.instance_norm(t)
        t = self.relu(t)
        
        # (3) Second Hidden Conv Layer
        t = self.conv2(t)
        t = self.instance_norm(t)
        
        return input_t + t
        
    
class ResNetGenerator(nn.Module):
    
    def __init__(self, num_residuals=9):
        
        super(ResNetGenerator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7)
        self.relu = nn.ReLU()
        
        self.downsample1 = DownSample(32, 32 * 2)
        self.downsample2 = DownSample(32 * 2, 32 * 4)
        
        self.residuals = []
        for i in range(num_residuals):
            self.residuals.append(ResidualBlock(32 * 4, 32 * 4))
            
        self.upsample1 = UpSample(32 * 4, 32 * 2)
        self.upsample2 = UpSample(32 * 2, 32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7,
                               padding=3)
        self.tanh = nn.Tanh()


    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) Hidden Conv Layer
        t = self.conv1(t)
        t = self.relu(t)
        
        # (3) First Downsample Layer
        t = self.downsample1(t)
        
        # (4) Second Downsample Layer
        t = self.downsample2(t)
        
        # (5) Residual Blocks
        for residual in self.residuals:
            t = residual(t)
        
        # (6) First Upsample Layer
        t = self.upsample1(t)
        
        # (7) Second Upsample Layer
        t = self.upsample2(t)
        
        # (8) Hidden Conv Layer
        t = self.conv2(t)
        t = self.tanh(t)
        
        return t