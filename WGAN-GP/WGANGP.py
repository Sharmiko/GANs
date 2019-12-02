import torch.nn as nn

# Generator class
class Generator(nn.Module):
    
    def __init__(self):
        
        super(Generator, self).__init__()
        
        # Fulle Connected Layer
        self.fc = nn.Linear(100, 3136)
        
        # Upsample Layer
        self.upsample = nn.Upsample(scale_factor=2)
        
        # ReLU Activation function
        self.relu = nn.ReLU()
        
        # Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5,
                               stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5,
                               stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,
                               stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5,
                               stride=1, padding=2)
        
        # Batch Normalization Layer
        self.batch_norm_fc = nn.BatchNorm1d(3136)
        self.batch_norm_conv1 = nn.BatchNorm2d(128)
        self.batch_norm_conv2 = nn.BatchNorm2d(64)
        self.batch_norm_conv3 = nn.BatchNorm2d(64)
        
        # Output Layer
        self.out = nn.Tanh()
        
    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) Hidden Fully Connected Layer
        t = self.fc(t)
        t = self.batch_norm_fc(t)
        t = self.relu(t)
        
        # (3) Reshape Layer -> Convert flat tensor into matrix
        t = t.view(-1, 64, 7, 7)

        # (4) Upsample Layer -> Scale factor = 2
        t = self.upsample(t)
        
        # (5) First Hidden Conv Layer
        t = self.conv1(t)
        t = self.batch_norm_conv1(t)
        t = self.relu(t)
        
        # (6) Upsample Layer -> Scale factor = 2
        t = self.upsample(t)
        
        # (7) Second Hidden Conv Layer
        t = self.conv2(t)
        t = self.batch_norm_conv2(t)
        t = self.relu(t)
        
        # (8) Third Hidden Conv Layer
        t = self.conv3(t)
        t = self.batch_norm_conv3(t)
        t = self.relu(t)
        
        # (9) Forth Hidden Conv Layer
        t = self.conv4(t)
        t = self.relu(t)
        
        # (10) Output Layer
        t = self.out(t)
        
        return t        


# Discriminator class
class Discriminator(nn.Module):
    
    def __init__(self):
        
        super(Discriminator, self).__init__()
        
        # Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5,
                               stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,
                               stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1,
                               stride=1, padding=0)
        
        # ReLU Activation function
        self.relu = nn.ReLU()
        
        # Dropout Layer
        self.droput = nn.Dropout(p=0.4)
        
        # Output Layer
        self.out = nn.Linear(4 * 4 * 128, 1)
        
    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) First Hidden Conv Layer
        t = self.conv1(t)
        t = self.relu(t)
        t = self.droput(t)
        
        # (3) Second Hidden Conv Layer
        t = self.conv2(t)
        t = self.relu(t)
        t = self.droput(t)
        
        # (4) Third Hidden Conv Layer
        t = self.conv3(t)
        t = self.relu(t)
        t = self.droput(t)
        
        # (5) Forth Hidden Conv Layer
        t = self.conv4(t)
        t = self.relu(t)
        t = self.droput(t)
        
        # (6) Flatten Layer
        t = t.view(-1, 4 * 4 * 128)
        
        # (7) Output Layer
        t = self.out(t)
        
        return t