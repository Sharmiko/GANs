import torch.nn as nn

# Generator class
class Generator(nn.Module):
    
    def __init__(self):
        
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(100, 3136)
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5,
                               stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5,
                               stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,
                               stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5,
                               stride=1, padding=2)
        
        self.batch_norm_fc1 = nn.BatchNorm1d(3136)
        self.batch_norm_conv1 = nn.BatchNorm2d(128)
        self.batch_norm_conv2 = nn.BatchNorm2d(64)
        self.batch_norm_conv3 = nn.BatchNorm2d(64)
        
        self.out = nn.Tanh()
        
    def forward(self, t):
        
        t = t
        
        t = self.fc1(t)
        t = self.batch_norm_fc1(t)
        t = self.relu(t)
        
        t = t.view(-1, 64, 7, 7)

        t = self.upsample(t)
        
        t = self.conv1(t)
        t = self.batch_norm_conv1(t)
        t = self.relu(t)
        
        t = self.upsample(t)
        
        t = self.conv2(t)
        t = self.batch_norm_conv2(t)
        t = self.relu(t)
        
        t = self.conv3(t)
        t = self.batch_norm_conv3(t)
        t = self.relu(t)
        
        t = self.conv4(t)
        t = self.relu(t)
        
        t = self.out(t)
        
        return t        


# Discriminator class
class Discriminator(nn.Module):
    
    def __init__(self):
        
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5,
                               stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,
                               stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1,
                               stride=1, padding=0)
        
        self.relu = nn.ReLU()
        self.droput = nn.Dropout(p=0.4)
        
        self.out = nn.Linear(4 * 4 * 128, 1)
        
    def forward(self, t):
        
        t = t
        
        t = self.conv1(t)
        t = self.relu(t)
        t = self.droput(t)
        
        t = self.conv2(t)
        t = self.relu(t)
        t = self.droput(t)
        
        t = self.conv3(t)
        t = self.relu(t)
        t = self.droput(t)
        
        t = self.conv4(t)
        t = self.relu(t)
        t = self.droput(t)
        
        t = t.view(-1, 4 * 4 * 128)
        
        t = self.out(t)
        
        return t