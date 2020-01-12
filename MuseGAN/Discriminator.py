import torch.nn as nn

class CustomConv3d(nn.Module):
    
    def __init__(self, input_size, output_size, kernel_size, padding,
                 strides, activation='relu'):
        
        super(CustomConv3d, self).__init__()
        
        self.conv = nn.Conv3d(in_channels=input_size, out_channels=output_size, 
                              kernel_size=kernel_size, padding=padding,
                              stride=strides)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        
        
    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) Conv Layer
        t = self.conv(t)
        
        # (3) Activation Function
        t = self.activation(t)
        
        return t
    
    
class Discriminator(nn.Module):
    
    def __init__(self, n_bars):
        
        super(Discriminator, self).__init__()
        self.n_bars = n_bars
        
        self.conv1 = CustomConv3d(input_size=4, output_size=128,
                                  kernel_size=(2, 1, 1), padding=(0, 0, 0),
                                  strides=(1, 1, 1), activation='lrelu')
        self.conv2 = CustomConv3d(input_size=128, output_size=128,
                                  kernel_size=(self.n_bars - 1, 1, 1), padding=(0, 0, 0),
                                  strides=(1, 1, 1), activation='lrelu')
        self.conv3 = CustomConv3d(input_size=128, output_size=128,
                                  kernel_size=(1, 1, 12), padding=(0, 0, 0),
                                  strides=(1, 1, 12), activation='lrelu')
        self.conv4 = CustomConv3d(input_size=128, output_size=128,
                                  kernel_size=(1, 1, 7), padding=(0, 0, 0),
                                  strides=(1, 1, 7), activation='lrelu')
        self.conv5 = CustomConv3d(input_size=128, output_size=128,
                                  kernel_size=(1, 2, 1), padding=(0, 0, 0),
                                  strides=(1, 2, 1, ), activation='lrelu')
        self.conv6 = CustomConv3d(input_size=128, output_size=128,
                                  kernel_size=(1, 2, 1), padding=(0, 0, 0),
                                  strides=(1, 2, 1), activation='lrelu')
        self.conv7 = CustomConv3d(input_size=128, output_size=256,
                                  kernel_size=(1, 4, 1), padding=(0, 1, 0),
                                  strides=(1, 2, 1), activation='lrelu')
        self.conv8 = CustomConv3d(input_size=256, output_size=512,
                                  kernel_size=(1, 3, 1), padding=(0, 1, 0),
                                  strides=(1, 2, 1), activation='lrelu') 
        
        self.fc = nn.Linear(512, 1024)
        self.leaky_relu = nn.LeakyReLU()
        self.out = nn.Linear(1024, 1)
    
    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) Reshape Layer
        t = t.view(-1, 4, 2, 16, 84)
        
        # (3) First Hidden Conv Layer
        t = self.conv1(t)

        # (4) Second Hidden Conv Layer
        t = self.conv2(t)

        # (5) Third Hidden Conv Layer
        t = self.conv3(t)
       
        # (6) Forth Hidden Conv Layer
        t = self.conv4(t)
   
        # (7) Fifth Hidden Conv Layer
        t = self.conv5(t)
        
        # (8) Sixth Hidden Conv Layer
        t = self.conv6(t)

        # (9) Seventh Hidden Conv Layer
        t = self.conv7(t)

        # (10) Eigth Hidden Conv Layer
        t = self.conv8(t)
        
        # (11) Reshape Layer
        t = t.view((-1, 512))
        
        # (12) Fully Connected Layer
        t = self.fc(t)
        
        # (13) Leaky ReLU Activation Function
        t = self.leaky_relu(t)
        
        # (14) Output Fully Connected Layer
        t = self.out(t)
        
        return t