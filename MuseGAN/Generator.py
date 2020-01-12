import torch
import torch.nn as nn

device = ("cuda:0" if torch.cuda.is_available() else "cpu")    

class CustomConv2dTranspose(nn.Module):
    
    def __init__(self, input_size, output_size, kernel_size, padding,
                 strides, activation='relu', bn=True):
        
        super(CustomConv2dTranspose, self).__init__()
        self.bn = bn
        
        self.conv_t = nn.ConvTranspose2d(in_channels=input_size, out_channels=output_size,
                                         kernel_size=kernel_size, padding=padding,
                                         stride=strides)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        
        self.batch_norm = nn.BatchNorm2d(output_size)
            
            
    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) Conv Transpose Layer
        t = self.conv_t(t)
        
        # (3) Optional: Batch Norm Layer
        if self.bn == True:
            t = self.batch_norm(t)
        
        # (4) Activation: ReLU or LeakyReLU
        t = self.activation(t)
        
        return t
        

class TemporalNetwork(nn.Module):
    
    def __init__(self, dim, n_bars):
        
        super(TemporalNetwork, self).__init__()
        self.dim = dim
        self.n_bars = n_bars
        
        self.conv_t1 = CustomConv2dTranspose(self.dim, 1024, kernel_size=(2, 1),
                                             padding=(0, 0), strides=(1, 1))
        self.conv_t2 = CustomConv2dTranspose(1024, self.dim, kernel_size=(self.n_bars-1, 1),
                                             padding=(0, 0), strides=(1, 1))        
        
    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) Reshape Layer
        t = t.view(-1, self.dim, 1, 1)
        
        # (3) Conv Transpose Layer
        t = self.conv_t1(t)
        
        # (4) Conv Transpose Layer
        t = self.conv_t2(t)
       
        # (5) Reshape Layer
        t = t.view(-1, self.n_bars, self.dim)
        
        return t
        
    
class BarGenerator(nn.Module):
    
    def __init__(self, dim, steps, pitches):
        
        super(BarGenerator, self).__init__()
        self.dim = dim
        self.steps = steps
        self.pitches = pitches
        
        self.fc = nn.Linear(self.dim * 4, 1024)
        self.batch_norm = nn.BatchNorm1d(1024, momentum=0.9)
        self.relu = nn.ReLU()
        
        self.conv_t1 = CustomConv2dTranspose(512, 512, kernel_size=(2, 1),
                                             padding=(0, 0), strides=(2, 1))
        self.conv_t2 = CustomConv2dTranspose(512, 256, kernel_size=(2, 1),
                                             padding=(0, 0), strides=(2, 1))
        self.conv_t3 = CustomConv2dTranspose(256, 256, kernel_size=(2, 1),
                                             padding=(0, 0), strides=(2, 1))
        self.conv_t4 = CustomConv2dTranspose(256, 256, kernel_size=(1, 7),
                                             padding=(0, 0), strides=(1, 7))
        self.conv_t5 = CustomConv2dTranspose(256, 1, kernel_size=(1, 12),
                                             padding=(0, 0), strides=(1, 12), bn=False)
    def forward(self, t):
        
        # (1) Input Layer
        t = t

        # (2) Fully Connected Layer
        t = self.fc(t)
        
        # (3) Batch Normalization
        t = self.batch_norm(t)
        
        # (4) ReLU Activation Function
        t = self.relu(t)

        # (5) Reshape Layer
        t = t.view(-1, 512, 2, 1,)
        
        # (6) Conv Transpose Layer
        t = self.conv_t1(t)
        
        # (7) Conv Transpose Layer
        t = self.conv_t2(t)
        
        # (8) Conv Transpose Layer
        t = self.conv_t3(t)
        
        # (9) Conv Transpose Layer
        t = self.conv_t4(t)
      
        # (10) Conv Transpose Layer
        t = self.conv_t5(t)
        
        # (11) Reshape Layer
        t = t.view(-1, 1, self.steps, self.pitches, 1)
        
        return t
        
class Generator(nn.Module):
    
    def __init__(self, dim, n_bars, n_steps, n_pitches, n_tracks):
        
        super(Generator, self).__init__()
        self.dim = dim
        self.n_bars = n_bars
        self.n_steps = n_steps
        self.n_pitches = n_pitches
        self.n_tracks = n_tracks
        
        self.chords_temp = TemporalNetwork(dim, 4)
        
        self.melody_temp = [None] * self.n_tracks
        for i in range(self.n_tracks):
            self.melody_temp[i] = TemporalNetwork(dim, 4).to(device)
        
        self.barGen = [None] * self.n_tracks
        for i in range(self.n_tracks):
            self.barGen[i] = BarGenerator(self.dim, self.n_steps, self.n_pitches).to(device)
        
    def forward(self, chords, style, melody, groove):
        
        # (1) Inputs
        chords = chords
        style = style
        melody = melody
        groove = groove
        
        # (2) Chords -> Temporal Network
        chords_over_time = self.chords_temp(chords)
        
        # (3) Melody -> Temporal Network
        melody_over_time = [None] * self.n_tracks
                
        for i in range(self.n_tracks):
            melody_track = melody[:, i, :]
            melody_track = self.melody_temp[i](melody_track)
            melody_over_time[i] = melody_track
            
        # (4) Output for every track and bar
        bar_outputs = [None] * self.n_bars
        for bar in range(self.n_bars):
            track_output = [None] * self.n_tracks
            
            c = chords_over_time[:, bar, :]
            s = style
            
            for track in range(self.n_tracks):
                
                m = melody_over_time[track][:, bar, :]
                g = groove[:, track, :]
                
                # Concatenate: Chords - Style - Melody - Groove
                z_input = torch.cat([c, s, m, g], axis=1)
                
                track_output[track] = self.barGen[track](z_input)
            
            # Concatenate Tracks
            bar_outputs[bar] = torch.cat(track_output, axis=-1)
        
        # Concatenate Bars
        generator_output = torch.cat(bar_outputs, axis=1)
        
        return generator_output