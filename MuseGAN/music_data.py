import os
import numpy as np

class MusicData():
    
    def __init__(self, file_path, n_bars, n_steps):
        self.file_path = os.path.join(file_path)
        self.n_bars = n_bars
        self.n_steps = n_steps
        
        with np.load(self.file_path, encoding='bytes', allow_pickle=True) as f:
            self.data = f['train']

        self.data_ints = self.data_ints(self.data)
        n_songs = self.data_ints.shape[0]
        n_tracks = self.data_ints.shape[2]

        self.data_ints = self.data_ints.reshape([n_songs, self.n_bars, self.n_steps, n_tracks])
        
        MAX_NOTE = 83
        
        NaNs = np.isnan(self.data_ints)
        self.data_ints[NaNs] = MAX_NOTE + 1
        MAX_NOTE = MAX_NOTE + 1
        
        self.data_ints = self.data_ints.astype(int)
        
        num_classes = MAX_NOTE + 1
        
        self.data_binary = np.eye(num_classes)[self.data_ints]
        self.data_binary[self.data_binary == 0] = -1
        self.data_binary = np.delete(self.data_binary, MAX_NOTE, -1)
        
        self.data_binary = self.data_binary.transpose([0, 1, 2, 4, 3])
        
        
    def data_ints(self, data):
        data_ints = []
        for x in data:
            counter = 0
            cont = True
            while cont:
                if not np.any(np.isnan(x[counter:(counter+4)])):
                    cont = False
                else:
                    counter += 4
            
            if self.n_bars * self.n_steps < x.shape[0]:
                data_ints.append(x[counter:(counter + (self.n_bars * self.n_steps)), :])
        return np.array(data_ints)
        
                
        
        