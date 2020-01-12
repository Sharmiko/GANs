import torch
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader 

from music_data import MusicData
from Generator import Generator
from Discriminator import Discriminator 

from tqdm import tqdm

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
N_BARS = 2
N_STEPS_PER_BAR = 16
N_PITCHES = 84
N_TRACKS = 4
Z_DIM = 32

# load data
music_data = MusicData('../data/Jsb16thSeparated.npz', N_BARS, N_STEPS_PER_BAR)

data_binary = music_data.data_binary
data_ints = music_data.data_ints
raw_data = music_data.data

data_binary = np.squeeze(data_binary)
data_binary = torch.from_numpy(data_binary)

# create data loader
data_loader = DataLoader(data_binary, batch_size=BATCH_SIZE)

# create models
generator = Generator(Z_DIM, N_BARS, N_STEPS_PER_BAR, N_PITCHES, N_TRACKS)
generator.to(device)

discriminator = Discriminator(N_BARS)
discriminator.to(device)

# create optmizer and loss function

optimizerG = optim.Adam(generator.parameters())
optimizerD = optim.Adam(discriminator.parameters())

def compute_gradient_penalty(D, y_true, y_pred):
    """
    Calculate gradient penalty for WGAN-GP
    """
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    # random weight for interpolation
    alpha = Tensor(np.random.random((y_true.size()[0], 1, 1, 1, 1)))
    # interpolation between real and fake images
    interpolates = (alpha * y_true + ((1 - alpha) * y_pred)).requires_grad_(True)
    D_interpolates = D(interpolates)
    
    fake = torch.cuda.FloatTensor(y_true.shape[0], 1).fill_(1.0)
    
    # get gradient interpolates
    gradients = torch.autograd.grad(
            outputs=D_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
            )[0]
   
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) -1) ** 2).mean()
    
    return gradient_penalty
    
def wasserstein(y_true, y_pred):
    return -torch.mean(y_true * y_pred)

lambda_gp = 10
    
EPOCHS = 600
    
real = torch.ones((BATCH_SIZE, 1), dtype=torch.float).to(device)
fake = -torch.ones((BATCH_SIZE, 1), dtype=torch.float).to(device)
dummy = torch.zeros((BATCH_SIZE, 1), dtype=torch.float).to(device)

generated_music = []

for epoch in tqdm(range(1, EPOCHS + 1)):
    
    for i, real_data in enumerate(data_loader, 0):
        
        real_data = real_data.float().to(device)
        
        BATCH_SIZE = real_data.size()[0]
        
        chords_noise = torch.randn((BATCH_SIZE, Z_DIM)).to(device)
        style_noise = torch.randn((BATCH_SIZE, Z_DIM)).to(device)
        melody_noise = torch.randn((BATCH_SIZE, N_TRACKS, Z_DIM)).to(device)
        groove_noise = torch.randn((BATCH_SIZE, N_TRACKS, Z_DIM)).to(device)

        
        #######################
        # Train Discriminator #
        #######################
        
        optimizerG.zero_grad()
        
        fake_data = generator(chords_noise, style_noise, 
                              melody_noise, groove_noise).detach()
        # real music
        real_valid = discriminator(real_data)
        # fake music
        fake_valid = discriminator(fake_data)
        # compute gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data)
        
        # Wasserstein loss
        d_loss = wasserstein(real_data, fake_data)
        d_loss += lambda_gp * gradient_penalty
        
        d_loss.backward()
        optimizerD.step()

        
        # train generator every 5 iterations
        if i % 5 == 0:
            
            ###################
            # Train Generator #
            ###################
            
            optimizerG.zero_grad()
            
            # generated music
            gen_music = generator(chords_noise, style_noise,
                                   melody_noise, groove_noise)
            # generator loss
            g_loss = -torch.mean(discriminator(gen_music))
            
            g_loss.backward()
            optimizerG.step()
            
    if epoch % 25 == 0:
        generated_music.append(gen_music)
    
    print("Epoch {}/{}; Loss D: {}, Loss G:{}".format(
        epoch, EPOCHS, d_loss.item(), g_loss.item()))
        
            
