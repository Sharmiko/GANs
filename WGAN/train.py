import numpy as np
import torch
import torch.utils.data as utils
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt

from tqdm import tqdm

from WGAN import Generator, Discriminator

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

# Load training data from numpy array and convert it to tensor
train_data = torch.from_numpy(np.load('../data/train_camel.npy'))

train_data = train_data.float()
train_data /= 255.0

# Create torch Dataset object
train_data = utils.TensorDataset(train_data)


# Create DataLoader object
trainloader = utils.DataLoader(train_data, batch_size=32)

# Create model

generator = Generator()
discriminator = Discriminator()

generator.to(device)
discriminator.to(device)

# Create optimizer and loss function

optimizerG = optim.RMSprop(generator.parameters(), lr=0.0002)
optimizerD = optim.RMSprop(discriminator.parameters(), lr=0.0002)

# real and fake labels for training
real_label = 1
fake_label = -1

# create latent vector for generator
noise = torch.randn(32, 100, 1, 1, device=device)

# Training loop

img_list=[]
iters = 0

EPOCHS = 5

for epoch in tqdm(range(EPOCHS)):
    
    for i, data in enumerate(trainloader, 0):
        
        #######################
        # Train Discriminator #
        #######################
        
        # format batch
        images = data[0].to(device)
        images = images.view(-1, 1, 28, 28)
        real_imgs = Variable(images.type(torch.Tensor)).to(device)
        b_size = len(images)
        
        real = torch.full((b_size,), real_label, device=device).to(device)
        fake = torch.full((b_size,), fake_label, device=device).to(device)
        
        optimizerD.zero_grad()
        
        # sample noise for generator
        z = Variable(torch.Tensor(np.random.normal(0, 1, (b_size, 100)))).to(device)
        
        # generate fake images
        fake_imgs = generator(z).detach()
        
        # Wasserstein loss
        d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        
        d_loss.backward()
        optimizerD.step()
        
        # Clip weights of discriminator
        for param in discriminator.parameters():
            param.data.clamp_(-0.01, 0.01)
            
        # train generator every 5 iterations
        if i % 5 == 0:
            
            ###################
            # Train Generator #
            ###################
            
            optimizerG.zero_grad()
            
            # generate images
            gen_images = generator(z)
            
            # generator loss
            g_loss = -torch.mean(discriminator(gen_images))
            
            g_loss.backward()
            optimizerG.step()
            
        
        if i % 500 == 0:
            img_list.append(gen_images)
      
    print("Epoch {} / {}".format(epoch + 1, EPOCHS))
    
print("G loss: {} \t D loss: {}".format(g_loss, d_loss))    

    
# Save model state
torch.save(generator.state_dict(), "../model-states/WGen-[{}-Epochs]".format(EPOCHS))
torch.save(discriminator.state_dict(), "../model-states/WDis-[{}-Epochs]".format(EPOCHS))

# Load generator and discriminator
generator = Generator()
generator.load_state_dict(torch.load('../model-states/WGen-[5-Epochs]'))

discriminator = Discriminator()
discriminator.load_state_dict(torch.load('../model-states/WDis-[5-Epochs]'))
        
# Helper function to show generated images
def imshow(inputs):
    # create subplots
    row = len(inputs) // 6
    fig, axes = plt.subplots(row, 6, figsize=(14, 6))
    j = 0
    # plot original images
    for k in range(row):
        for i in range(6):
            axes[k, i].imshow(inputs[j].view(28,28).numpy(), cmap="gray")
            j += 1
    plt.show()
    

# show generated images from last epoch
imshow(img_list[34].detach().cpu())    
        