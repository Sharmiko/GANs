import numpy as np
import torch
import torch.utils.data as utils
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt

from tqdm import tqdm

from WGANGP import Generator, Discriminator

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

optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# loss weight for GP
lambda_gp = 10

def compute_gradient_penalty(D, real_img, fake_img):
    """
    Calculate gradient penalty for WGAN-GP
    """
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    # random weight for interpolation
    alpha = Tensor(np.random.random((real_img.size(0), 1, 1, 1)))
    
    # interpolation between real and fake images
    interpolates = (alpha * real_img + ((1 - alpha) * fake_img)).requires_grad_(True)
    D_interpolates = D(interpolates)
    
    fake = torch.cuda.FloatTensor(real_img.shape[0], 1).fill_(1.0)
    
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

# Training loop

img_list = []

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
        
        optimizerD.zero_grad()
        
        # sample noise for generator
        z = Variable(torch.Tensor(np.random.normal(0, 1, (b_size, 100)))).to(device)
        
        # generate fake images
        fake_imgs = generator(z).detach()
        
        # real images
        real_valid = discriminator(real_imgs)
        
        # fake images
        fake_valid = discriminator(fake_imgs)
        
        # graident penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
        
        # Wasserstein loss
        d_loss = -torch.mean(real_valid) + torch.mean(fake_valid) + lambda_gp * gradient_penalty
        
        d_loss.backward()
        optimizerD.step()
        
        optimizerG.zero_grad()
        
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
torch.save(generator.state_dict(), "../model_states/WGAN-GP/Gen-[{}-Epochs]".format(EPOCHS))
torch.save(discriminator.state_dict(), "../model_states/WGAN-GP/Dis-[{}-Epochs]".format(EPOCHS))

# Load generator and discriminator
generator = Generator()
generator.load_state_dict(torch.load('../model_states/WGAN-GP/Gen-[5-Epochs]'))

discriminator = Discriminator()
discriminator.load_state_dict(torch.load('../model_states/WGAN-GP/Dis-[5-Epochs]'))
        
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
        