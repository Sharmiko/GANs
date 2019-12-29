import torch
import itertools
import torch.optim as optim
import torch.nn as nn
import torchvision

from torchvision.utils import save_image
from torch.utils.data import DataLoader

from ResNetGenerator import ResNetGenerator
from UNetGenerator import UnetGenerator
from Discriminator import Discriminator

from tqdm import tqdm

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
EPOCHS = 5
dataset = "monet2photo" # or "apple2orange"
generator = "resnet" # or "unet"

def load_dataset(path, b_size):
    """
    Function for data loading
    """
    dataset = torchvision.datasets.ImageFolder(
            root=path,
            transform=torchvision.transforms.ToTensor())
    loader = DataLoader(
            dataset, batch_size=b_size,
            num_workers=0, shuffle=True)
    return loader

trainA_loader = load_dataset('../data/{}/train_A/'.format(dataset), BATCH_SIZE)
testA_loader = load_dataset('../data/{}/test_A/'.format(dataset), BATCH_SIZE)

trainB_loader = load_dataset('../data/{}/train_B/'.format(dataset), BATCH_SIZE)
testB_loader = load_dataset('../data/{}/test_B/'.format(dataset), BATCH_SIZE)

# Create models
if generator == "resnet":
    generator_AB = ResNetGenerator()
else :
    generator_AB = UnetGenerator()
generator_AB.to(device)

if generator == "resnet":
    generator_BA = ResNetGenerator()
else:
    generator_BA = UnetGenerator()
generator_BA.to(device)


discriminator_A = Discriminator()
discriminator_A.to(device)

discriminator_B = Discriminator()
discriminator_B.to(device)

# Create loss function

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Create optimizers

optimizer_G = optim.Adam(
        itertools.chain(generator_AB.parameters(),
                        generator_BA.parameters()),
                        lr=0.0002, betas=(0.5,0.999)
        )

optimizer_DA = optim.Adam(discriminator_A.parameters(),
                          lr=0.0002, betas=(0.5, 0.999))
optimizer_DB = optim.Adam(discriminator_B.parameters(),
                          lr=0.0002, betas=(0.5, 0.999))


##### Training Loop #####

for epoch in tqdm(range(EPOCHS)):
    
    for images_A, images_B in zip(trainA_loader, trainB_loader):
        
        # real images
        real_A = images_A[0].to(device)
        real_B = images_B[0].to(device)
        
        size = (real_A.size()[0], 1, 16, 16)
        
        target_real = torch.ones(size, requires_grad=False).to(device)
        target_fake = torch.zeros(size, requires_grad=False).to(device)
        
        # Generators - AB and BA
        optimizer_G.zero_grad()
        
        # (1) Identity loss
        same_B = generator_AB(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        
        same_A = generator_BA(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0
        
        # GAN loss
        fake_B = generator_AB(real_A)
        pred_fake = discriminator_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = generator_BA(real_B)
        pred_fake = discriminator_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        
        # Cycle loss
        recovered_A = generator_BA(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0
        
        recovered_B = generator_AB(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
        
        # Total loss
        loss_G = loss_identity_A + loss_identity_B
        loss_G += loss_GAN_A2B + loss_GAN_B2A
        loss_G += loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        
        
        ##### Discriminator A
        optimizer_DA.zero_grad()
        
        # real loss
        pred_real = discriminator_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # fake loss
        pred_fake = discriminator_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        
        # total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        
        optimizer_DA.step()
        
        
        ##### Discriminator B
        optimizer_DB.zero_grad()
        
        # real loss
        pred_real = discriminator_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # fake loss
        pred_fake = discriminator_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        
        # total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        
        optimizer_DB.step()
        
    print("Epoch {}/{}; G loss: {}; D loss{}".format(
            epoch + 1, EPOCHS, loss_G, (loss_D_A.item(), loss_D_B.item())))

# save models
torch.save(generator_AB.state_dict(), 
           '../model_states/CycleGAN/netG_AB_E{}'.format(EPOCHS))
torch.save(generator_BA.state_dict(), 
           '../model_states/CycleGAN/netG_BA_E{}'.format(EPOCHS))
torch.save(discriminator_A.state_dict(), 
           '../model_states/CycleGAN/netD_A_E{}'.format(EPOCHS))
torch.save(discriminator_B.state_dict(), 
           '../model_states/CycleGAN/netD_B_E{}'.format(EPOCHS))


# Generate fake images

num_images = 5
i = 0
for images_A, images_B in zip(testA_loader, testB_loader):
    
    real_A = images_A[0].to(device)
    real_B = images_B[0].to(device)
    
    # generate output
    fake_B = generator_AB(real_A)
    fake_A = generator_BA(real_B)
    
    save_image(fake_A, 'A-{}.png'.format(i+1))
    save_image(fake_B, 'B-{}.png'.format(i+1))
    
    i += 1
    
    if i == num_images:
        break






