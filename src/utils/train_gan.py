import torch
import wandb
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib
from tqdm import tqdm

from torchvision.utils import (
    make_grid,
    save_image,
) 
from matplotlib import pyplot as plt


def label_real(size, device):
    """
    Function to create real labels (1s)
    """

    data = torch.ones(size, 1)

    return data.to(device)


def label_fake(size, device):
    """
    Function to create fake labels (0s)
    """

    data = torch.zeros(size, 1)

    return data.to(device)


def create_noise(sample_size, nz, device):
    """
    Function to create the noise vector
    """

    return torch.randn(sample_size, nz).to(device)



def train_discriminator(
    discriminator, optimizer, data_real,device, criterion ,data_fake
):
    """
    Function to train the discriminator network
    """

    #discriminator.to(device)
    b_size = data_real.size(0)
    real_label = label_real(b_size, device)
    fake_label = label_fake(b_size, device)
    optimizer.zero_grad()
    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()

    return loss_real + loss_fake


def train_generator(discriminator, optimizer,  device, criterion,data_fake):
    """
    Function to train the generator network
    """
    
    b_size = data_fake.size(0)
    real_label = label_real(b_size, device)
    optimizer.zero_grad()
    output = discriminator(data_fake)
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    return loss


def training_loop_gan(
    dataloader,
    discriminator,
    generator,
    discriminator_optimizer,
    generator_optimizer,
    device,
    criterion,
    epochs,
    nz,
):
    # Lists to keep track of progress
    losses_g = [] # to store generator loss after each epoch
    losses_d = [] # to store discriminator loss after each epoch
    images = [] # to store images generatd by the generator
    iters = 0

    print("Starting Training Loop...")
    generator.train()
    discriminator.train()
    # For each epoch
    for epoch in range(epochs):
        loss_g = 0.0
        loss_d = 0.0
        for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader))):
            image, _ = data
            image = image.to(device)
            b_size = len(image)
            data_fake = generator(create_noise(b_size, nz, device)).detach()
            data_real = image
            # train the discriminator network
            loss_d += train_discriminator(discriminator, discriminator_optimizer, data_real, device, criterion, data_fake)
            data_fake = generator(create_noise(b_size, nz, device))
            # train the generator network
            loss_g += train_generator(discriminator, generator_optimizer, device, criterion,data_fake)
        # create the final fake image for the epoch
        noise = create_noise(64, nz, device)
        generated_img = generator(noise).cpu().detach()
        # make the images as grid
        generated_img = make_grid(generated_img)
        # save the generated torch tensor models to disk
        save_image(generated_img, f"outputs/gen_img{epoch}.png")
        images.append(generated_img)
        epoch_loss_g = loss_g / bi # total generator loss for the epoch
        epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
        losses_g.append(epoch_loss_g)
        losses_d.append(epoch_loss_d)

        # log the losses to wandb
        wandb.log({"Generator Loss": epoch_loss_g, "Discriminator Loss": epoch_loss_d})
    
        print(f"Epoch {epoch} of {epochs}")
        print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

    print('DONE TRAINING')
    torch.save(generator.state_dict(), 'outputs/generator.pth')

    # save the generated images as GIF file
    to_pil_image = transforms.ToPILImage()
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('outputs/generator_images.gif', imgs)

    return generator, discriminator, losses_g, losses_d