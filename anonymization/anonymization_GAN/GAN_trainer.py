import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid

import os
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(os.path.dirname(current))
# adding the parent directory to the sys.path.
sys.path.append(parent)

from anonymization.anonymization_GAN.discriminator import Discriminator
from anonymization.anonymization_GAN.generator import Generator


class GANTrainer:
    """ The trainer class performs functions related to training DCGAN model """

    def __init__(self, d_conv_dim, g_conv_dim, latent_dim, embedding_size, lr, device):
        """
        Initialize the GANTrainer class
        :param discriminator, generator: The instantiated discriminator and generator models
        :param d_optimizer, g_optimizer: The respective optimizers for the discriminator and the generator
        :param device: The device the model is trained on (CPU/GPU)
        """
        super().__init__()
        self.device = device
        self.discriminator, self.generator = self.build_network(d_conv_dim, g_conv_dim, latent_dim, embedding_size)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr, (0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr, (0.5, 0.999))


    def weights_init(self, m):
        """
        (Re-)Initialize weights and bias terms to certain layers in a model.
        :param m: A module or layer in a network
        """

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    (Re-)Initialize weights and bias to convolutional, convolutional-      #
        #    transpose, linear and batch normalization (2d) layers as specified by  #
        #    the original DCGAN paper
        #############################################################################
        # Set mean and std as specified by the original DCGAN paper
        mean = 0.0
        std = 0.02

        # classname will be e.g. `Conv2d`, `ConvTranspose2d`, `BatchNorm2d`, `Linear`
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            # Or:
            # if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean, std)
        elif classname.find('BatchNorm2d') != -1:
            # Or:
            # elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, std)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

    def build_network(self, d_conv_dim, g_conv_dim, latent_dim, embedding_size):
        # define discriminator and generator
        discriminator = Discriminator(conv_dim=d_conv_dim, embedding_size=embedding_size).to(self.device)
        generator = Generator(latent_dim=latent_dim, conv_dim=g_conv_dim, embedding_size=embedding_size).to(self.device)

        # initialize model weights
        discriminator.apply(self.weights_init)
        generator.apply(self.weights_init)

        return discriminator.to(self.device), generator.to(self.device)

    def train_discriminator(self, real_images, latent):
        """
        Implement the training of the discriminator (for each batch)
        :param real_images: The input batch of real images
        :param latent: A fixed set of latent (noise) input vectors to the generator
        :return:
            - d_loss (float): the overall loss (as scalar value) of the discriminator over a batch
            - real_score, fake_score (float): the average of the model predictions for real images and fake images respectively
        """
        # Clear discriminator gradients
        self.d_optimizer.zero_grad()

        # Define criterion to be binary cross entropy with logits loss
        criterion = nn.BCELoss()

        # Pass real images through discriminator
        real_preds = self.discriminator(real_images) # Or: .squeeze()
        real_targets = torch.ones(real_images.size(0), 1, device=self.device)
        # Alternative:
        # real_preds = self.discriminator(real_images).view(-1)  # Or: .squeeze() # shape: (batch_size,)
        # real_label = 1
        # targets = torch.ones((real_images.size(0),), device=self.device) or = torch.full((real_images.size(0),), real_label, device=self.device) # shape: (batch_size,)
        real_loss = criterion(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Generate fake images
        fake_images = self.generator(latent)

        # Pass fake images through discriminator
        fake_preds = self.discriminator(fake_images)
        fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
        # Alternative:
        # fake_preds = self.discriminator(fake_images.detach()).view(-1)  # Or: .squeeze() # shape: (batch_size,)
        # fake_label = 0
        # targets = torch.zeros((fake_images.size(0),), device=self.device) or = torch.full((fake_images.size(0),), fake_label, device=self.device) # shape: (batch_size,)
        fake_loss = criterion(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        d_loss = d_loss.item()
        return d_loss, real_score, fake_score

    def train_generator(self, latent):
        """
        Implement the training of the generator (for each batch)
        :param latent: A fixed set of latent (noise) input vectors to the generator
        :return:
            - g_loss (float): the loss (as scalar value) of the generator over a batch
        """
        # Clear generator gradients
        self.g_optimizer.zero_grad()

        # Define criterion to be binary cross entropy with logits loss
        criterion = nn.BCELoss()

        # Generate fake images
        fake_images = self.generator(latent)

        # Try to fool the discriminator
        preds = self.discriminator(fake_images) # shape: (batch_size, 1)
        # flip target labels from 0 (~ "fake") to 1 to fool the discriminator
        targets = torch.ones(fake_images.size(0), 1, device=self.device) # shape: (batch_size, 1)
        # Alternative:
        #preds = self.discriminator(fake_images).view(-1)  # Or: .squeeze() # shape: (batch_size,)
        #real_label = 1
        #targets = torch.ones((fake_images.size(0),), device=self.device) or = torch.full((fake_images.size(0),), real_label, device=self.device) # shape: (batch_size,)
        g_loss = criterion(preds, targets) #preds and targets need to have the same shape

        # Update generator weights
        g_loss.backward()
        self.g_optimizer.step()

        g_loss = g_loss.item()

        return g_loss

    def save_samples(self, index, sample_size, latent_dim, mean, std, generated_sample_dir='./outputs/', show=True):
        # Sample a fixed set of latent (noise) input vectors to the generator. These are images that are held
        # constant throughout training, and allow us to see how the individual generated images
        # evolve over time as we train the model
        latent = torch.randn(sample_size, latent_dim, 1, 1, device=self.device)
        fake_images = self.generator(latent)

        # denormalize pixel values
        mean_tensor = torch.tensor(mean, device=self.device).view(3, 1, 1)
        std_tensor = torch.tensor(std, device=self.device).view(3, 1, 1)
        fake_images = fake_images * std_tensor + mean_tensor

        fake_filename = 'generated-images-{0:0=4d}.png'.format(index)
        save_image(fake_images, os.path.join(generated_sample_dir, fake_filename), nrow=8)
        print('Saving', fake_filename)
        if show:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

    def save_checkpoint(self, g_losses, d_losses, real_scores, fake_scores, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'device': self.device,
            'g_state_dict': self.generator.state_dict(),
            'd_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': g_losses,
            'd_losses': d_losses,
            'real_scores': real_scores,
            'fake_scores': fake_scores,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.discriminator.load_state_dict(checkpoint['d_state_dict'])
        self.generator.load_state_dict(checkpoint['g_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        g_losses = checkpoint['g_losses']
        d_losses = checkpoint['d_losses']
        real_scores = checkpoint['real_scores']
        fake_scores = checkpoint['fake_scores']
        return self.discriminator, self.generator, g_losses, d_losses, real_scores, fake_scores

    def fit(self, real_embeddings_database, batch_size, n_epochs, latent_dim):
        # Losses & scores
        d_losses = []
        g_losses = []
        real_scores = []
        fake_scores = []

        for epoch in range(n_epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_real_score = 0.0
            epoch_fake_score = 0.0

            num_batches = int(np.floor(len(real_embeddings_database) / batch_size))

            #for batch_i, real_images in enumerate(train_loader):
            for i in range(0, len(real_embeddings_database), batch_size):
                real_images = torch.tensor(real_embeddings_database[i:i + batch_size], device=self.device)

                # Create a batch of latent (noise) input vectors to the generator. These are images that are held
                # constant throughout training, and allow us to see how the individual generated images
                # evolve over time as we train the model
                latent = torch.randn(real_images.size(0), latent_dim, device=self.device)

                # Train discriminator
                d_loss, real_score, fake_score = self.train_discriminator(real_images, latent)
                epoch_d_loss += d_loss
                epoch_real_score += real_score
                epoch_fake_score += fake_score

                # Train generator
                g_loss = self.train_generator(latent)
                epoch_g_loss += g_loss

            # Calculate average losses and scores for the epoch
            epoch_d_loss /= num_batches
            epoch_g_loss /= num_batches
            epoch_real_score /= num_batches
            epoch_fake_score /= num_batches

            # Record losses & scores
            d_losses.append(epoch_d_loss)
            g_losses.append(epoch_g_loss)
            real_scores.append(epoch_real_score)
            fake_scores.append(epoch_fake_score)

            # Log average losses & scores
            print("Epoch [{:3d}/{:3d}] | Discriminator loss: {:.4f} | Generator loss: {:.4f} | Real score: {:.4f} | Fake score: {:.4f}".format(
                    epoch + 1, n_epochs, epoch_d_loss, epoch_g_loss, epoch_real_score, epoch_fake_score))

            # Save generated images
            #self.save_samples(epoch + 1, sample_size, latent_dim, mean, std, show=show)

        return g_losses, d_losses, real_scores, fake_scores

