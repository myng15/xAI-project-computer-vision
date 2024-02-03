import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, latent_dim, conv_dim, embedding_size):
        """
        Initialize the Generator Module
        :param latent_dim: The length of the latent input vectors
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        self.fc1 = nn.Linear(latent_dim, conv_dim * 4)
        self.fc2 = nn.Linear(conv_dim * 4, conv_dim * 8)
        self.fc3 = nn.Linear(conv_dim * 8, embedding_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network (a set of latent input vectors of some length latent_dim)
        :return:
            - out (FloatTensor): A batch of generated 32x32x3 Tensor image
        """

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        out = x
        
        return out
