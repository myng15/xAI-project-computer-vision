import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, conv_dim, embedding_size):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        # input: (batch_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, conv_dim * 8)
        self.fc2 = nn.Linear(conv_dim * 8, conv_dim * 4)
        self.fc3 = nn.Linear(conv_dim * 4, conv_dim)
        self.fc4 = nn.Linear(conv_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network (a batch of 32x32x3 tensor images)
        :return:
            - out (FloatTensor): the logits output by the discriminator as a tensor of shape (batch_size, 1)
        """

        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        out = self.sigmoid(self.fc4(x))

        return out
