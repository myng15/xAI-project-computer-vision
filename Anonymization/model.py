# model.py

import torch

class ModifiedModel(torch.nn.Module):
    """Custom CNN model for image classification."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)  # Adjust the size as needed
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
