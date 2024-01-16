# model.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from anonymization import anonymize_embeddings_density_based
from data_loader import load_data


# Load CIFAR-10 train and test datasets
train_file_path = '/Users/dominicliebel/Downloads/DinoV2-CIFAR10-CIFAR100/CIFAR10-DINOV2-BASE/train.npz'
test_file_path = '/Users/dominicliebel/Downloads/DinoV2-CIFAR10-CIFAR100/CIFAR10-DINOV2-BASE/test.npz'

train_filenames, train_embeddings, train_labels = load_data(train_file_path)
test_filenames, test_embeddings, test_labels = load_data(test_file_path)

# Assign original embeddings to new variable for future use
test_embeddings_original = test_embeddings

# Anonymize train and test embeddings using density-based clustering
train_embeddings_anonymized = anonymize_embeddings_density_based(train_embeddings)
test_embeddings_anonymized = anonymize_embeddings_density_based(test_embeddings)

# Convert NumPy arrays to PyTorch tensors
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)
train_embeddings_anonymized = torch.as_tensor(train_embeddings_anonymized, dtype=torch.float32).clone().detach()
test_embeddings_anonymized = torch.as_tensor(test_embeddings_anonymized, dtype=torch.float32).clone().detach()

class ModifiedModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ModifiedModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)  # Adjust the size as needed
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = train_embeddings.shape[1]
output_size = len(np.unique(train_labels.numpy()))
model = ModifiedModel(input_size, output_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 20
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, len(train_embeddings_anonymized), batch_size):
        inputs = train_embeddings_anonymized[i:i + batch_size]
        targets = train_labels[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    test_outputs = model(test_embeddings_anonymized)
    _, predicted_labels = torch.max(test_outputs, 1)
    accuracy = accuracy_score(test_labels.numpy(), predicted_labels.numpy())