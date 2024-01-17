# model.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from anonymization import anonymize_embeddings_density_based
from data_loader import load_data
from train_util import train_and_evaluate

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")

# Load CIFAR-10 train and test datasets
train_file_path = 'train_cifar10.npz'
test_file_path = 'test_cifar10.npz'

train_filenames, train_embeddings, train_labels = load_data(train_file_path)
test_filenames, test_embeddings, test_labels = load_data(test_file_path)

# Assign original embeddings to new variable for future use
test_embeddings_original = test_embeddings

# Anonymize train and test embeddings using density-based clustering
train_embeddings_anonymized = anonymize_embeddings_density_based(train_embeddings).to(device)
test_embeddings_anonymized = anonymize_embeddings_density_based(test_embeddings).to(device)

# Normalize embeddings
train_embeddings_anonymized = torch.nn.functional.normalize(train_embeddings_anonymized, dim=1)
test_embeddings_anonymized = torch.nn.functional.normalize(test_embeddings_anonymized, dim=1)

# Convert NumPy arrays to PyTorch tensors
train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
train_embeddings_anonymized = torch.as_tensor(train_embeddings_anonymized, dtype=torch.float32).clone().detach()
test_embeddings_anonymized = torch.as_tensor(test_embeddings_anonymized, dtype=torch.float32).clone().detach()

# Create DataLoader for efficient batching
train_dataset = torch.utils.data.TensorDataset(train_embeddings_anonymized, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

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
input_size = train_embeddings_anonymized.shape[1]
output_size = len(np.unique(train_labels.cpu().numpy()))
model = ModifiedModel(input_size, output_size).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 20

accuracy = train_and_evaluate(model, train_dataloader, test_embeddings, test_labels)

# Continue with the rest of your model.py code...


for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

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
    accuracy = accuracy_score(test_labels.cpu().numpy(), predicted_labels.cpu().numpy())
