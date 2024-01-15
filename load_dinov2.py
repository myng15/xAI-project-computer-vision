import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN

# Function to load data from npz files
def load_data(file_path):
    data = np.load(file_path)
    filenames = data['filenames']
    embeddings = data['embeddings']
    labels = data['labels']
    return filenames, embeddings, labels

# Anonymization function (example: adding random noise)
def anonymize_embeddings(embeddings, noise_factor=0.1):
    noise = noise_factor * torch.randn_like(embeddings)
    anonymized_embeddings = embeddings + noise
    return anonymized_embeddings

# Anonymization function using density-based clustering
def anonymize_embeddings_density_based(embeddings, eps=0.5, min_samples=5, noise_scale=0.1):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    cluster_labels = clustering.labels_

    anonymized_embeddings = torch.zeros_like(embeddings)

    for label in np.unique(cluster_labels[cluster_labels != -1]):
        cluster_indices = np.where(cluster_labels == label)[0]
        noise = noise_scale * torch.randn_like(embeddings[cluster_indices])
        anonymized_embeddings[cluster_indices] = embeddings[cluster_indices] + noise

    return anonymized_embeddings

# Load CIFAR-10 train and test datasets
train_file_path = '/Users/dominicliebel/Downloads/DinoV2-CIFAR10-CIFAR100/CIFAR10-DINOV2-BASE/train.npz'
test_file_path = '/Users/dominicliebel/Downloads/DinoV2-CIFAR10-CIFAR100/CIFAR10-DINOV2-BASE/test.npz'

train_filenames, train_embeddings, train_labels = load_data(train_file_path)
test_filenames, test_embeddings, test_labels = load_data(test_file_path)

# Convert NumPy arrays to PyTorch tensors
train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Anonymize train and test embeddings using density-based clustering
train_embeddings = anonymize_embeddings_density_based(train_embeddings)
test_embeddings = anonymize_embeddings_density_based(test_embeddings)

# Define a simple neural network model
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Initialize the model, loss function, and optimizer
input_size = train_embeddings.shape[1]
output_size = len(np.unique(train_labels.numpy()))
model = SimpleModel(input_size, output_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, len(train_embeddings), batch_size):
        inputs = train_embeddings[i:i+batch_size]
        targets = train_labels[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    test_outputs = model(test_embeddings)
    _, predicted_labels = torch.max(test_outputs, 1)
    accuracy = accuracy_score(test_labels.numpy(), predicted_labels.numpy())

print(f'Accuracy on CIFAR-10 Test Dataset: {accuracy * 100:.2f}%')
