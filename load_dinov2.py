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

def anonymize_embeddings_laplace(embeddings, epsilon=0.1):
    noise = np.random.laplace(scale=epsilon, size=embeddings.shape)
    return embeddings + noise

def anonymize_embeddings_dp(embeddings, epsilon=1.0):
    noise = np.random.normal(scale=epsilon, size=embeddings.shape)
    return embeddings + noise

def anonymize_embeddings_permutation(embeddings):
    permutation = np.random.permutation(embeddings.shape[1])
    return embeddings[:, permutation]

def anonymize_embeddings_hashing(embeddings, salt="secret_salt"):
    hashed_embeddings = np.vectorize(hash)(embeddings.astype(str) + salt)
    return hashed_embeddings

from sklearn.decomposition import PCA

def anonymize_embeddings_pca(embeddings, n_components=4):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

# Anonymization function using density-based clustering
def anonymize_embeddings_density_based(embeddings, eps=20.0, min_samples=3, noise_scale=0.001):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    cluster_labels = clustering.labels_

    anonymized_embeddings = torch.zeros_like(torch.tensor(embeddings))  # Convert NumPy array to PyTorch tensor

    for label in np.unique(cluster_labels[cluster_labels != -1]):
        cluster_indices = np.where(cluster_labels == label)[0]
        noise = noise_scale * torch.randn_like(torch.tensor(embeddings[cluster_indices]))  # Convert NumPy array to PyTorch tensor
        anonymized_embeddings[cluster_indices] = torch.tensor(embeddings[cluster_indices]) + noise  # Convert NumPy array to PyTorch tensor

    return anonymized_embeddings

# Load CIFAR-10 train and test datasets
train_file_path = '/Users/dominicliebel/Downloads/DinoV2-CIFAR10-CIFAR100/CIFAR10-DINOV2-BASE/train.npz'
test_file_path = '/Users/dominicliebel/Downloads/DinoV2-CIFAR10-CIFAR100/CIFAR10-DINOV2-BASE/test.npz'

train_filenames, train_embeddings, train_labels = load_data(train_file_path)
test_filenames, test_embeddings, test_labels = load_data(test_file_path)



# Anonymize train and test embeddings using density-based clustering
train_embeddings = anonymize_embeddings_density_based(train_embeddings)
test_embeddings = anonymize_embeddings_density_based(test_embeddings)

# Convert NumPy arrays to PyTorch tensors
#train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
#test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)
train_embeddings = torch.as_tensor(train_embeddings, dtype=torch.float32).clone().detach()
test_embeddings = torch.as_tensor(test_embeddings, dtype=torch.float32).clone().detach()

# Define a simple neural network model
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

# Training the model
num_epochs = 20
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
