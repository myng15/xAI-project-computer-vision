import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd

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

def k_anonymization(data, k=2):
    """
    Perform k-anonymization on a given NumPy array.

    Parameters:
    - data: numpy.ndarray
        The input NumPy array containing sensitive information.
    - k: int, default=2
        The parameter for k-anonymity. Each record is made indistinguishable
        from at least k-1 other records.

    Returns:
    - masked_data: pandas DataFrame
        The NumPy array after applying k-anonymization.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data should be a NumPy array")

    # Convert NumPy array to Pandas DataFrame
    data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])

    masked_data = k_anonymization(data, k=k)

    # Convert Pandas DataFrame back to NumPy array
    masked_data = masked_data.to_numpy()

    return masked_data


# Anonymization function using density-based clustering
def anonymize_embeddings_density_based(embeddings, eps=60.0, min_samples=20, noise_scale=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    cluster_labels = clustering.labels_

    anonymized_embeddings = torch.zeros_like(torch.tensor(embeddings))  # Convert NumPy array to PyTorch tensor

    for label in np.unique(cluster_labels[cluster_labels != -1]):
        cluster_indices = np.where(cluster_labels == label)[0]
        noise = noise_scale * torch.randn_like(torch.tensor(embeddings[cluster_indices]))  # Convert NumPy array to PyTorch tensor
        anonymized_embeddings[cluster_indices] = torch.tensor(embeddings[cluster_indices]) * (1+noise)  # Convert NumPy array to PyTorch tensor

    return anonymized_embeddings

# Load CIFAR-10 train and test datasets
train_file_path = '/Users/dominicliebel/Downloads/DinoV2-CIFAR10-CIFAR100/CIFAR10-DINOV2-BASE/train.npz'
test_file_path = '/Users/dominicliebel/Downloads/DinoV2-CIFAR10-CIFAR100/CIFAR10-DINOV2-BASE/test.npz'

train_filenames, train_embeddings, train_labels = load_data(train_file_path)
test_filenames, test_embeddings, test_labels = load_data(test_file_path)

test_embeddings_original = test_embeddings





# Anonymize train and test embeddings using density-based clustering
train_embeddings = anonymize_embeddings_density_based(train_embeddings)
test_embeddings = anonymize_embeddings_density_based(test_embeddings)






# Convert NumPy arrays to PyTorch tensors
train_labels = torch.tensor(train_labels, dtype=torch.long)
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


# Assuming your original embeddings have more than two dimensions
#pca = PCA(n_components=2)
#reduced_embeddings = pca.fit_transform(train_embeddings)

# Visualize the clusters after anonymization
#plt.scatter(test_embeddings[:, 0], test_embeddings[:, 1], c=test_labels, cmap='viridis', s=20)
#plt.title('Clusters After Anonymization')
#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.show()

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings_tsne = tsne.fit_transform(test_embeddings)

# Visualize the clusters after anonymization using t-SNE
#plt.scatter(reduced_embeddings_tsne[:, 0], reduced_embeddings_tsne[:, 1], c=test_labels, cmap='viridis', s=20)
#plt.title('Clusters After Anonymization (t-SNE)')
#plt.xlabel('t-SNE Component 1')
#plt.ylabel('t-SNE Component 2')
#plt.show()

def check_reconstruction(original_embeddings, anonymized_embeddings):
    # Flatten and normalize the embeddings
    scaler = StandardScaler()
    original_normalized = scaler.fit_transform(original_embeddings.flatten().reshape(-1, 1)).flatten()
    anonymized_normalized = scaler.transform(anonymized_embeddings.flatten().reshape(-1, 1)).flatten()

    # Calculate mean squared error
    mse = mean_squared_error(original_normalized, anonymized_normalized)

    return mse

# Assuming you have original and anonymized embeddings
original_embeddings = test_embeddings_original  # Adjust dimensions based on your data
anonymized_embeddings = test_embeddings  # Anonymized embeddings

# Check reconstruction error
reconstruction_error = check_reconstruction(original_embeddings, anonymized_embeddings)
print(f'Reconstruction Error: {reconstruction_error:.4f}')



# Calculate Silhouette Score
silhouette_avg = silhouette_score(test_embeddings_original, test_labels)
print(f'Silhouette Score: {silhouette_avg:.4f}')




def check_embedding_overlap(original_embeddings, anonymized_embeddings):
    """
    Check if any of the anonymized embeddings can be found in the original set of embeddings.

    Parameters:
    - original_embeddings: PyTorch tensor, the original set of embeddings
    - anonymized_embeddings: PyTorch tensor, the anonymized set of embeddings

    Returns:
    - bool: True if any anonymized embedding is found in the original set, False otherwise
    """

    original_set = set(map(tuple, original_embeddings))
    anonymized_set = set(map(tuple, anonymized_embeddings))

    return any(embedding in original_set for embedding in anonymized_set)

# Example usage:
has_overlap = check_embedding_overlap(test_embeddings_original, test_embeddings)

if has_overlap:
    print("Anonymized embeddings found in the original set.")
else:
    print("No overlap between original and anonymized embeddings.")