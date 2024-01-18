# main.py

import torch
import memory_profiler
from model import ModifiedModel
from visualization import visualize_clusters
from anonymization import anonymize_embeddings_density_based
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_data
from torch.utils.data import DataLoader
from train_util import train_and_evaluate

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

@memory_profiler.profile
def main(train_file_path, test_file_path):
    # Load train and test datasets
    train_filenames, train_embeddings, original_train_labels = load_data(train_file_path)
    test_filenames, test_embeddings, original_test_labels = load_data(test_file_path)
    print("Datasets loaded")

    # Normalize train and test embeddings using vectorized operations
    normalized_train_embeddings = torch.nn.functional.normalize(train_embeddings, dim=1).to(device)
    normalized_test_embeddings = torch.nn.functional.normalize(test_embeddings, dim=1).to(device)
    print("Normalized embeddings")

    # Anonymize train and test embeddings using density-based clustering
    train_embeddings_anonymized = anonymize_embeddings_density_based(normalized_train_embeddings)
    test_embeddings_anonymized = anonymize_embeddings_density_based(normalized_test_embeddings)

    print("Anonymized embeddings")

    # Convert NumPy arrays to PyTorch tensors
    train_embeddings_anonymized = torch.as_tensor(train_embeddings_anonymized, dtype=torch.float32).clone().detach()
    test_embeddings_anonymized = torch.as_tensor(test_embeddings_anonymized, dtype=torch.float32).clone().detach()

    # Visualize clusters using t-SNE
    visualize_clusters(test_embeddings_anonymized, original_test_labels, method='t-SNE')

    # Visualize clusters using PCA
    visualize_clusters(test_embeddings_anonymized, original_test_labels, method='PCA')

    model = ModifiedModel(input_size=train_embeddings_anonymized.shape[1], output_size=len(np.unique(original_train_labels))).to(device)
    accuracy = train_and_evaluate(model, train_embeddings_anonymized, original_train_labels, test_embeddings_anonymized, original_test_labels, device=device)

    print(f'Accuracy on Test Dataset: {accuracy * 100:.2f}%')

    # Calculate reconstruction error using vectorized operations
    reconstruction_error = torch.mean((normalized_test_embeddings - test_embeddings_anonymized)**2).item()
    print(f'Reconstruction Error: {reconstruction_error:.4f}')

    # Check embedding overlap using a more efficient approach
    has_overlap = any(np.array_equal(embedding, anonymized_embedding) for embedding in normalized_test_embeddings for anonymized_embedding in test_embeddings_anonymized)

    if has_overlap:
        print("Anonymized embeddings found in the original set.")
    else:
        print("No overlap between original and anonymized embeddings.")


    # Plot accuracy loss against reconstruction error
    epsilons = [0.1]  # Different values of noise to try
    reconstruction_errors = []
    accuracy_losses = []

    # Train and evaluate the original model once
    original_model = ModifiedModel(input_size=test_embeddings.shape[1], output_size=len(np.unique(original_test_labels))).to(device)
    print("Original model created")
    original_model_accuracy = train_and_evaluate(original_model, train_embeddings, original_train_labels, test_embeddings, original_test_labels)
    print(f'Accuracy on Original Dataset: {original_model_accuracy * 100:.2f}%')

    # Plotting
    plt.plot(reconstruction_errors, accuracy_losses, marker='o')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Accuracy Loss')
    plt.title('Accuracy Loss vs. Reconstruction Error')
    plt.show()

if __name__ == "__main__":
    train_file_path = 'train_cifar10.npz'
    test_file_path = 'test_cifar10.npz'
    main(train_file_path, test_file_path)