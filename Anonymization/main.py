# main.py

import torch
from model import ModifiedModel
from evaluation import check_reconstruction, calculate_silhouette_score, check_embedding_overlap
from visualization import visualize_clusters
from anonymization import anonymize_embeddings_density_based
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_data
from torch.utils.data import DataLoader
from train_util import normalize_embeddings, train_and_evaluate

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def main(train_file_path, test_file_path):
    # Load CIFAR-10 train and test datasets
    train_filenames, train_embeddings, original_train_labels = load_data(train_file_path)
    test_filenames, test_embeddings, original_test_labels = load_data(test_file_path)

    test_embeddings_original = torch.as_tensor(test_embeddings, dtype=torch.float32).clone().detach()

    # Convert NumPy arrays to PyTorch tensors
    original_train_labels = torch.tensor(original_train_labels, dtype=torch.long).to(device)
    test_labels = torch.tensor(original_test_labels, dtype=torch.long).to(device)
    train_embeddings = torch.as_tensor(train_embeddings, dtype=torch.float32).clone().detach().to(device)
    test_embeddings = torch.as_tensor(test_embeddings, dtype=torch.float32).clone().detach().to(device)

    # Normalize train and test embeddings
    normalized_train_embeddings = normalize_embeddings(train_embeddings)
    normalized_test_embeddings = normalize_embeddings(test_embeddings)

    # Anonymize train and test embeddings using density-based clustering
    train_embeddings_anonymized = anonymize_embeddings_density_based(normalized_train_embeddings).to(device)
    test_embeddings_anonymized = anonymize_embeddings_density_based(normalized_test_embeddings).to(device)

    # Visualize clusters using t-SNE
    visualize_clusters(test_embeddings_anonymized, original_test_labels, method='t-SNE')

    # Train and evaluate the model
    train_dataset = torch.utils.data.TensorDataset(train_embeddings_anonymized, original_train_labels)  # Use a different variable for normalized labels
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ModifiedModel(input_size=train_embeddings_anonymized.shape[1], output_size=len(np.unique(original_test_labels))).to(device)
    accuracy = train_and_evaluate(model, train_dataloader, test_embeddings_anonymized, original_test_labels)

    print(f'Accuracy on CIFAR-10 Test Dataset: {accuracy * 100:.2f}%')

    # Check reconstruction error
    reconstruction_error = check_reconstruction(test_embeddings_original, test_embeddings_anonymized)
    print(f'Reconstruction Error: {reconstruction_error:.4f}')

    # Calculate Silhouette Score
    silhouette_avg = calculate_silhouette_score(test_embeddings_anonymized, test_labels)
    print(f'Silhouette Score: {silhouette_avg:.4f}')

    # Check embedding overlap
    has_overlap = check_embedding_overlap(test_embeddings_original, test_embeddings_anonymized)

    if has_overlap:
        print("Anonymized embeddings found in the original set.")
    else:
        print("No overlap between original and anonymized embeddings.")

    # Plot accuracy loss against reconstruction error
    epsilons = [0.1, 0.5, 1.0]  # Different values of noise to try
    reconstruction_errors = []
    accuracy_losses = []

    original_model = ModifiedModel(input_size=test_embeddings.shape[1], output_size=len(np.unique(original_test_labels))).to(device)
    original_model_accuracy = train_and_evaluate(original_model, train_embeddings, original_train_labels, test_embeddings, original_test_labels)

    for epsilon in epsilons:
        # Anonymize embeddings using density-based clustering with different values of noise
        anonymized_train_embeddings = anonymize_embeddings_density_based(train_embeddings, eps=epsilon)
        anonymized_test_embeddings = anonymize_embeddings_density_based(test_embeddings, eps=epsilon)
        # Calculate reconstruction error
        reconstruction_error = check_reconstruction(test_embeddings, anonymized_test_embeddings)
        reconstruction_errors.append(reconstruction_error)

        # Train and evaluate the model using anonymized embeddings
        model = ModifiedModel(input_size=anonymized_train_embeddings.shape[1], output_size=len(np.unique(original_test_labels.numpy()))).to(device)
        model_accuracy = train_and_evaluate(model, anonymized_train_embeddings, original_test_labels, anonymized_test_embeddings, original_test_labels)

        # Calculate accuracy loss
        accuracy_loss = original_model_accuracy - model_accuracy
        accuracy_losses.append(accuracy_loss)

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
