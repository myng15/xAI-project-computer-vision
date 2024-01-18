# main.py

import torch
from model import OptimizedModel
from anonymization import anonymize_embeddings_dp, anonymize_embeddings
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_data
from train_util import train_and_evaluate
from evaluation import find_best_parameters

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")


def main(train_file_path, test_file_path):
    # Load train and test datasets
    train_filenames, train_embeddings, original_train_labels = load_data(train_file_path)
    test_filenames, test_embeddings, original_test_labels = load_data(test_file_path)
    print("Datasets loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize train and test embeddings using vectorized operations
    normalized_train_embeddings = torch.nn.functional.normalize(train_embeddings.to(device), dim=1).to(device)
    normalized_test_embeddings = torch.nn.functional.normalize(test_embeddings.to(device), dim=1).to(device)
    print("Normalized embeddings")

    # Train and evaluate the original model once
    original_model = OptimizedModel(input_size=normalized_test_embeddings.shape[1],
                                    output_size=len(np.unique(original_test_labels))).to(device)
    print("Original model created")
    original_model_accuracy = train_and_evaluate(original_model, normalized_train_embeddings,original_train_labels,
                                                 normalized_test_embeddings, original_test_labels, device=device)
    print(f'Accuracy on Original Dataset: {original_model_accuracy * 100:.2f}%')

    # Anonymize train and test embeddings
    # train_embeddings_anonymized = anonymize_embeddings_density_based(normalized_train_embeddings, eps=0.5, min_samples=10, noise_scale=5, device=device).to(device)
    # test_embeddings_anonymized = anonymize_embeddings_density_based(normalized_test_embeddings, eps=0.5, min_samples=10, noise_scale=5, device=device).to(device)
    train_embeddings_anonymized = anonymize_embeddings_dp(normalized_train_embeddings)
    test_embeddings_anonymized = anonymize_embeddings_dp(normalized_test_embeddings)

    # Convert NumPy arrays to PyTorch tensors
    train_embeddings_anonymized = torch.as_tensor(train_embeddings_anonymized, dtype=torch.float32).clone().detach().to(device)
    test_embeddings_anonymized = torch.as_tensor(test_embeddings_anonymized, dtype=torch.float32).clone().detach().to(device)
    print("Anonymized embeddings")
    # Visualize clusters using t-SNE
    #visualize_clusters(test_embeddings_anonymized, original_test_labels, method='t-SNE')

    # Visualize clusters using PCA
    #visualize_clusters(test_embeddings_anonymized, original_test_labels, method='PCA')

    model = OptimizedModel(input_size=train_embeddings_anonymized.shape[1], output_size=len(np.unique(original_train_labels))).to(device)
    accuracy = train_and_evaluate(model, train_embeddings_anonymized, original_train_labels, test_embeddings_anonymized, original_test_labels, device=device)

    print(f'Accuracy on Test Dataset: {accuracy * 100:.2f}%')

    reconstruction_error = torch.mean((normalized_test_embeddings - test_embeddings_anonymized)**2).item()
    print(f'Reconstruction Error: {reconstruction_error:.4f}')

    # Convert normalized_test_embeddings and test_embeddings_anonymized to sets
    normalized_test_set = {tuple(embedding.flatten()) for embedding in normalized_test_embeddings}
    anonymized_set = {tuple(embedding.flatten()) for embedding in test_embeddings_anonymized}

    # Check for overlap
    overlap = normalized_test_set.intersection(anonymized_set)

    if len(overlap) > 0:
        print(f"Overlap: {len(overlap)}")
    else:
        print("No overlap between original and anonymized embeddings.")

    method_to_test = 'density_based'  # Change to 'dp' for testing anonymize_embeddings_dp

    (best_epsilon, best_min_samples, best_noise_scale, best_accuracy,
     best_reconstruction_error, reconstruction_errors, accuracy_losses) = find_best_parameters(
        original_model_accuracy, normalized_train_embeddings, normalized_test_embeddings,
        original_train_labels, original_test_labels, device,
        method_to_test,
        epsilons=[1.1, 1.15, 1.2, 1.25, 1.275, 1.30],
        min_samples_values=[10, 30],
        noise_scale_values=[0.5, 2, 5]
    )

    print(f"Best Epsilon: {best_epsilon}, Best Min Sample: {best_min_samples}, Best Noise Scale: {best_noise_scale},  Best Accuracy: {best_accuracy * 100:.2f}%, Best Reconstruction Error: {best_reconstruction_error}")

    # Plotting for each combination
    plt.figure()
    plt.plot(reconstruction_errors, accuracy_losses, marker='o')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Accuracy Loss')
    plt.title('Accuracy Loss vs. Reconstruction Error')
    plt.suptitle(f'Parameters: Method={method_to_test}, EPS={best_epsilon}, min_samples={min_samples}, noise_scale={noise_scale}')
    plt.show()


if __name__ == "__main__":
    train_file_path = 'train_cifar10.npz'
    test_file_path = 'test_cifar10.npz'
    main(train_file_path, test_file_path)
