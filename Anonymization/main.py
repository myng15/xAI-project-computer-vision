# main.py

import torch
from model import OptimizedModel
from anonymization import anonymize_embeddings_density_based, anonymize_embeddings_pca, anonymize_embeddings_dp
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_data
from train_util import train_and_evaluate

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

    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)

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

    # Anonymize train and test embeddings using density-based clustering
    # train_embeddings_anonymized = anonymize_embeddings_density_based(normalized_train_embeddings, eps=0.5, min_samples=10, noise_scale=5, device=device)
    # test_embeddings_anonymized = anonymize_embeddings_density_based(normalized_test_embeddings, eps=0.5, min_samples=10, noise_scale=5, device=device)
    train_embeddings_anonymized = anonymize_embeddings_dp(normalized_train_embeddings)
    test_embeddings_anonymized = anonymize_embeddings_dp(normalized_test_embeddings)

    # Convert NumPy arrays to PyTorch tensors
    train_embeddings_anonymized = train_embeddings_anonymized.to(device)
    test_embeddings_anonymized = test_embeddings_anonymized.to(device)

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

    if normalized_test_embeddings.size(1) != test_embeddings_anonymized.size(1):
        # Handle the dimension mismatch, possibly by adjusting the size or reshaping one of the tensors
        # For example, if the second dimension represents the number of classes, you might want to make sure it's the same
        # You may need to adapt this depending on your specific use case

        # Example: Assuming the second dimension represents the number of classes
        num_classes = min(normalized_test_embeddings.size(1), test_embeddings_anonymized.size(1))
        normalized_test_embeddings = normalized_test_embeddings[:, :num_classes]
        test_embeddings_anonymized = test_embeddings_anonymized[:, :num_classes]

        # Convert normalized_test_embeddings and test_embeddings_anonymized to sets
        normalized_test_set = set(tuple(embedding.flatten()) for embedding in normalized_test_embeddings)
        anonymized_set = set(tuple(embedding.flatten()) for embedding in test_embeddings_anonymized)

        # Check for overlap between the sets
        overlap = normalized_test_set.intersection(anonymized_set)
        print(f"Overlap: {len(overlap)}")

    reconstruction_error = torch.mean((normalized_test_embeddings - test_embeddings_anonymized)**2).item()
    print(f'Reconstruction Error: {reconstruction_error:.4f}')

    # Convert normalized_test_embeddings and test_embeddings_anonymized to sets
    normalized_test_set = {tuple(embedding.flatten()) for embedding in normalized_test_embeddings}
    anonymized_set = {tuple(embedding.flatten()) for embedding in test_embeddings_anonymized}

    # Check for overlap
    has_overlap = any(embedding in anonymized_set for embedding in normalized_test_set)

    if has_overlap:
        print("Anonymized embeddings found in the original set.")
    else:
        print("No overlap between original and anonymized embeddings.")

    # Values to try for density based anonymization
    epsilons = [1.1, 1.15, 1.2, 1.25, 1.275, 1.30]
    min_samples_values = [10, 30]
    noise_scale_values = [0,5, 2, 5]
    reconstruction_errors = []
    accuracy_losses = []

    # Loop through combinations of density based anonymization
    for eps_idx, eps in enumerate(epsilons):
        for min_samples_idx, min_samples in enumerate(min_samples_values):
            for noise_scale_idx, noise_scale in enumerate(noise_scale_values):
                # Anonymize embeddings using density-based clustering
                test_anonymized_embeddings = anonymize_embeddings_density_based(normalized_test_embeddings, eps=eps, min_samples=min_samples, noise_scale=noise_scale, device=device)
                train_embeddings_anonymized = anonymize_embeddings_density_based(normalized_train_embeddings, eps=eps, min_samples=min_samples, noise_scale=noise_scale, device=device)

                test_anonymized_embeddings = test_anonymized_embeddings.to(device)
                train_embeddings_anonymized = train_embeddings_anonymized.to(device)
                normalized_test_embeddings = normalized_test_embeddings.to(device)

                # Train and evaluate the model on anonymized data
                anonymized_model = OptimizedModel(input_size=test_anonymized_embeddings.shape[1], output_size=len(np.unique(original_test_labels))).to(device)
                anonymized_model_accuracy = train_and_evaluate(
                    anonymized_model,
                    train_embeddings_anonymized,
                    original_train_labels,
                    test_anonymized_embeddings,
                    original_test_labels
                )
                # Print accuracy for each combination
                print(f'Accuracy for EPS={eps}, min_samples={min_samples}, noise_scale={noise_scale}: {anonymized_model_accuracy * 100:.2f}%')

                # Calculate reconstruction error and accuracy loss
                reconstruction_error = torch.mean((normalized_test_embeddings - test_anonymized_embeddings)**2).item()
                accuracy_loss = original_model_accuracy - anonymized_model_accuracy

                # Append values to lists for plotting
                reconstruction_errors.append(reconstruction_error)
                accuracy_losses.append(accuracy_loss)
                print(f"Iteration: EPS {eps_idx+1} of {len(epsilons)}, min_samples {min_samples_idx+1} of "
                      f"{len(min_samples_values)}, noise {noise_scale_idx+1} of {len(noise_scale_values)}")

    # Plotting for each combination
    plt.figure()
    plt.plot(reconstruction_errors, accuracy_losses, marker='o')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Accuracy Loss')
    plt.title('Accuracy Loss vs. Reconstruction Error')
    plt.suptitle(f'Parameters: EPS={eps}, min_samples={min_samples}, noise_scale={noise_scale}')
    plt.show()


if __name__ == "__main__":
    train_file_path = 'train_cifar10.npz'
    test_file_path = 'test_cifar10.npz'
    main(train_file_path, test_file_path)
