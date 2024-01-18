# main.py

import torch
from model import OptimizedModel
from anonymization import anonymize_embeddings, anonymize_embeddings_laplace, anonymize_embeddings_dp, anonymize_embeddings_density_based
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_data
from train_util import train_and_evaluate
from evaluation import find_best_parameters
from sklearn.preprocessing import StandardScaler
from visualization import visualize_clusters


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

    # Normalize train and test embeddings
    scaler = StandardScaler()
    normalized_train_embeddings = scaler.fit_transform(train_embeddings).astype(np.float32)
    normalized_train_embeddings = torch.as_tensor(normalized_train_embeddings, dtype=torch.float32)
    normalized_test_embeddings = scaler.transform(test_embeddings).astype(np.float32)
    normalized_test_embeddings = torch.as_tensor(normalized_test_embeddings, dtype=torch.float32)
    print("Normalized embeddings")


    mean_value = np.mean(normalized_train_embeddings.cpu().numpy())
    std_dev = np.std(normalized_train_embeddings.cpu().numpy())

    # Check if mean is close to 0 and standard deviation is close to 1
    is_normalized = np.isclose(mean_value, 0.0, atol=1e-5) and np.isclose(std_dev, 1.0, atol=1e-5)

    if is_normalized:
        print("normalized_train_embeddings is normalized.")
    else:
        print("normalized_train_embeddings is not normalized.")

    # This code ensures that both the training and test embeddings are normalized using
    # the mean and standard deviation computed from the training set.
    # The resulting `normalized_train_embeddings` and `normalized_test_embeddings`
    # are NumPy arrays with the embeddings in a normalized form.

    # Train and evaluate the original model once
    original_model = OptimizedModel(input_size=normalized_test_embeddings.shape[1],
                                    output_size=len(np.unique(original_test_labels))).to(device)
    print("Original model created")
    original_model_accuracy = train_and_evaluate(original_model, normalized_train_embeddings, original_train_labels,
                                                 normalized_test_embeddings, original_test_labels, device=device)
    print(f'Accuracy on Original Dataset: {original_model_accuracy * 100:.2f}%')

    # Visualize clusters using t-SNE
    visualize_clusters(normalized_test_embeddings, original_test_labels, method='t-SNE')

    # Visualize clusters using PCA
    visualize_clusters(normalized_test_embeddings, original_test_labels, method='PCA')




    # Anonymize train and test embeddings
    train_embeddings_anonymized = anonymize_embeddings(normalized_train_embeddings, "density_based",
                                                           eps=0.575, min_samples=3, noise_scale=0.01, device=device)
    test_embeddings_anonymized = anonymize_embeddings(normalized_test_embeddings, "density_based",
                                                      eps=0.575, min_samples=3, noise_scale=0.01, device=device)
    print("Anonymized embeddings")

    # Visualize clusters using t-SNE
    visualize_clusters(test_embeddings_anonymized, original_test_labels, method='t-SNE')

    # Visualize clusters using PCA
    visualize_clusters(test_embeddings_anonymized, original_test_labels, method='PCA')

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
     best_reconstruction_error, reconstruction_errors, accuracy_losses,
     all_epsilons, all_min_samples_values, all_noise_scale_values) = find_best_parameters(
        original_model_accuracy, normalized_train_embeddings, normalized_test_embeddings,
        original_train_labels, original_test_labels, device,
        method_to_test,
        epsilons=[1.5, 1.75, 1.9, 2, 2.1, 2.25, 2.5],
        min_samples_values=[2,3,4,5],
        noise_scale_values=[0.5, 1, 1.5, 2]
    )

    print(f"Best Epsilon: {best_epsilon}, Best Min Sample: {best_min_samples}, Best Noise Scale: {best_noise_scale},  Best Accuracy: {best_accuracy * 100:.2f}%, Best Reconstruction Error: {best_reconstruction_error}")

    # Plotting for each combination
    plt.figure()
    plt.plot(reconstruction_errors, accuracy_losses, marker='o')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Accuracy Loss')
    plt.title('Accuracy Loss vs. Reconstruction Error')
    plt.suptitle(f'Parameters: Method={method_to_test}, EPS={best_epsilon}, min_samples={best_min_samples}, noise_scale={best_noise_scale}')

    # Add text annotations for each point with epsilon, min_samples, and noise_scale values
    for i, (error, loss, epsilon, min_samples, noise_scale) in enumerate(zip(reconstruction_errors, accuracy_losses, all_epsilons, all_min_samples_values, all_noise_scale_values)):
        plt.text(error, loss, f'({epsilon=:.6f}, {min_samples=}, {noise_scale=:.4f})', fontsize=8, ha='right', va='bottom')

    plt.show()

if __name__ == "__main__":
    train_file_path = 'train_cifar10.npz'
    test_file_path = 'test_cifar10.npz'
    main(train_file_path, test_file_path)
