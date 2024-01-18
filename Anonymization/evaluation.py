# evaluation.py

import torch
from anonymization import anonymize_embeddings
from model import OptimizedModel
from train_util import train_and_evaluate
import numpy as np

def find_best_parameters(original_model_accuracy, normalized_train_embeddings, normalized_test_embeddings,
                         original_train_labels, original_test_labels, device, method,
                         epsilons, min_samples_values, noise_scale_values):
    best_epsilon = None
    best_min_samples = None
    best_noise_scale = None
    best_accuracy = 0.0
    best_reconstruction_error = float('inf')
    reconstruction_errors = []
    accuracy_losses = []

    all_epsilons = []
    all_min_samples_values = []
    all_noise_scale_values = []

    for eps in epsilons:
        for min_samples in min_samples_values:
            for noise_scale in noise_scale_values:
                # Anonymize embeddings using selected method
                test_anonymized_embeddings = (
                    anonymize_embeddings(normalized_test_embeddings, method,
                                         eps=eps, min_samples=min_samples, noise_scale=noise_scale, device=device))
                train_embeddings_anonymized = (
                    anonymize_embeddings(normalized_train_embeddings, method,
                                         eps=eps, min_samples=min_samples, noise_scale=noise_scale, device=device))

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

                # Calculate reconstruction error and accuracy loss
                reconstruction_error = torch.mean((normalized_test_embeddings - test_anonymized_embeddings)**2).item()
                accuracy_loss = original_model_accuracy - anonymized_model_accuracy

                # Append to lists
                reconstruction_errors.append(reconstruction_error)
                accuracy_losses.append(accuracy_loss)

                # Update best parameters based on accuracy or reconstruction error
                if anonymized_model_accuracy > best_accuracy:
                    best_accuracy = anonymized_model_accuracy
                    best_epsilon = eps
                    best_min_samples = min_samples
                    best_noise_scale = noise_scale
                    best_reconstruction_error = reconstruction_error

                # Update lists for all parameters
                all_epsilons.append(eps)
                all_min_samples_values.append(min_samples)
                all_noise_scale_values.append(noise_scale)

                # Print results for the current iteration
                print(f"Iteration: Epsilon={eps}, Min Samples={min_samples}, Noise Scale={noise_scale}, "
                      f"Accuracy={anonymized_model_accuracy * 100:.2f}%,"
                      f"Reconstruction Error={reconstruction_error:.4f}")

    return (best_epsilon, best_min_samples, best_noise_scale, best_accuracy, best_reconstruction_error,
            reconstruction_errors, accuracy_losses, all_epsilons, all_min_samples_values, all_noise_scale_values)
