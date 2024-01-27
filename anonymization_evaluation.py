import torch

from knn_utils import train_knn_classifier, evaluate_knn_classifier
from anonymization import anonymize_embeddings


def calculate_relative_difference(original_embedding, anonymized_embedding):
    """
    Calculate the relative difference between original and anonymized embeddings.

    Parameters:
    - original_embedding: float, the original embedding value
    - anonymized_embedding: float, the anonymized embedding value

    Returns:
    - float: Relative difference as a percentage
    """
    if torch.any(original_embedding == 0):
        raise ValueError("Cannot calculate relative difference when the original embedding is 0.")

    difference = anonymized_embedding - original_embedding
    relative_difference = (difference / abs(original_embedding)) * 100.0

    return relative_difference


def calculate_mean_relative_difference(original_embeddings, anonymized_embeddings):
    """
    Calculate the mean relative difference for each image.

    Parameters:
    - original_embeddings: list of original embedding values
    - anonymized_embeddings: list of anonymized embedding values

    Returns:
    - list of floats: Mean relative difference for each image
    """
    mean_relative_differences = []
    for original, anonymized in zip(original_embeddings, anonymized_embeddings):
        relative_difference = calculate_relative_difference(original, anonymized)
        mean_relative_difference = torch.mean(relative_difference).item()
        mean_relative_differences.append(mean_relative_difference)

    return mean_relative_differences


def tune_anonymization_parameters(train_embeddings, test_embeddings, train_labels, test_labels, n_neighbors,
                                  anonym_method, noise_scale_tuning=None, n_components_tuning=None,
                                  max_dist_tuning=None, min_samples_tuning=None):
    reconstruction_errors = []
    accuracy_losses = []

    # Train KNN classifier on the original embeddings
    original_knn_classifier = train_knn_classifier(train_embeddings, train_labels, n_neighbors=n_neighbors)
    # Evaluate KNN classifier on the test set
    original_knn_accuracy = evaluate_knn_classifier(original_knn_classifier, test_embeddings, test_labels)
    print(f"KNN Classifier Accuracy on Original Test Set: {original_knn_accuracy * 100:.2f}%")

    if "pca" in anonym_method:
        for n_components in n_components_tuning:
            # Anonymize embeddings using selected method
            train_embeddings_anonymized = anonymize_embeddings(train_embeddings, anonym_method, n_components=n_components)
            test_embeddings_anonymized = anonymize_embeddings(test_embeddings, anonym_method, n_components=n_components)

            # Train KNN classifier on the extracted embeddings
            knn_classifier = train_knn_classifier(train_embeddings_anonymized, train_labels,
                                                  n_neighbors=n_neighbors)

            # Evaluate KNN classifier on the test set
            knn_accuracy = evaluate_knn_classifier(knn_classifier, test_embeddings_anonymized, test_labels)
            print(f"KNN Classifier Accuracy on Anonymized Test Set (n_components={n_components}): {knn_accuracy * 100:.2f}%")

            reconstruction_error = torch.mean(
                (torch.from_numpy(test_embeddings) - test_embeddings_anonymized) ** 2).item()
            accuracy_loss = original_knn_accuracy - knn_accuracy

            # Append to lists
            reconstruction_errors.append(reconstruction_error)
            accuracy_losses.append(accuracy_loss)

    elif "density-based" in anonym_method:
        for max_dist in max_dist_tuning:
            for min_samples in min_samples_tuning:
                for noise_scale in noise_scale_tuning:
                    # Anonymize embeddings using selected method
                    train_embeddings_anonymized = (
                        anonymize_embeddings(train_embeddings, anonym_method, noise_scale=noise_scale,
                                             max_dist=max_dist, min_samples=min_samples))
                    test_embeddings_anonymized = (
                        anonymize_embeddings(test_embeddings, anonym_method, noise_scale=noise_scale, max_dist=max_dist, min_samples=min_samples))
                    print("ANONYMIZATION OF EMBEDDINGS FINISHED.")

                    # Train KNN classifier on the extracted embeddings
                    knn_classifier = train_knn_classifier(train_embeddings_anonymized, train_labels,
                                                          n_neighbors=n_neighbors)

                    # Evaluate KNN classifier on the test set
                    knn_accuracy = evaluate_knn_classifier(knn_classifier, test_embeddings_anonymized, test_labels)
                    print(f"KNN Classifier Accuracy on Anonymized Test Set (noise_scale={noise_scale}, max_dist={max_dist}, min_samples={min_samples}): {knn_accuracy * 100:.2f}%")

                    reconstruction_error = torch.mean((torch.from_numpy(test_embeddings) - test_embeddings_anonymized)**2).item()
                    accuracy_loss = original_knn_accuracy - knn_accuracy

                    # Append to lists
                    reconstruction_errors.append(reconstruction_error)
                    accuracy_losses.append(accuracy_loss)

    else:
        for noise_scale in noise_scale_tuning:
            # Anonymize embeddings using selected method
            train_embeddings_anonymized = anonymize_embeddings(train_embeddings, anonym_method, noise_scale=noise_scale)
            test_embeddings_anonymized = anonymize_embeddings(test_embeddings, anonym_method, noise_scale=noise_scale)
            print("ANONYMIZATION OF EMBEDDINGS FINISHED.")

            # Train KNN classifier on the extracted embeddings
            knn_classifier = train_knn_classifier(train_embeddings_anonymized, train_labels,
                                                  n_neighbors=n_neighbors)

            # Evaluate KNN classifier on the test set
            knn_accuracy = evaluate_knn_classifier(knn_classifier, test_embeddings_anonymized, test_labels)
            print(f"KNN Classifier Accuracy on Anonymized Test Set (noise_scale={noise_scale}): {knn_accuracy * 100:.2f}%")

            reconstruction_error = torch.mean(
                (torch.from_numpy(test_embeddings) - test_embeddings_anonymized) ** 2).item()
            accuracy_loss = (original_knn_accuracy - knn_accuracy) * 100

            # Append to lists
            reconstruction_errors.append(reconstruction_error)
            accuracy_losses.append(accuracy_loss)

    return reconstruction_errors, accuracy_losses
