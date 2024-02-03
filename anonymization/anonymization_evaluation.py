import torch
import numpy as np

from scipy import linalg
from sklearn.metrics.pairwise import euclidean_distances

import sys
import os
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from knn.knn_utils import train_knn_classifier, evaluate_knn_classifier
from anonymization.anonymization_methods import anonymize_embeddings


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


def calculate_variance_retention(original_embeddings, anonymized_embeddings):
    original_variance = np.var(original_embeddings)
    anonymized_variance = np.var(anonymized_embeddings)
    variance_retention = anonymized_variance / original_variance
    return variance_retention


def calculate_projection_robustness(original_embeddings, anonymized_embeddings, original_test_embeddings,
                                    anonymized_test_embeddings):
    original_distances = euclidean_distances(original_embeddings, original_test_embeddings)
    anonymized_distances = euclidean_distances(anonymized_embeddings, anonymized_test_embeddings)
    projection_robustness = np.mean(np.abs(original_distances - anonymized_distances))
    return projection_robustness


# Calculate mean and covariance statistics
def calculate_embedding_statistics(embeddings):
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


# Calculate Frechet Inception Distance (FID)
def calculate_fid(real_embeddings, generated_embeddings):
    mu1, sigma1 = calculate_embedding_statistics(real_embeddings)
    mu2, sigma2 = calculate_embedding_statistics(generated_embeddings)

    # Compute the sum of squared differences between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Compute the square root of the product between covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]

    # Check for imaginary number
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def tune_anonymization_parameters(train_embeddings, test_embeddings, train_labels, test_labels, n_neighbors,
                                  anonym_method, noise_scale_tuning=None, n_components_tuning=None,
                                  max_dist_tuning=None, min_samples_tuning=None,
                                  n_clusters_tuning=None, assign_labels_tuning=('centroid', 'majority')):
    reconstruction_errors_train = []
    accuracy_losses_anonymized_test = []
    accuracy_losses_original_test = []

    if anonym_method == "pca":
        for n_components in n_components_tuning:
            # Anonymize embeddings using selected method
            train_embeddings_anonymized = anonymize_embeddings(train_embeddings, train_labels,
                                                               anonym_method, noise_scale=None,
                                                               n_components=n_components)
            test_embeddings_anonymized = anonymize_embeddings(test_embeddings, test_labels,
                                                              anonym_method, noise_scale=None,
                                                              n_components=n_components)

            train_labels_anonymized = train_labels
            test_labels_anonymized = test_labels

            print("ANONYMIZATION OF EMBEDDINGS FINISHED.")

            # EVALUATE PERFORMANCE METRICS OF THE ANONYMIZATION METHOD
            print(f"Performance Metrics (n_components={n_components}): ")

            reconstruction_error_train, accuracy_loss_anonymized_test, accuracy_loss_original_test, \
            variance_retention, projection_robustness = (
                calculate_anonymization_metrics(train_embeddings, test_embeddings, train_labels, test_labels,
                                                train_embeddings_anonymized, test_embeddings_anonymized,
                                                train_labels_anonymized, test_labels_anonymized,
                                                n_neighbors, anonym_method, n_components=n_components))
            # Append to the lists of all loops' metrics
            reconstruction_errors_train.extend(reconstruction_error_train)
            accuracy_losses_anonymized_test.extend(accuracy_loss_anonymized_test)
            accuracy_losses_original_test.extend(accuracy_loss_original_test)

    elif anonym_method == "density_based":
        for max_dist in max_dist_tuning:
            for min_samples in min_samples_tuning:
                for noise_scale in noise_scale_tuning:
                    # Anonymize embeddings using selected method
                    train_embeddings_anonymized = anonymize_embeddings(
                        train_embeddings, train_labels, anonym_method, noise_scale=noise_scale,
                        max_dist=max_dist, min_samples=min_samples)
                    test_embeddings_anonymized = anonymize_embeddings(
                        test_embeddings, test_labels, anonym_method, noise_scale=noise_scale,
                        max_dist=max_dist, min_samples=min_samples)

                    train_labels_anonymized = train_labels
                    test_labels_anonymized = test_labels

                    print("ANONYMIZATION OF EMBEDDINGS FINISHED.")

                    # EVALUATE PERFORMANCE METRICS OF THE ANONYMIZATION METHOD
                    print(f"Performance Metrics "
                          f"(max_dist={max_dist}, min_samples={min_samples}, noise_scale={noise_scale}): ")

                    reconstruction_error_train, accuracy_loss_anonymized_test, accuracy_loss_original_test, \
                    variance_retention, projection_robustness = (
                        calculate_anonymization_metrics(train_embeddings, test_embeddings, train_labels, test_labels,
                                                        train_embeddings_anonymized, test_embeddings_anonymized,
                                                        train_labels_anonymized, test_labels_anonymized,
                                                        n_neighbors, anonym_method))
                    # Append to the lists of all loops' metrics
                    reconstruction_errors_train.append(reconstruction_error_train)
                    accuracy_losses_anonymized_test.append(accuracy_loss_anonymized_test)
                    accuracy_losses_original_test.append(accuracy_loss_original_test)

    elif anonym_method == "kmeans":
        for n_clusters in n_clusters_tuning:
            for assign_labels in assign_labels_tuning:
                if noise_scale_tuning is None:
                    noise_scale_tuning = [None]
                for noise_scale in noise_scale_tuning:
                    # Anonymize embeddings using selected method
                    train_embeddings_anonymized, train_labels_anonymized = anonymize_embeddings(
                        train_embeddings, train_labels, anonym_method,
                        noise_scale=noise_scale,
                        n_clusters=n_clusters, assign_labels=assign_labels)
                    test_embeddings_anonymized, test_labels_anonymized = anonymize_embeddings(
                        test_embeddings, test_labels, anonym_method,
                        noise_scale=noise_scale,
                        n_clusters=n_clusters, assign_labels=assign_labels)

                    print("ANONYMIZATION OF EMBEDDINGS FINISHED.")

                    # EVALUATE PERFORMANCE METRICS OF THE ANONYMIZATION METHOD
                    print(f"Performance Metrics "
                          f"(n_clusters={n_clusters}, assign_labels={assign_labels}, noise_scale={noise_scale}): ")

                    reconstruction_error_train, accuracy_loss_anonymized_test, accuracy_loss_original_test, \
                    variance_retention, projection_robustness = (
                        calculate_anonymization_metrics(train_embeddings, test_embeddings, train_labels, test_labels,
                                                        train_embeddings_anonymized, test_embeddings_anonymized,
                                                        train_labels_anonymized, test_labels_anonymized,
                                                        n_neighbors, anonym_method))
                    # Append to the lists of all loops' metrics
                    reconstruction_errors_train.append(reconstruction_error_train)
                    accuracy_losses_anonymized_test.append(accuracy_loss_anonymized_test)
                    accuracy_losses_original_test.append(accuracy_loss_original_test)

    else:
        for noise_scale in noise_scale_tuning:
            # Anonymize embeddings using selected method
            train_embeddings_anonymized = anonymize_embeddings(train_embeddings, train_labels, anonym_method,
                                                               noise_scale=noise_scale)
            test_embeddings_anonymized = anonymize_embeddings(test_embeddings, test_labels, anonym_method,
                                                              noise_scale=noise_scale)

            train_labels_anonymized = train_labels
            test_labels_anonymized = test_labels

            print("ANONYMIZATION OF EMBEDDINGS FINISHED.")

            # EVALUATE PERFORMANCE METRICS OF THE ANONYMIZATION METHOD
            print(f"Performance Metrics (noise_scale={noise_scale}): ")

            reconstruction_error_train, accuracy_loss_anonymized_test, accuracy_loss_original_test, \
            variance_retention, projection_robustness = (
                calculate_anonymization_metrics(train_embeddings, test_embeddings, train_labels, test_labels,
                                                train_embeddings_anonymized, test_embeddings_anonymized,
                                                train_labels_anonymized, test_labels_anonymized,
                                                n_neighbors, anonym_method))
            # Append to the lists of all loops' metrics
            reconstruction_errors_train.append(reconstruction_error_train)
            accuracy_losses_anonymized_test.append(accuracy_loss_anonymized_test)
            accuracy_losses_original_test.append(accuracy_loss_original_test)

    return reconstruction_errors_train, accuracy_losses_anonymized_test, accuracy_losses_original_test


def calculate_anonymization_metrics(train_embeddings, test_embeddings, train_labels, test_labels,
                                    train_embeddings_anonymized, test_embeddings_anonymized,
                                    train_labels_anonymized, test_labels_anonymized,
                                    n_neighbors, anonym_method, n_components=None):
    # Train KNN classifier on the original embeddings
    original_knn_classifier = train_knn_classifier(train_embeddings, train_labels, n_neighbors=n_neighbors)
    # Evaluate KNN classifier on the test set
    original_knn_accuracy = evaluate_knn_classifier(original_knn_classifier, test_embeddings, test_labels)
    print(f"KNN Classifier (trained on Original Train Embeddings) Accuracy on Original Test Embeddings: {original_knn_accuracy * 100:.2f}%")

    # EVALUATE PERFORMANCE METRICS OF THE ANONYMIZATION METHOD
    # Train KNN classifier on the extracted embeddings
    knn_classifier = train_knn_classifier(train_embeddings_anonymized, train_labels_anonymized,
                                          n_neighbors=n_neighbors)

    # Evaluate KNN classifier on the test set
    knn_accuracy_anonymized_test = evaluate_knn_classifier(knn_classifier, test_embeddings_anonymized,
                                                           test_labels_anonymized)
    print(f"KNN Classifier (trained on Anonymized Train Embeddings) Accuracy on Anonymized Test Embeddings "
          f"{knn_accuracy_anonymized_test * 100:.2f}%")
    # Evaluate classification accuracy loss using a kNN classifier trained on anonymized train embeddings
    # to classify anonymized test embeddings
    accuracy_loss_anonymized_test = original_knn_accuracy - knn_accuracy_anonymized_test

    # Evaluate KNN classifier trained on the anonymized train set and tested on the original test set
    if anonym_method == 'pca' and n_components is not None:
        accuracy_loss_original_test = None
    else:
        knn_accuracy_original_test = evaluate_knn_classifier(knn_classifier, test_embeddings, test_labels)
        print(f"KNN Classifier (trained on Anonymized Train Embeddings) Accuracy on Original Test Embeddings "
              f"{knn_accuracy_original_test * 100:.2f}%")
        # Evaluate classification accuracy loss using a kNN classifier trained on anonymized train embeddings
        # to classify original test embeddings
        accuracy_loss_original_test = original_knn_accuracy - knn_accuracy_original_test

    # Evaluate reconstruction error of anonymized train and test embeddings against the original embeddings
    if (anonym_method == 'pca' and n_components is not None) or anonym_method == 'kmeans': # or anonym_method == 'gan':
        reconstruction_error_train = None
        variance_retention = None
        projection_robustness = None
    else:
        if isinstance(train_embeddings_anonymized, torch.Tensor):
            train_embeddings_anonymized = train_embeddings_anonymized.cpu().numpy()
        if isinstance(test_embeddings_anonymized, torch.Tensor):
            test_embeddings_anonymized = test_embeddings_anonymized.cpu().numpy()

        reconstruction_error_train = np.mean((train_embeddings - train_embeddings_anonymized) ** 2)
        print(f"Reconstruction Error of Anonymized Train Embeddings: {reconstruction_error_train:.4f}")

        # Compute variance retention
        variance_retention = calculate_variance_retention(train_embeddings, train_embeddings_anonymized)
        print(f"Variance Retention of Anonymized Train Embeddings: {variance_retention:.4f}")

        # Compute projection robustness
        projection_robustness = calculate_projection_robustness(train_embeddings, train_embeddings_anonymized,
                                                                test_embeddings, test_embeddings_anonymized)
        print(f"Projection Rebustness of Anonymized Train Embeddings: {projection_robustness:.4f}")

    if anonym_method == 'gan':
        fid_train = calculate_fid(train_embeddings, train_embeddings_anonymized)
        fid_test = calculate_fid(test_embeddings, test_embeddings_anonymized)
        print(f"FID between Original and Anonymized Train Embeddings: {fid_train:.4e}")
        print(f"FID between Original and Anonymized Test Embeddings: {fid_test:.4e}")

    return reconstruction_error_train, accuracy_loss_anonymized_test * 100, accuracy_loss_original_test * 100, \
           variance_retention, projection_robustness


