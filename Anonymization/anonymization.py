# anonymization.py

import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd

def anonymize_embeddings(embeddings, noise_factor=0.1):
    anonymized_embeddings = noise_factor * torch.randn_like(embeddings)
    return anonymized_embeddings

def anonymize_embeddings_laplace(embeddings, epsilon=0.1):
    anonymized_embeddings = embeddings + torch.tensor(np.random.laplace(scale=epsilon, size=embeddings.shape))
    return anonymized_embeddings

def anonymize_embeddings_dp(embeddings, epsilon=1.0):
    anonymized_embeddings = embeddings + torch.tensor(np.random.normal(scale=epsilon, size=embeddings.shape), dtype=torch.float32)
    return embeddings

def anonymize_embeddings_permutation(embeddings):
    permutation = torch.randperm(embeddings.shape[1])
    return embeddings[:, permutation]

def anonymize_embeddings_hashing(embeddings, salt="secret_salt"):
    hashed_embeddings = torch.tensor(np.vectorize(hash)(embeddings.cpu().numpy().astype(str) + salt), dtype=torch.long)
    return hashed_embeddings

def anonymize_embeddings_pca(embeddings, n_components=10):
    pca = PCA(n_components=n_components)
    return torch.tensor(pca.fit_transform(embeddings.cpu().numpy()), dtype=torch.float32)

def k_anonymization(data, k=2):
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data should be a NumPy array")

    data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
    masked_data = k_anonymization(data, k=k)
    masked_data = masked_data.to_numpy()
    return masked_data

def anonymize_embeddings_density_based(embeddings, eps=60.0, min_samples=20, noise_scale=2):
    """
    Anonymize embeddings using density-based clustering.

    Parameters:
    - embeddings: PyTorch tensor, the original set of embeddings
    - eps: float, the maximum distance between two samples for one to be considered as in the neighborhood of the other
    - min_samples: int, the number of samples (or total weight) in a neighborhood for a point to be considered as a core point
    - noise_scale: int, the scaling factor for adding noise to anonymize embeddings

    Returns:
    - PyTorch tensor: Anonymized embeddings
    """
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    cluster_labels = clustering.labels_

    # Add noise to the original embeddings based on clusters
    anonymized_embeddings = torch.tensor(embeddings)
    for label in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_noise = noise_scale * torch.randn_like(torch.tensor(embeddings[cluster_indices]))
        anonymized_embeddings[cluster_indices] += cluster_noise

    return anonymized_embeddings