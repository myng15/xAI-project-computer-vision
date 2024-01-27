import torch
import numpy as np
from sklearn.cluster import DBSCAN


def anonymize_embeddings_random(embeddings, noise_scale=0.1):
    anonymized_embeddings = embeddings + noise_scale * torch.randn_like(embeddings)
    return anonymized_embeddings


def anonymize_embeddings_laplace(embeddings, noise_scale=0.1):
    """
    Anonymize embeddings using Laplace noise.

    Parameters:
    - embeddings: PyTorch tensor, the original embeddings
    - noise_scale: float, scale parameter for Laplace distribution
            (Smaller values mean the generated noise will have lower variance, i.e. less distortion to the original data, but also less privacy.
             Larger values mean more variability in the noise, which enhances privacy but might reduce the utility of the data.)

    Returns:
    - PyTorch tensor, anonymized embeddings
    """
    laplace_noise = torch.tensor(np.random.laplace(loc=0.0, scale=noise_scale, size=embeddings.shape), dtype=torch.float32, device="cpu")
    anonymized_embeddings = embeddings + laplace_noise
    return anonymized_embeddings


def anonymize_embeddings_gaussian(embeddings, noise_scale=0.1):
    gaussian_noise = torch.tensor(np.random.normal(loc=0.0, scale=noise_scale, size=embeddings.shape), dtype=torch.float32, device="cpu")
    anonymized_embeddings = embeddings + gaussian_noise
    return anonymized_embeddings


def anonymize_embeddings_permutation(embeddings):
    permutation = torch.randperm(embeddings.shape[1], device="cpu")
    return embeddings[:, permutation]


def anonymize_embeddings_hashing(embeddings, salt="secret_salt"):
    hashed_embeddings = torch.tensor(np.vectorize(hash)(embeddings.cpu().numpy().astype(str) + salt), dtype=torch.long, device="cpu")
    return hashed_embeddings


def anonymize_embeddings_pca(embeddings, n_components=None):
    #pca = PCA(n_components=n_components)
    #return torch.tensor(pca.fit_transform(embeddings.cpu().numpy()), dtype=torch.float32)

    # Reshape to (batch_size, num_features) where num_features is 3 * 32 * 32
    flattened_embeddings = embeddings.view(embeddings.size(0), -1)

    # Center the Data
    mean = torch.mean(flattened_embeddings, dim=0)
    centered_embeddings = flattened_embeddings - mean

    # Convert to complex tensor
    centered_embeddings_complex = centered_embeddings.to(torch.complex64)

    # Calculate Covariance Matrix
    covariance_matrix = torch.matmul(centered_embeddings_complex.t(), centered_embeddings_complex) / centered_embeddings_complex.size(0)

    # Eigenvalue Decomposition
    eigvalues_complex, eigvectors_complex = torch.linalg.eig(covariance_matrix)

    # Whiten the Data
    if n_components:
        anonymized_embeddings_complex = torch.matmul(centered_embeddings_complex, eigvectors_complex[:, :n_components])
    else:
        anonymized_embeddings_complex = torch.matmul(centered_embeddings_complex, eigvectors_complex)

    # Convert back to real tensor
    anonymized_embeddings = anonymized_embeddings_complex.real

    return anonymized_embeddings


def anonymize_embeddings_density_based(embeddings, max_dist=0.5, min_samples=5, noise_scale=0.1):
    """
    Anonymize embeddings using density-based clustering.

    Parameters:
    - embeddings: PyTorch tensor, the original embeddings
    - max_dist: float, maximum distance between two samples for one to be considered as in the neighborhood of the other
    - min_samples: int, the number of samples in a neighborhood for a point to be considered as a core point
    - noise_scale: float, scale parameter for Laplace noise
    - device: str, device to place the noise tensor on ("cpu" or "cuda")

    Returns:
    - PyTorch tensor, anonymized embeddings
    """
    embeddings = embeddings.numpy()

    # Perform density-based clustering using DBSCAN
    db = DBSCAN(eps=max_dist, min_samples=min_samples).fit(embeddings)

    # Assign a cluster label to each data point
    cluster_labels = db.labels_

    # Generate Laplace noise
    laplace_noise = np.random.laplace(loc=0.0, scale=noise_scale, size=embeddings.shape)

    # Add noise to each cluster separately
    unique_labels = np.unique(cluster_labels)
    anonymized_embeddings = embeddings.copy()
    for label in unique_labels:
        cluster_indices = (cluster_labels == label)
        anonymized_embeddings[cluster_indices] += laplace_noise[cluster_indices]

    return torch.tensor(anonymized_embeddings, dtype=torch.float32, device="cpu")


def anonymize_embeddings(embeddings, method, noise_scale=None, n_components=None, max_dist=None, min_samples=None):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings, dtype=torch.float32, device="cpu")
    if method == 'random':
        return anonymize_embeddings_random(embeddings, noise_scale=noise_scale)
    elif method == 'laplace':
        return anonymize_embeddings_laplace(embeddings, noise_scale=noise_scale)
    elif method == 'gaussian':
        return anonymize_embeddings_gaussian(embeddings, noise_scale=noise_scale)
    elif method == 'permutation':
        return anonymize_embeddings_permutation(embeddings)
    elif method == 'hashing':
        return anonymize_embeddings_hashing(embeddings)
    elif method == 'pca':
        return anonymize_embeddings_pca(embeddings, n_components=n_components)
    elif method == 'density_based':
        return anonymize_embeddings_density_based(embeddings, max_dist=max_dist, min_samples=min_samples, noise_scale=noise_scale)
    else:
        raise ValueError("Unsupported anonymization method")
