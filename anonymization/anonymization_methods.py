import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans


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
    laplace_noise = torch.tensor(np.random.laplace(loc=0.0, scale=noise_scale, size=embeddings.shape),
                                 dtype=torch.float32)
    anonymized_embeddings = embeddings + laplace_noise
    return anonymized_embeddings


def anonymize_embeddings_gaussian(embeddings, noise_scale=0.1):
    gaussian_noise = torch.tensor(np.random.normal(loc=0.0, scale=noise_scale, size=embeddings.shape),
                                  dtype=torch.float32)
    anonymized_embeddings = embeddings + gaussian_noise
    return anonymized_embeddings


def anonymize_embeddings_permutation(embeddings):
    permutation = torch.randperm(embeddings.shape[1], device="cpu")
    return embeddings[:, permutation]


def anonymize_embeddings_hashing(embeddings, salt="secret_salt"):
    hashed_embeddings = torch.tensor(np.vectorize(hash)(embeddings.cpu().numpy().astype(str) + salt), dtype=torch.long)
    return hashed_embeddings


def anonymize_embeddings_pca(embeddings, n_components):
    if n_components is None:
        print("Anonymization will have no effect if n_components is None.")
        return

    # pca = PCA(n_components=n_components)
    # return torch.tensor(pca.fit_transform(embeddings.cpu().numpy()), dtype=torch.float32)

    # Reshape to (batch_size, num_features) where num_features is 3 * 32 * 32
    flattened_embeddings = embeddings.view(embeddings.size(0), -1)

    # Center the Data
    mean = torch.mean(flattened_embeddings, dim=0)
    centered_embeddings = flattened_embeddings - mean

    # Convert to complex tensor
    centered_embeddings_complex = centered_embeddings.to(torch.complex64)

    # Calculate Covariance Matrix
    covariance_matrix = torch.matmul(centered_embeddings_complex.t(),
                                     centered_embeddings_complex) / centered_embeddings_complex.size(0)

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
    Returns: PyTorch tensor, anonymized embeddings
    """
    embeddings = embeddings.numpy()

    # Compute pairwise distances between embeddings
    # pairwise_distances = np.linalg.norm(embeddings[:, None] - embeddings, axis=-1)
    # pairwise_distances = cdist(embeddings, embeddings)
    pairwise_distances = euclidean_distances(embeddings, embeddings)

    # Initialize an array to store cluster labels (-1 for noise points)
    cluster_labels = np.full(embeddings.shape[0], -1, dtype=int)

    # Initialize a list to store visited points
    visited = [False] * embeddings.shape[0]

    # Initialize a list to store core points
    core_points = []

    # Find core points
    for i in range(embeddings.shape[0]):
        if visited[i]:
            continue

        neighbors = np.where(pairwise_distances[i] <= max_dist)[0]
        if len(neighbors) >= min_samples:
            core_points.append(i)
            for j in neighbors:
                visited[j] = True

    # Assign cluster labels
    cluster_index = 0
    for point_index in core_points:
        if cluster_labels[point_index] != -1:
            continue

        cluster_labels[point_index] = cluster_index
        neighbors = np.where(pairwise_distances[point_index] <= max_dist)[0]

        for neighbor_index in neighbors:
            if cluster_labels[neighbor_index] == -1:
                cluster_labels[neighbor_index] = cluster_index

        cluster_index += 1

    # Generate random perturbations for core points
    # gaussian_noise = np.random.normal(loc=0.0, scale=noise_scale, size=embeddings.shape)

    # Apply perturbations to core points and nearby samples
    anonymized_embeddings = embeddings.copy()
    for cluster_id in range(cluster_index):
        cluster_indices = (cluster_labels == cluster_id)
        # anonymized_embeddings[cluster_indices] += gaussian_noise[cluster_indices]
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=anonymized_embeddings[cluster_indices].shape)
        anonymized_embeddings[cluster_indices] += noise

    return torch.tensor(anonymized_embeddings, dtype=torch.float32)


def anonymize_embeddings_kmeans(embeddings, labels, n_clusters=5000, assign_labels='majority', noise_scale=0.0):
    """
    Anonymize embeddings by aggregating using KMeans clustering.

    Parameters:
    - embeddings (Numpy array): the original embeddings
    - labels (Numpy array): the corresponding labels for original embeddings
    - n_clusters (int): number of clusters to form (must be less than number of embeddings)
    - assign_labels (str): determines how to assign labels to aggregated embeddings.
                     Options: 'majority', 'centroid'

    Returns:
    - aggregated_embeddings (FloatTensor): aggregated embeddings
    - aggregated_labels (FloatTensor): labels for aggregated embeddings
    """
    embeddings = embeddings.numpy()

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)

    # Initialize arrays to store aggregated embeddings and labels
    aggregated_embeddings_list = []
    aggregated_labels_list = []

    for cluster_id in range(n_clusters):
        # Find indices of original embeddings in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        # Aggregating embeddings based on cluster
        aggregated_embedding = np.mean(embeddings[cluster_indices], axis=0)

        # Add Laplace noise to the aggregated embedding if desired
        if noise_scale > 0.0:
            noise = np.random.laplace(loc=0.0, scale=noise_scale, size=aggregated_embedding.shape)
            aggregated_embedding += noise

        # Assigning labels to aggregated embedding
        if assign_labels == 'centroid':
            centroid_label = np.mean(labels[cluster_indices])
            aggregated_label = int(np.round(centroid_label))
        elif assign_labels == 'majority':
            majority_label = np.bincount(labels[cluster_indices]).argmax()
            aggregated_label = majority_label
        else:
            raise ValueError("Invalid method option for assign_labels. Choose 'majority' or 'centroid'.")

        # Append aggregated embedding and label
        aggregated_embeddings_list.append(aggregated_embedding)
        aggregated_labels_list.append(aggregated_label)

    aggregated_embeddings = np.array(aggregated_embeddings_list)
    aggregated_labels = np.array(aggregated_labels_list)

    aggregated_embeddings = torch.from_numpy(aggregated_embeddings)
    aggregated_labels = torch.from_numpy(aggregated_labels)

    return aggregated_embeddings, aggregated_labels


def anonymize_embeddings_gan(generator, num_embeddings_to_generate, latent_dim, device):
    with torch.no_grad():
        latent = torch.randn(num_embeddings_to_generate, latent_dim).to(device)
        synthetic_embeddings = generator(latent)
    return synthetic_embeddings


def anonymize_embeddings(embeddings, labels, method,
                         noise_scale, n_components=None,
                         max_dist=None, min_samples=None,
                         n_clusters=10, assign_labels='majority', generator=None, batch_size=None, latent_dim=None,
                         device="cpu"):
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
        return anonymize_embeddings_density_based(embeddings, max_dist=max_dist, min_samples=min_samples,
                                                  noise_scale=noise_scale)
    elif method == 'kmeans':
        return anonymize_embeddings_kmeans(embeddings, labels, n_clusters=n_clusters, assign_labels=assign_labels,
                                           noise_scale=noise_scale)
    elif method == 'gan':
        return anonymize_embeddings_gan(generator, batch_size, latent_dim, device)
    else:
        raise ValueError(f"Unsupported anonymization method: {method}")
