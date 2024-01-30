import torch
import numpy as np
from sklearn.cluster import DBSCAN
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
    laplace_noise = torch.tensor(np.random.laplace(loc=0.0, scale=noise_scale, size=embeddings.shape), dtype=torch.float32)
    anonymized_embeddings = embeddings + laplace_noise
    return anonymized_embeddings


def anonymize_embeddings_gaussian(embeddings, noise_scale=0.1):
    gaussian_noise = torch.tensor(np.random.normal(loc=0.0, scale=noise_scale, size=embeddings.shape), dtype=torch.float32)
    anonymized_embeddings = embeddings + gaussian_noise
    return anonymized_embeddings


def anonymize_embeddings_permutation(embeddings):
    permutation = torch.randperm(embeddings.shape[1], device="cpu")
    return embeddings[:, permutation]


def anonymize_embeddings_hashing(embeddings, salt="secret_salt"):
    hashed_embeddings = torch.tensor(np.vectorize(hash)(embeddings.cpu().numpy().astype(str) + salt), dtype=torch.long)
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

    torch.set_printoptions(threshold=float('inf'))
    print(embeddings.shape)

    # Perform density-based clustering using DBSCAN
    db = DBSCAN(eps=max_dist, min_samples=min_samples).fit(embeddings)

    # Assign a cluster label to each data point
    cluster_labels = db.labels_
    print(cluster_labels)

    # Generate random perturbations for core points
    unique_labels = np.unique(cluster_labels)
    core_points_indices = db.core_sample_indices_
    gaussian_noise = np.random.normal(loc=0.0, scale=noise_scale, size=embeddings.shape)
    #laplace_noise = np.random.laplace(loc=0.0, scale=noise_scale, size=embeddings.shape)

    print(unique_labels)
    print(gaussian_noise)
    print(gaussian_noise.shape)

    # Apply perturbations to core points and nearby samples
    anonymized_embeddings = embeddings.copy()
    for label in unique_labels:
        cluster_indices = (cluster_labels == label)
        cluster_core_indices = np.intersect1d(np.where(cluster_indices)[0], core_points_indices)

        #perturbations = np.zeros_like(embeddings)
        #perturbations[cluster_core_indices] = gaussian_noise[cluster_core_indices]
        #anonymized_embeddings += perturbations

        anonymized_embeddings[cluster_core_indices] += gaussian_noise[cluster_core_indices]

        print(cluster_core_indices)
        #print(perturbations[cluster_core_indices])
        #print(perturbations)
        print(anonymized_embeddings[cluster_core_indices])

    #return torch.tensor(anonymized_embeddings, dtype=torch.float32)
    return anonymized_embeddings


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
            aggregated_embedding += np.random.laplace(loc=0.0, scale=noise_scale, size=aggregated_embedding.shape)

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

    #return torch.tensor(aggregated_embeddings, dtype=torch.float32), torch.tensor(aggregated_labels, dtype=torch.int)
    return aggregated_embeddings, aggregated_labels


def anonymize_embeddings(embeddings, labels, method,
                         noise_scale, n_components=None,
                         max_dist=None, min_samples=None,
                         n_clusters=10, assign_labels='majority'):
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
    elif method == 'kmeans':
        return anonymize_embeddings_kmeans(embeddings, labels, n_clusters=n_clusters, assign_labels=assign_labels, noise_scale=noise_scale)
    else:
        raise ValueError(f"Unsupported anonymization method: {method}")
