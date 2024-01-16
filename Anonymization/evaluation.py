from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler

def check_reconstruction(original_embeddings, anonymized_embeddings):
    """
    Check the reconstruction error between original and anonymized embeddings.

    Parameters:
    - original_embeddings: PyTorch tensor, the original set of embeddings
    - anonymized_embeddings: PyTorch tensor, the anonymized set of embeddings

    Returns:
    - float: Mean Squared Error (MSE) between normalized original and anonymized embeddings
    """
    # Flatten and normalize the embeddings
    scaler = StandardScaler()
    original_normalized = scaler.fit_transform(original_embeddings.flatten().reshape(-1, 1)).flatten()
    anonymized_normalized = scaler.transform(anonymized_embeddings.flatten().reshape(-1, 1)).flatten()

    # Calculate mean squared error
    mse = mean_squared_error(original_normalized, anonymized_normalized)

    return mse

def calculate_silhouette_score(anonymized_embeddings, original_labels):
    """
    Calculate the Silhouette Score for the original embeddings.

    Parameters:
    - anonymized_embeddings: PyTorch tensor, the anonymized set of embeddings
    - original_labels: PyTorch tensor, the original labels corresponding to the embeddings

    Returns:
    - float: Silhouette Score for the anonymized embeddings
    """
    scaler = StandardScaler()
    anonymized_normalized = scaler.fit_transform(anonymized_embeddings)
    silhouette_avg = silhouette_score(anonymized_normalized, original_labels)
    return silhouette_avg


def check_embedding_overlap(original_embeddings, anonymized_embeddings):
    """
    Check if any of the anonymized embeddings can be found in the original set of embeddings.

    Parameters:
    - original_embeddings: PyTorch tensor, the original set of embeddings
    - anonymized_embeddings: PyTorch tensor, the anonymized set of embeddings

    Returns:
    - bool: True if any anonymized embedding is found in the original set, False otherwise
    """
    original_set = set(map(tuple, original_embeddings))
    anonymized_set = set(map(tuple, anonymized_embeddings))

    return any(embedding in original_set for embedding in anonymized_set)
