from model import *
from evaluation import check_reconstruction, silhouette_score, check_embedding_overlap
from visualization import visualize_clusters

if __name__ == "__main__":

    # Visualize clusters using t-SNE
    visualize_clusters(test_embeddings_anonymized, test_labels, method='t-SNE')

    print(f'Accuracy on CIFAR-10 Test Dataset: {accuracy * 100:.2f}%')

    # Check reconstruction error
    reconstruction_error = check_reconstruction(test_embeddings_original, test_embeddings_anonymized)
    print(f'Reconstruction Error: {reconstruction_error:.4f}')

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(test_embeddings_anonymized, test_labels)
    print(f'Silhouette Score: {silhouette_avg:.4f}')

    # Check embedding overlap
    has_overlap = check_embedding_overlap(test_embeddings_original, test_embeddings_anonymized)

    if has_overlap:
        print("Anonymized embeddings found in the original set.")
    else:
        print("No overlap between original and anonymized embeddings.")