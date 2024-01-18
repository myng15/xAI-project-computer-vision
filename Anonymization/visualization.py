# visualization.py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_clusters(embeddings, labels, method='t-SNE', n_components=2):
    if method == 't-SNE':
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
    elif method == 'PCA':
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
    else:
        raise ValueError(f"Unsupported visualization method: {method}")


    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=20)
    plt.title(f'Clusters after Anonymization ({method})')
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.show()