# Embedding Anonymization

Continued development on personal GitHub:
https://github.com/DominicLiebel/EmbeddingAnonymization/

--> Outdated readme, see master/main for up to date version.

This Python project aims to anonymize embeddings while maintaining high accuracy with a high reconstruction error.

## Original CIFAR100 Embeddings
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/1da5bc1e-e0fb-4d9f-b83f-a4976925a5b7">



## Measures of Anonymization
- **hasOverlap:** Simple check to see if anonymized embeddings have any overlap with original embeddings.<br>
  e.g.: `No overlap between original and anonymized embeddings.`
- **Reconstruction Error:** Reconstruction error quantifies how well the anonymized embeddings can reconstruct the original embeddings.<br>
  e.g.: `Reconstruction Error: 4.0027`
- **Mean Relative Difference:** Mean relative differences measure the average percentage change between the original and anonymized embeddings for each image.<br>
  e.g.: `Image 1 -> Mean Relative Difference: 69.68424224853516%`

The reconstruction error focuses on the fidelity of the anonymized embeddings compared to the original embeddings, the mean relative differences give an average percentage change, providing a broader understanding of the overall impact on individual embeddings across the dataset. Both metrics are valuable depending on the specific goals and considerations of the anonymization process.

### Density Based Anonymization (CIFAR100)
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/7020740f-63ee-4231-b50e-49fa3cb3ddd6">


<br>
`Epsilon=1.1, Min Samples=3, Noise Scale=1, Accuracy=94.57%,Reconstruction Error=2.0002`



## Different Anonymization Techniques Employed
- `anonymize_embeddings_random(embeddings, noise_factor=0.1)`
- `anonymize_embeddings_laplace(embeddings, epsilon=0.1, device="cpu")`
- `anonymize_embeddings_dp(embeddings, epsilon=0.1, device="cpu")`
- `anonymize_embeddings_permutation(embeddings)`
- `anonymize_embeddings_hashing(embeddings, salt="secret_salt")`
- `anonymize_embeddings_pca(embeddings, n_components=2)`
- `anonymize_embeddings_density_based(embeddings, eps=0.5, min_samples=5, noise_scale=0.01, device="cpu")`

## Project Structure
The project is structured as follows:
- **main.py:** The main script to run the anonymization process.
- **anonymization.py:** Contains different functions for anonymizing embeddings using various techniques.
- **model.py:** Defines the PyTorch model used in the project.
- **train_util.py:** Provides utility functions for training and evaluating the model.
- **evaluation.py:** Contains a function to find the best parameters for anonymization.
- **visualization.py:** Provides a visualization function.
- **data_loader.py:** Contains a function to load the data.


# Experimental Results Interpretation (CIFAR10)
<img width="454" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/2b761cf2-fb24-49c4-ba3d-91cfcfaf77c3">


These results are part of an experimental optimization process where various parameters are systematically altered, and the resulting accuracy and reconstruction error are documented. Let's delve into the interpretation:

--- Further information is available at:
https://github.com/DominicLiebel/EmbeddingAnonymization/
