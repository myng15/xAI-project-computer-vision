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

- **Epsilon**: Varied in the range of 1.0 to 1.5 with increments of 0.1.
- **Min Samples**: Kept constant at 3 throughout all iterations.
- **Noise Scale**: Varied from 1.0 to 1.5 in increments of 0.25.

Now, let's analyze the outcomes:

1. **Accuracy**: Represents the percentage of correctly classified instances. Higher accuracy values are generally preferred.

   - Accuracy ranges from approximately 78.56% to 94.65%.
   - Generally, there seems to be a negative correlation between epsilon and accuracy.
   - Within a specific epsilon value, an increase in noise scale tends to result in decreased accuracy.

2. **Reconstruction Error**: Indicates how well the reconstructed data aligns with the original data. Higher reconstruction error values are desirable for anonymization.

   - Reconstruction error ranges from approximately 1.9978 to 4.5029.
   - The reconstruction error does not exhibit a clear trend with epsilon or noise scale.

It appears there is a discernible trade-off between accuracy and epsilon, with higher epsilon values corresponding to lower accuracy. The relationship with noise scale is not as straightforward.

# Experimental Results Interpretation (CIFAR100)
<img width="454" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/3a98a94f-05bb-4e30-8023-f45f50d01a3f">

## Negative Accuracy Loss in Anonymization

In the context of the anonymization process, the term "accuracy loss" refers to the difference in accuracy between the original (unanonymized) embeddings and the anonymized embeddings. A negative accuracy loss indicates that the anonymized embeddings outperformed the unanonymized embeddings.

## Interpretation of Negative Accuracy Loss

Consider the data point in the generated graph:

- **Epsilon=0.85, Min Samples=3, Noise Scale=0.5, Accuracy=77.84%, Reconstruction Error=0.4998**

In this case, the accuracy of the anonymized embeddings is approximately 77.84%, and the reconstruction error is 0.4998. The key observation is that the accuracy of the anonymized embeddings is higher than that of the original (unanonymized) embeddings. The negative accuracy loss at around -0.1 indicates an improvement in accuracy after the anonymization process.

## Implications

- **Positive Anonymization Impact:** A negative accuracy loss suggests that the anonymization process, under the specified parameters (Epsilon, Min Samples, Noise Scale), has resulted in embeddings that perform better in terms of accuracy compared to the original embeddings.

- **Enhanced Privacy:** Achieving better accuracy with anonymized embeddings implies that privacy-preserving measures, such as noise addition and anonymization techniques, were successful in maintaining or even improving the utility of the data while protecting individual privacy.

## Considerations

- **Parameter Sensitivity:** The interpretation is based on the specific parameters used in the anonymization process (Epsilon, Min Samples, Noise Scale). Different parameter combinations may lead to varied results, and it is essential to carefully choose these parameters based on the desired trade-off between privacy and utility.

- **Visualizing Trends:** It is recommended to visualize trends across different parameter values to gain insights into the impact of anonymization on accuracy and reconstruction error.

## Conclusion

Negative accuracy loss in anonymization, as observed in this specific data point, signifies an encouraging outcome where the anonymized embeddings perform better in terms of accuracy than the original embeddings. This documentation serves to highlight the positive impact of the anonymization process on the utility of the data while preserving privacy.




## Getting Started
To get started, follow these steps:
1. Clone the repository: `git clone https://github.com/DominicLiebel/EmbeddingAnonymization.git`
2. Navigate to the project directory: `cd EmbeddingAnonymization`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Provide: `test_cifar10.npz, train_cifar10.npz, test_cifar100.npz, train_cifar100.npz`
5. Run the main script: `python main.py`

Feel free to explore and modify the code based on your specific requirements.


## First Tries
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/4f288bd4-02eb-4530-af99-8da8cdfbd8c2">
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/cb621191-1809-4a17-8b4c-ca5516c52ca3">
<img width="330" alt="image" src="https://github.com/DominicLiebel/EmbeddingAnonymization/assets/20253502/7474b35a-0ec8-45df-9bde-4224b04af091">
