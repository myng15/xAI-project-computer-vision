# evaluation.py

import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_embeddings, test_labels):
    """
    Evaluate the model on the test set.

    Parameters:
    - model: PyTorch model
    - test_embeddings: PyTorch tensor, the test set of embeddings
    - test_labels: PyTorch tensor, the labels corresponding to the test embeddings

    Returns:
    - float: Accuracy of the model on the test set
    """
    with torch.no_grad():
        model.eval()
        test_outputs = model(test_embeddings)
        _, predicted_labels = torch.max(test_outputs, 1)
        accuracy = accuracy_score(test_labels, predicted_labels)

    return accuracy