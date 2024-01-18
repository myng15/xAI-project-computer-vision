# train_util.py

import torch
from sklearn.metrics import accuracy_score

def save_model(model, accuracy, device):
    """Saves the trained model to a specified filepath."""
    model_filepath = f"model_{accuracy:.3f}.pth"

    if device == "cuda":
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save(model_state_dict, model_filepath)
    print(f"Model saved successfully to: {model_filepath}")

def train_and_evaluate(model, train_embeddings, train_labels, test_embeddings, test_labels, device="cpu", num_epochs=30, batch_size=32):
    """Trains the model and evaluates its performance on the test set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for i in range(0, len(train_embeddings), batch_size):
            inputs = torch.from_numpy((train_embeddings[i:i + batch_size].cpu().numpy()))
            targets = train_labels[i:i+batch_size]
            targets = torch.from_numpy(targets).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    accuracy = evaluate_model(model, test_embeddings, test_labels, device)
    return accuracy

def evaluate_model(model, test_embeddings, test_labels, device="cpu"):
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
        test_embeddings = torch.as_tensor(test_embeddings, dtype=torch.float32).to(device)  # Convert to PyTorch tensor
        model.eval()
        test_outputs = model(test_embeddings).to(device)
        _, predicted_labels = torch.max(test_outputs, 1)

        accuracy = accuracy_score(test_labels, predicted_labels)

    return accuracy
