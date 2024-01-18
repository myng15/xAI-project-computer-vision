# train_util.py

import torch
from evaluation import evaluate_model
from torch.utils.data import DataLoader

def save_model(model, accuracy, device):
    """Saves the trained model to a specified filepath."""
    model_filepath = f"model_{accuracy:.3f}.pth"

    if device == "cuda":
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save(model_state_dict, model_filepath)
    print(f"Model saved successfully to: {model_filepath}")

def train_and_evaluate(model, train_embeddings, train_labels, test_embeddings, test_labels, device="cpu", num_epochs=20, batch_size=64):
    """Trains the model and evaluates its performance on the test set."""

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for i in range(0, len(train_embeddings), batch_size):
            inputs = train_embeddings[i:i+batch_size]
            targets = train_labels[i:i+batch_size]
            targets = torch.from_numpy(targets)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    accuracy = evaluate_model(model, test_embeddings, test_labels)
    return accuracy
