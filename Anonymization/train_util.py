# train_util.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from evaluation import evaluate_model
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_embeddings(embeddings):
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    return normalized_embeddings

def train_and_evaluate(model, train_dataloader, test_embeddings, test_labels, num_epochs=20, batch_size=32, device=device):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert test_embeddings to PyTorch tensor
    test_embeddings = torch.as_tensor(test_embeddings, dtype=torch.float32).to(device)

    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Clear intermediate variables
            del inputs, targets, outputs
            torch.cuda.empty_cache()

    accuracy = evaluate_model(model, test_embeddings, test_labels)
    return accuracy