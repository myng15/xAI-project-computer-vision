import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

import sys
import os
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from utils import get_available_device, DeviceDataLoader


def create_data_loaders_knn(train_data, test_data, batch_size, num_workers):
    # Prepare data loaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Move all the tensors to GPU if available
    device = get_available_device()
    train_dl = DeviceDataLoader(train_loader, device)
    test_dl = DeviceDataLoader(test_loader, device)
    return train_dl, test_dl


def extract_embeddings(model, dataloader, classes, output_subfolder):
    model.eval()
    embeddings = []
    labels = []
    filenames_list = []

    last_idx = 0  # Initialize last filename index from the previous batch

    with torch.no_grad():
        for inputs, target in dataloader:
            if hasattr(model, 'get_embeddings') and callable(getattr(model, 'get_embeddings')):
                outputs = model.get_embeddings(inputs)
            else:
                get_embeddings = nn.Sequential(*list(model.children())[:-1])
                outputs = get_embeddings(inputs)
            # Flatten the 3D array into 2D
            outputs = outputs.view(outputs.size(0), -1)
            embeddings.append(outputs.cpu().numpy())
            labels.append(target.cpu().numpy())

            # Create dynamic file names based on class labels
            class_labels = target.cpu().numpy()
            for idx, label in enumerate(class_labels):
                filenames_list.append(f'{output_subfolder}/{classes[label]}/{last_idx + 1:05d}.png')
                last_idx += 1

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    return embeddings, labels, filenames_list


def train_knn_classifier(embeddings, labels, n_neighbors):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(embeddings, labels)
    return classifier


def evaluate_knn_classifier(classifier, test_embeddings, test_labels):
    predictions = classifier.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy
