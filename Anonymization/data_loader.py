# data_loader.py

import numpy as np
from torch import from_numpy


def load_data(file_path):
    data = np.load(file_path)
    filenames = data['filenames']
    embeddings = from_numpy(data['embeddings'])
    labels = data['labels']
    return filenames, embeddings, labels
