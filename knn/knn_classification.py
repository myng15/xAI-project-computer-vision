import torch

import sys
sys.path.append('..')

import sys
import os
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from models import get_model
from utils import get_available_device, seed_all
from datasets import get_normalized_cifar_datasets
from knn.knn_utils import create_data_loaders_knn, extract_embeddings, train_knn_classifier, evaluate_knn_classifier
from visualization import visualize_embeddings, visualize_knn_results

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--num_classes', type=int,
                    help='10 -> the model to be tested is trained on CIFAR-10; 100 -> CIFAR-100')
parser.add_argument('-dt', '--cp_datetime', type=str,
                    help='datetime of the checkpoint to be tested')
parser.add_argument('-o', '--optim_code', type=str, default='',
                    help='code of the optimization scheme used when training the model to be tested')
parser.add_argument('-m', '--model_name', type=str,
                    help='name of model to be tested')
parser.add_argument('-b', '--batch_size', type=int,
                    help='number of samples per batch to load')
parser.add_argument('-w', '--num_workers', type=int,
                    help='number of subprocesses to use for data loading')
parser.add_argument('--n_neighbors', type=int,
                    help='number of nearest neighbors for kNN classifier')
parser.add_argument('-ve', '--visualize_embeds', action="store_true",
                    help='visualize extracted train and test embeddings')
parser.add_argument('-vk', '--visualize_knn', action="store_true",
                    help='visualize a test batch of images that are embedded by the selected model and '
                         'classified using kNN')
args = vars(parser.parse_args())

num_classes = args['num_classes']
model_name = args['model_name']
cp_datetime = args['cp_datetime']
optim_code = args['optim_code']
batch_size = args['batch_size']
num_workers = args['num_workers']
n_neighbors = args['n_neighbors']
visualize_embeds = args['visualize_embeds']
visualize_knn = args['visualize_knn']

# set seed
seed_all(42)

# get device
device = get_available_device()
print(f"Device: {device}\n")

if num_classes == 10:
    output_folder = 'cifar10'
elif num_classes == 100:
    output_folder = 'cifar100'
else:
    output_folder = None


if __name__ == '__main__':
    if model_name == 'DINOv2':
        dinov2_train_embeddings_data = np.load(
            f'embeddings/DinoV2-CIFAR10-CIFAR100/CIFAR{num_classes}-DINOV2-BASE/train.npz')
        dinov2_test_embeddings_data = np.load(
            f'embeddings/DinoV2-CIFAR10-CIFAR100/CIFAR{num_classes}-DINOV2-BASE/test.npz')

        train_embeddings, train_labels, train_filenames = dinov2_train_embeddings_data['embeddings'], dinov2_train_embeddings_data['labels'], dinov2_train_embeddings_data['filenames']
        test_embeddings, test_labels, test_filenames = dinov2_test_embeddings_data['embeddings'], dinov2_test_embeddings_data['labels'], dinov2_test_embeddings_data['filenames']

        # Train KNN classifier on the extracted embeddings
        knn_classifier = train_knn_classifier(train_embeddings, train_labels, n_neighbors=n_neighbors)

        # Evaluate KNN classifier on the test set
        knn_accuracy = evaluate_knn_classifier(knn_classifier, test_embeddings, test_labels)
        print(f"KNN Classifier Accuracy on Test Embeddings: {knn_accuracy * 100:.2f}%")

    else:
        # Load CIFAR dataset
        train_data, test_data = get_normalized_cifar_datasets(num_classes)
        train_dl, test_dl = create_data_loaders_knn(train_data, test_data, batch_size, num_workers)
        classes = train_data.classes

        model = get_model(model_name, batch_size, num_classes)
        best_model_cp = torch.load(f'outputs/{output_folder}/best_model_{model_name}{optim_code}_{cp_datetime}.pth')
        model.load_state_dict(best_model_cp['model_state_dict'])

        # Extract train embeddings
        train_embeddings, train_labels, train_filenames = extract_embeddings(model, train_dl, classes, output_subfolder='train')

        # Save embeddings and labels to a .npz file
        np.savez(f'embeddings/{output_folder}/train_{model_name}{optim_code}_{cp_datetime}.npz',
                 filenames=train_filenames, embeddings=train_embeddings, labels=train_labels)

        # Extract test embeddings
        test_embeddings, test_labels, test_filenames = extract_embeddings(model, test_dl, classes, output_subfolder='test')

        # Save embeddings and labels to a .npz file
        np.savez(f'embeddings/{output_folder}/test_{model_name}{optim_code}_{cp_datetime}.npz',
                 filenames=test_filenames, embeddings=test_embeddings, labels=test_labels)

        # Train KNN classifier on the extracted embeddings
        knn_classifier = train_knn_classifier(train_embeddings, train_labels, n_neighbors=n_neighbors)

        # Evaluate KNN classifier on the test set
        knn_accuracy = evaluate_knn_classifier(knn_classifier, test_embeddings, test_labels)
        print(f"KNN Classifier Accuracy on Test Embeddings: {knn_accuracy * 100:.2f}%")

        # Visualize sample test results
        if visualize_knn:
            visualize_knn_results(model, test_dl, knn_classifier, classes,
                                  output_folder, model_name, cp_datetime, optim_code)

    # Visualize train and test embeddings
    if visualize_embeds:
        visualize_embeddings(train_embeddings, train_labels, test_embeddings, test_labels,
                             output_folder, model_name, cp_datetime, optim_code=optim_code,
                             method='t-SNE', plot_n_components=2)


