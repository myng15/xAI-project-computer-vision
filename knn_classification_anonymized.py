from utils import get_available_device, seed_all
from knn_utils import train_knn_classifier, evaluate_knn_classifier
from visualization import visualize_embeddings, plot_anonymization_accuracy_vs_error
from anonymization import anonymize_embeddings
from anonymization_evaluation import tune_anonymization_parameters

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
parser.add_argument('-n', '--n_neighbors', type=int,
                    help='number of nearest neighbors for kNN classifier')
parser.add_argument('-a', '--anonym_method', type=str,
                    help='name of the method used to anonymize the embeddings')
parser.add_argument('-ns', '--noise_scale', type=float, default=None,
                    help='noise scale for anonymization')
parser.add_argument('-nc', '--n_components', type=int, default=None,
                    help='number of PCA components for anonymization')
parser.add_argument('-md', '--max_dist', type=float, default=0.5,
                    help='maximum distance between two samples for density-based clustering anonymization')
parser.add_argument('-ms', '--min_samples', type=int, default=5,
                    help='minimum number of samples for density-based clustering anonymization')
parser.add_argument('--tuning', action="store_true",
                    help='tuning mode for anonymization hyperparameters')
parser.add_argument('--noise_scale_tuning', type=float, nargs='+', default=None,
                    help='noise scale values (as a list) for tuning anonymization')
parser.add_argument('--n_components_tuning', type=int, nargs='+', default=None,
                    help='number of PCA components values (as a list) for tuning anonymization')
parser.add_argument('-ve', '--visualize_embeds', action="store_true",
                    help='visualize extracted train and test embeddings')
args = vars(parser.parse_args())

num_classes = args['num_classes']
model_name = args['model_name']
cp_datetime = args['cp_datetime']
if args['optim_code']:
    optim_code = args['optim_code']
else:
    optim_code = ''
n_neighbors = args['n_neighbors']
anonym_method = args['anonym_method']
noise_scale = args['noise_scale']
n_components = args['n_components']
tuning = args['tuning']
noise_scale_tuning = args['noise_scale_tuning']
n_components_tuning = args['n_components_tuning']
visualize_embeds = args['visualize_embeds']

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
    # Load original train and test embeddings
    if model_name == 'DINOv2':
        original_train_embeddings_data = np.load(
            f'embeddings/DinoV2-CIFAR10-CIFAR100/CIFAR{num_classes}-DINOV2-BASE/train.npz')
        original_test_embeddings_data = np.load(
            f'embeddings/DinoV2-CIFAR10-CIFAR100/CIFAR{num_classes}-DINOV2-BASE/test.npz')

    else:
        original_train_embeddings_data = np.load(
            f'embeddings/{output_folder}/train_{model_name}{optim_code}_{cp_datetime}.npz')
        original_test_embeddings_data = np.load(
            f'embeddings/{output_folder}/test_{model_name}{optim_code}_{cp_datetime}.npz')

    train_embeddings, train_labels, train_filenames = original_train_embeddings_data['embeddings'], original_train_embeddings_data['labels'], original_train_embeddings_data['filenames']
    test_embeddings, test_labels, test_filenames = original_test_embeddings_data['embeddings'], original_test_embeddings_data['labels'], original_test_embeddings_data['filenames']

    if tuning:
        reconstruction_errors, accuracy_losses = tune_anonymization_parameters(
            train_embeddings, test_embeddings, train_labels, test_labels, n_neighbors,
            anonym_method, noise_scale_tuning, n_components_tuning)
        plot_anonymization_accuracy_vs_error(output_folder, model_name, cp_datetime, optim_code, anonym_method, noise_scale_tuning, n_components_tuning, reconstruction_errors, accuracy_losses)
    else:
        # Anonymize train and test embeddings
        train_embeddings_anonymized = anonymize_embeddings(train_embeddings, anonym_method,
                                                           noise_scale=noise_scale, n_components=n_components)
        test_embeddings_anonymized = anonymize_embeddings(test_embeddings, anonym_method,
                                                          noise_scale=noise_scale, n_components=n_components)
        print("ANONYMIZATION OF EMBEDDINGS FINISHED.")

        # Save anonymized embeddings and labels to a .npz file
        np.savez(f'embeddings/{output_folder}/train_anonymized_{anonym_method}_{model_name}{optim_code}_{cp_datetime}.npz',
                 filenames=train_filenames, embeddings=train_embeddings_anonymized, labels=train_labels)
        np.savez(f'embeddings/{output_folder}/test_anonymized_{anonym_method}_{model_name}{optim_code}_{cp_datetime}.npz',
                 filenames=test_filenames, embeddings=test_embeddings_anonymized, labels=test_labels)

        # Train KNN classifier on the extracted embeddings
        knn_classifier = train_knn_classifier(train_embeddings_anonymized, train_labels, n_neighbors=n_neighbors)

        # Evaluate KNN classifier on the test set
        knn_accuracy = evaluate_knn_classifier(knn_classifier, test_embeddings_anonymized, test_labels)
        print(f"KNN Classifier Accuracy on Anonymized Test Set: {knn_accuracy * 100:.2f}%")

        # Visualize train and test embeddings
        if visualize_embeds:
            visualize_embeddings(train_embeddings_anonymized, train_labels, test_embeddings_anonymized, test_labels,
                                 output_folder, model_name, cp_datetime, optim_code=optim_code,
                                 anonymized=True, anonym_method=anonym_method, noise_scale=noise_scale, n_components=n_components,
                                 method='t-SNE', plot_n_components=2)

