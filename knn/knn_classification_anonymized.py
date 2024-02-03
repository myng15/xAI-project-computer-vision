import sys
import os

# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from utils import get_available_device, seed_all
from visualization import visualize_embeddings, plot_anonymization_accuracy_vs_error
from anonymization.anonymization_methods import anonymize_embeddings
from anonymization.anonymization_evaluation import tune_anonymization_parameters, calculate_anonymization_metrics
from anonymization.anonymization_GAN.GAN_trainer import GANTrainer
from knn.knn_utils import train_knn_classifier

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
parser.add_argument('--n_neighbors', type=int,
                    help='number of nearest neighbors for kNN classifier')
parser.add_argument('--anonym_method', type=str,
                    help='name of the method used to anonymize the embeddings')
parser.add_argument('--noise_scale', type=float, default=None,
                    help='noise scale for anonymization')
parser.add_argument('--n_components', type=int, default=None,
                    help='number of PCA components for anonymization')
parser.add_argument('--max_dist', type=float, default=None,
                    help='maximum distance between two samples for density-based clustering anonymization')
parser.add_argument('--min_samples', type=int, default=None,
                    help='minimum number of samples for density-based clustering anonymization')
parser.add_argument('--n_clusters', type=int, default=None,
                    help='number of clusters for k-means clustering anonymization')
parser.add_argument('--assign_labels', type=str, default='',
                    help='method of assigning labels for k-means clustering anonymization')
parser.add_argument('--tuning', action="store_true",
                    help='tuning mode for anonymization hyperparameters')
parser.add_argument('--noise_scale_tuning', type=float, nargs='+', default=None,
                    help='noise scale values for tuning anonymization')
parser.add_argument('--n_components_tuning', type=int, nargs='+', default=None,
                    help='number of PCA components values for tuning anonymization')
parser.add_argument('--max_dist_tuning', type=float, nargs='+', default=None,
                    help='maximum distance values for tuning anonymization')
parser.add_argument('--min_samples_tuning', type=int, nargs='+', default=None,
                    help='minimum number of samples values for tuning anonymization')
parser.add_argument('--n_clusters_tuning', type=int, nargs='+', default=None,
                    help='number of clusters values for tuning anonymization')
parser.add_argument('--assign_labels_tuning', type=str, nargs='+', default=None,
                    help='label assigning methods for tuning anonymization')
parser.add_argument('-ve', '--visualize_embeds', action="store_true",
                    help='visualize extracted train and test embeddings')
args = vars(parser.parse_args())

num_classes = args['num_classes']
model_name = args['model_name']
cp_datetime = args['cp_datetime']
optim_code = args['optim_code']
n_neighbors = args['n_neighbors']
anonym_method = args['anonym_method']
noise_scale = args['noise_scale']
n_components = args['n_components']
max_dist = args['max_dist']
min_samples = args['min_samples']
n_clusters = args['n_clusters']
assign_labels = args['assign_labels']
tuning = args['tuning']
noise_scale_tuning = args['noise_scale_tuning']
n_components_tuning = args['n_components_tuning']
max_dist_tuning = args['max_dist_tuning']
min_samples_tuning = args['min_samples_tuning']
n_clusters_tuning = args['n_clusters_tuning']
assign_labels_tuning = args['assign_labels_tuning']
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

    train_embeddings, train_labels, train_filenames = original_train_embeddings_data['embeddings'], \
                                                      original_train_embeddings_data['labels'], \
                                                      original_train_embeddings_data['filenames']
    test_embeddings, test_labels, test_filenames = original_test_embeddings_data['embeddings'], \
                                                   original_test_embeddings_data['labels'], \
                                                   original_test_embeddings_data['filenames']

    if tuning:
        reconstruction_errors_train, accuracy_losses_anonymized_test, accuracy_losses_original_test = tune_anonymization_parameters(
            train_embeddings, test_embeddings, train_labels, test_labels, n_neighbors,
            anonym_method, noise_scale_tuning, n_components_tuning,
            max_dist_tuning, min_samples_tuning, n_clusters_tuning, assign_labels_tuning)

        plot_anonymization_accuracy_vs_error(output_folder, model_name, cp_datetime, optim_code,
                                             anonym_method, noise_scale_tuning,
                                             max_dist_tuning, min_samples_tuning,
                                             n_clusters_tuning, assign_labels_tuning,
                                             reconstruction_errors_train,
                                             accuracy_losses=accuracy_losses_anonymized_test,
                                             test_set='Anonymized Test Set')
        plot_anonymization_accuracy_vs_error(output_folder, model_name, cp_datetime, optim_code,
                                             anonym_method, noise_scale_tuning,
                                             max_dist_tuning, min_samples_tuning,
                                             n_clusters_tuning, assign_labels_tuning,
                                             reconstruction_errors_train,
                                             accuracy_losses=accuracy_losses_original_test,
                                             test_set='Original Test Set')

    else:
        if anonym_method == 'kmeans':
            train_embeddings_anonymized, train_labels_anonymized = anonymize_embeddings(
                train_embeddings, train_labels, anonym_method,
                noise_scale=noise_scale,
                n_clusters=n_clusters, assign_labels=assign_labels)
            test_embeddings_anonymized, test_labels_anonymized = anonymize_embeddings(
                test_embeddings, test_labels, anonym_method,
                noise_scale=noise_scale,
                n_clusters=n_clusters, assign_labels=assign_labels)
        if anonym_method == 'gan':
            d_conv_dim = 32
            g_conv_dim = 32
            latent_dim = 100
            embedding_size = 768
            lr = 0.0002
            trainer = GANTrainer(d_conv_dim, g_conv_dim, latent_dim, embedding_size, lr, device)
            _, train_generator, _, _, _, _ = trainer.load_checkpoint(
                f'outputs/{output_folder}/train_dcgan_{model_name}{optim_code}_{cp_datetime}.pth')
            train_embeddings_anonymized = anonymize_embeddings(train_embeddings, train_labels, anonym_method,
                                                               noise_scale=noise_scale,
                                                               generator=train_generator,
                                                               batch_size=len(train_embeddings),
                                                               latent_dim=latent_dim, device=device).cpu()
            _, test_generator, _, _, _, _ = trainer.load_checkpoint(
                f'outputs/{output_folder}/test_dcgan_{model_name}{optim_code}_{cp_datetime}.pth')
            test_embeddings_anonymized = anonymize_embeddings(test_embeddings, test_labels, anonym_method,
                                                              noise_scale=noise_scale,
                                                              generator=test_generator,
                                                              batch_size=len(test_embeddings),
                                                              latent_dim=latent_dim, device=device).cpu()

            knn_classifier = train_knn_classifier(train_embeddings, train_labels, n_neighbors=n_neighbors)
            train_labels_anonymized = knn_classifier.predict(train_embeddings_anonymized)
            test_labels_anonymized = knn_classifier.predict(test_embeddings_anonymized)

        else:
            # Anonymize train and test embeddings
            train_embeddings_anonymized = anonymize_embeddings(train_embeddings, train_labels, anonym_method,
                                                               noise_scale=noise_scale, n_components=n_components,
                                                               max_dist=max_dist, min_samples=min_samples)
            test_embeddings_anonymized = anonymize_embeddings(test_embeddings, test_labels, anonym_method,
                                                              noise_scale=noise_scale, n_components=n_components,
                                                              max_dist=max_dist, min_samples=min_samples)

            train_labels_anonymized = train_labels
            test_labels_anonymized = test_labels

        print("ANONYMIZATION OF EMBEDDINGS FINISHED.")

        # Save anonymized embeddings and labels to a .npz file
        np.savez(
            f'embeddings/{output_folder}/train_anonymized_{anonym_method}_{model_name}{optim_code}_{cp_datetime}.npz',
            embeddings=train_embeddings_anonymized, labels=train_labels_anonymized)
        np.savez(
            f'embeddings/{output_folder}/test_anonymized_{anonym_method}_{model_name}{optim_code}_{cp_datetime}.npz',
            embeddings=test_embeddings_anonymized, labels=test_labels_anonymized)

        # EVALUATE PERFORMANCE METRICS OF THE ANONYMIZATION METHOD
        reconstruction_error_train, accuracy_loss_anonymized_test, accuracy_loss_original_test, \
        variance_retention, projection_robustness = (
            calculate_anonymization_metrics(train_embeddings, test_embeddings, train_labels, test_labels,
                                            train_embeddings_anonymized, test_embeddings_anonymized,
                                            train_labels_anonymized, test_labels_anonymized,
                                            n_neighbors, anonym_method))

        # Visualize train and test embeddings
        if visualize_embeds:
            visualize_embeddings(train_embeddings_anonymized, train_labels_anonymized,
                                 test_embeddings_anonymized, test_labels_anonymized,
                                 output_folder, model_name, cp_datetime, optim_code=optim_code,
                                 anonymized=True, anonym_method=anonym_method,
                                 noise_scale=noise_scale, n_components=n_components,
                                 n_clusters=n_clusters, assign_labels=assign_labels,
                                 method='t-SNE', plot_n_components=2)
