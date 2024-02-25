import numpy as np

import sys
import os
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from utils import get_available_device, seed_all
from online_learner import OnlineLearner
from anonymization.anonymization_evaluation import calculate_fid

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
parser.add_argument('--n_rounds', type=int,
                    help='number of online learning rounds')
parser.add_argument('-b', '--batch_size', type=int,
                    help='number of samples per batch to load')
parser.add_argument('--classifier', type=str, default='',
                    help='type of classifier to be used')
parser.add_argument('--n_neighbors', type=int,
                    help='number of nearest neighbors for kNN classifier')
parser.add_argument('--generative_replay', action="store_true",
                    help='train kNN classifier with generative replay')
parser.add_argument('--n_epochs_gan', type=int, default=None,
                    help='number of epochs for training GAN model used in generative replay')
parser.add_argument('--batch_size_gan', type=int, default=None,
                    help='batch size for training GAN model used in generative replay')
parser.add_argument('--memory_sample_percentage', type=float, default=1.0,
                    help='memory sample percentage for training kNN classifer with memory replay')
args = vars(parser.parse_args())

num_classes = args['num_classes']
model_name = args['model_name']
cp_datetime = args['cp_datetime']
optim_code = args['optim_code']
n_rounds = args['n_rounds']
batch_size = args['batch_size']
n_neighbors = args['n_neighbors']
classifier = args['classifier']
generative_replay = args['generative_replay']
n_epochs_gan = args['n_epochs_gan']
batch_size_gan = args['batch_size_gan']
memory_sample_percentage = args['memory_sample_percentage']

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

if __name__ == "__main__":
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

    d_conv_dim = 32
    g_conv_dim = 32
    latent_dim = 100
    embedding_size = 768
    lr = 0.0002
    online_learner = OnlineLearner(n_neighbors,
                                   generative_replay, d_conv_dim, g_conv_dim, latent_dim, embedding_size, lr, device)

    # Online learning loop
    accuracies = []
    training_time_list = []
    inference_time_list = []
    fid_train_list = []

    for round in range(n_rounds):
        # Generate mini-batch with 10 samples from each of the 10 CIFAR-10 classes
        # batch_embeddings, batch_labels = online_learner.generate_real_data_batch(train_embeddings, train_labels,
        #                                                                         num_classes, batch_size)

        print(f"Round {round + 1:03d}/{n_rounds:03d}:")

        batch_embeddings, batch_labels, fid_train = online_learner.generate_online_domain_batch(
            train_embeddings, train_labels,
            num_classes, batch_size,
            latent_dim, n_epochs_gan, batch_size_gan,
            memory_sample_percentage)

        if classifier == 'NCM':
            # Train kNN classifier with mini-batch data
            training_time = online_learner.train_ncm(batch_embeddings, batch_labels)

            # Evaluate kNN classifier on test set
            accuracy, inference_time = online_learner.evaluate_ncm(test_embeddings, test_labels)
        elif classifier == 'kNN-NCM':
            # Train kNN classifier with mini-batch data
            training_time = online_learner.train_knn_ncm(batch_embeddings, batch_labels)

            # Evaluate kNN classifier on test set
            accuracy, inference_time = online_learner.evaluate_knn_ncm(test_embeddings, test_labels)
        elif classifier == 'FC':
            # Train kNN classifier with mini-batch data
            training_time = online_learner.train_fc(batch_embeddings, batch_labels)

            # Evaluate kNN classifier on test set
            accuracy, inference_time = online_learner.evaluate_fc(test_embeddings, test_labels)
        else:
            # Train kNN classifier with mini-batch data
            training_time = online_learner.train_knn(batch_embeddings, batch_labels)

            # Evaluate kNN classifier on test set
            accuracy, inference_time = online_learner.evaluate_knn(test_embeddings, test_labels)

        training_time_list.append(training_time)
        inference_time_list.append(inference_time)
        accuracies.append(accuracy)
        if generative_replay:
            fid_train_list.append(fid_train)

        # logging
        print(f"KNN Classifier Accuracy = {accuracy * 100:.2f}%")

    # Calculate average accuracy and average forgetting
    avg_training_time = online_learner.average_training_time(training_time_list)
    avg_inference_time = online_learner.average_inference_time(inference_time_list)
    avg_acc_ci, avg_forgetting_ci, avg_bwtp_ci, avg_fwt_ci, corr_dict = online_learner.compute_end_performance_ci(accuracies)

    if generative_replay:
        max_fid = np.max(np.abs(fid_train_list))
        if max_fid != 0:
            normalized_fid_train_list = [fid / max_fid for fid in fid_train_list]
        else:
            normalized_fid_train_list = fid_train_list
        avg_fid_train = np.mean(normalized_fid_train_list)


    print('-' * 50)
    print("KNN Classifier in Online Domain Incremental setting: ")
    if generative_replay:
        print(f"Average FID of the generative model: {avg_fid_train:.4f}")
    print(f"Average Accuracy CI: {avg_acc_ci[0] * 100:.2f}% \u00B1 {avg_acc_ci[1] * 100:.2f}%")
    print(f"Average Forgetting CI: {avg_forgetting_ci[0] * 100:.2f}% \u00B1 {avg_forgetting_ci[1] * 100:.2f}%")
    print(f"Average Positive Backward Transfer CI: {avg_bwtp_ci[0] * 100:.2f}% \u00B1 {avg_bwtp_ci[1] * 100:.2f}%")
    print(f"Average Forward Transfer CI: {avg_fwt_ci[0] * 100:.2f}% \u00B1 {avg_fwt_ci[1] * 100:.2f}%")
    for key, value in corr_dict.items():
        print(f"{key} Correlation: {value:.4f}")

    print(f"Average Training Time: {avg_training_time:.4f} seconds")
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
