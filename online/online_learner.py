import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.stats as stats
import time
import random

import sys
import os


# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from models import FullyConnectedClassifier
from anonymization.anonymization_GAN.GAN_trainer import GANTrainer
from anonymization.anonymization_evaluation import calculate_fid


class OnlineLearner:
    def __init__(self, n_neighbors,
                 generative_replay, d_conv_dim, g_conv_dim, latent_dim, embedding_size, lr, device):
        self.device = device
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        self.fc_classifier = FullyConnectedClassifier(768, 10).to(self.device)
        self.fc_criterion = nn.CrossEntropyLoss()
        self.fc_optimizer = optim.Adam(self.fc_classifier.parameters(), lr=lr)

        self.seen_samples = set()
        self.class_prototypes = {}
        self.generative_replay = generative_replay
        self.gan_trainer = GANTrainer(d_conv_dim, g_conv_dim, latent_dim, embedding_size, lr, device)
        self.memory = {"embeddings": [],
                       "labels": []
                       }  # Memory buffer to store synthetic embeddings (no labels)

    # Classifier: kNN
    def train_knn(self, train_embeddings, train_labels):
        # Train kNN classifier on the extracted embeddings
        start_time = time.time()
        self.knn_classifier.fit(train_embeddings, train_labels)
        training_time = time.time() - start_time
        return training_time

    def evaluate_knn(self, test_embeddings, test_labels):
        # Evaluate kNN classifier on the test set
        start_time = time.time()
        accuracy = self.knn_classifier.score(test_embeddings, test_labels)
        inference_time = time.time() - start_time
        return accuracy, inference_time

    # Helper function for classifier using NCM
    def compute_prototypes(self, train_embeddings, train_labels):
        self.class_prototypes = {}
        unique_labels = np.unique(train_labels)
        for label in unique_labels:
            class_indices = np.where(train_labels == label)[0]
            class_prototype = np.mean(train_embeddings[class_indices], axis=0)
            self.class_prototypes[label] = class_prototype

    # Classifier: NCM
    def train_ncm(self, train_embeddings, train_labels):
        start_time = time.time()
        self.compute_prototypes(train_embeddings, train_labels)
        training_time = time.time() - start_time
        return training_time

    def predict_ncm(self, test_embeddings):
        # Predict labels using Nearest Class Mean (NCM)
        predicted_labels = []
        for sample in test_embeddings:
            # Compute distances to class prototypes
            distances = {label: np.linalg.norm(sample - prototype)
                         for label, prototype in self.class_prototypes.items()}
            # Predict the label of the sample as the label of the closest prototype
            predicted_label = min(distances, key=distances.get)
            predicted_labels.append(predicted_label)
        return np.array(predicted_labels)

    def evaluate_ncm(self, test_embeddings, test_labels):
        start_time = time.time()
        # Evaluate NCM classifier on the test set
        predicted_labels = self.predict_ncm(test_embeddings)
        accuracy = accuracy_score(test_labels, predicted_labels)
        inference_time = time.time() - start_time
        return accuracy, inference_time

    # Classifier: kNN-NCM
    def train_knn_ncm(self, train_embeddings, train_labels):
        start_time = time.time()
        self.compute_prototypes(train_embeddings, train_labels)
        self.knn_classifier.fit(list(self.class_prototypes.values()), list(self.class_prototypes.keys()))
        training_time = time.time() - start_time
        return training_time

    def evaluate_knn_ncm(self, test_embeddings, test_labels):
        start_time = time.time()
        # Compute distances to class prototypes
        distances = np.array([[np.linalg.norm(test_emb - prot) for prot in self.class_prototypes.values()]
                              for test_emb in test_embeddings])
        # Get labels of k nearest prototypes for each test sample
        nearest_labels = np.array(
            [np.array(list(self.class_prototypes.keys()))[np.argsort(dist)[:self.knn_classifier.n_neighbors]]
             for dist in distances])

        # Perform majority voting to get predicted labels
        predicted_labels = np.array([np.argmax(np.bincount(nearest)) for nearest in nearest_labels])
        # Compute accuracy
        accuracy = np.mean(predicted_labels == test_labels)
        inference_time = time.time() - start_time
        return accuracy, inference_time

    # Classifier: Fully-connected
    def train_fc(self, train_embeddings, train_labels, batch_size=32, n_epochs=10):
        start_time = time.time()

        # Create DataLoader
        dataset = TensorDataset(torch.tensor(train_embeddings), torch.tensor(train_labels))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(n_epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.fc_optimizer.zero_grad()

                outputs = self.fc_classifier(inputs)
                loss = self.fc_criterion(outputs, labels)
                loss.backward()
                self.fc_optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_embeddings)
            print(f"FC Classifier Training Epoch {epoch + 1:02d}/{n_epochs:02d}: Loss = {epoch_loss:.4f}")

        training_time = time.time() - start_time
        return training_time

    def evaluate_fc(self, test_embeddings, test_labels):
        start_time = time.time()
        # Convert to PyTorch tensor
        test_inputs = torch.tensor(test_embeddings).to(self.device)
        test_labels = torch.tensor(test_labels).to(self.device)

        # Get predictions
        outputs = self.fc_classifier(test_inputs)
        _, predicted_labels = torch.max(outputs, 1)

        # Calculate accuracy
        correct = (predicted_labels == test_labels).sum().item()
        total = test_labels.size(0)
        accuracy = correct / total

        inference_time = time.time() - start_time
        return accuracy, inference_time

    def generate_real_data_batch(self, train_embeddings, train_labels, num_classes, batch_size=100):
        batch_embeddings = []
        batch_labels = []
        remaining_classes = set(range(num_classes))
        samples_per_class = int(np.floor(batch_size / num_classes))

        # Ensure each round's mini-batch includes samples_per_class samples from each class
        while len(batch_embeddings) < batch_size and remaining_classes:
            class_label = np.random.choice(list(remaining_classes))  # Randomly choose a class
            class_indices = np.where(train_labels == class_label)[0]

            # Exclude seen samples from this class
            available_indices = [idx for idx in class_indices
                                 if (tuple(train_embeddings[idx]), train_labels[idx]) not in self.seen_samples]

            # Check if enough unseen samples are available for this class
            if len(available_indices) >= samples_per_class:
                sampled_indices = np.random.choice(available_indices, size=samples_per_class, replace=False)
                batch_embeddings.extend(train_embeddings[sampled_indices])
                batch_labels.extend(train_labels[sampled_indices])

                # Add the selected samples to the set of seen samples
                for idx in sampled_indices:
                    self.seen_samples.add((tuple(train_embeddings[idx]), train_labels[idx]))

                remaining_classes.remove(class_label)

        return np.array(batch_embeddings), np.array(batch_labels)

    def create_replay_data(self, memory_sample_percentage, real_batch_embeddings, real_batch_labels):
        # Create replay data by combining real data and synthetic data from memory
        if len(self.memory["embeddings"]) > 0:
            if memory_sample_percentage == 1.0:
                memory_embeddings = self.memory["embeddings"]
                memory_labels = self.memory["labels"]
            else:
                # Sample from memory buffer to augment training batch
                sample_size = int(np.floor(len(self.memory["embeddings"]) * memory_sample_percentage))
                memory_embeddings = random.sample(self.memory["embeddings"], sample_size)

                # Get the indices of the sampled embeddings in the original embeddings list
                #sampled_indices = [self.memory["embeddings"].index(em) for em in memory_embeddings]
                sampled_indices = [next(i for i, x in enumerate(self.memory["embeddings"])
                                        if np.array_equal(x, em)) for em in memory_embeddings]
                # Retrieve the corresponding labels using the sampled indices
                memory_labels = [self.memory["labels"][idx] for idx in sampled_indices]

            # Generate batch with a mix of real and synthetic data
            replay_batch_embeddings = np.concatenate(
                (np.array(memory_embeddings), real_batch_embeddings))
            replay_batch_labels = np.concatenate((np.array(memory_labels), real_batch_labels))
        else:
            replay_batch_embeddings = real_batch_embeddings
            replay_batch_labels = real_batch_labels

        return replay_batch_embeddings, replay_batch_labels

    def augment_synthetic_data_memory(self, real_batch_embeddings, batch_size,
                                      latent_dim,n_epochs_gan, batch_size_gan,
                                      replay_batch_embeddings, replay_batch_labels):
        # Train GAN on the incoming data to generate synthetic data that mimic the real data
        _, _, _, _ = self.gan_trainer.fit(real_batch_embeddings, batch_size_gan, n_epochs_gan, latent_dim)

        # Generate synthetic embeddings
        with torch.no_grad():
            latent = torch.randn(batch_size, latent_dim).to(self.device)
            synthetic_batch_embeddings = self.gan_trainer.generator(latent)
            synthetic_batch_embeddings = synthetic_batch_embeddings.cpu().numpy()

            # Evaluate the generative model
            fid_train = calculate_fid(np.array(real_batch_embeddings), synthetic_batch_embeddings)
            #print("FID of the Generative Model Between Incoming Real Data and Synthetic Data: ", fid_train)

            # Save synthetic embeddings of current round's real embeddings for later rounds
            for em in synthetic_batch_embeddings:
                self.memory["embeddings"].append(em)

        # Classify the synthetic embeddings using kNN (except for the first round when the memory is still empty)
        # to get the synthetic labels
        self.knn_classifier.fit(replay_batch_embeddings, replay_batch_labels)
        synthetic_batch_labels = self.knn_classifier.predict(synthetic_batch_embeddings)

        # Save synthetic labels of current round's real labels for later rounds
        self.memory["labels"].extend(synthetic_batch_labels)

        return fid_train

    def generate_online_domain_batch(self, train_embeddings, train_labels, num_classes, batch_size,
                                     latent_dim, n_epochs_gan, batch_size_gan, memory_sample_percentage):
        real_batch_embeddings, real_batch_labels = self.generate_real_data_batch(train_embeddings, train_labels,
                                                                                 num_classes, batch_size)
        fid_train = None

        if self.generative_replay:
            # Create replay data by combining real data and synthetic data from memory
            replay_batch_embeddings, replay_batch_labels = self.create_replay_data(memory_sample_percentage,
                                                                                   real_batch_embeddings,
                                                                                   real_batch_labels)

            # Generate and save synthetic data of current round's real data for later rounds,
            # NOT using for training kNN in this round
            fid_train = self.augment_synthetic_data_memory(real_batch_embeddings, batch_size,
                                                           latent_dim, n_epochs_gan, batch_size_gan,
                                                           replay_batch_embeddings, replay_batch_labels)

            return replay_batch_embeddings, replay_batch_labels, fid_train
        else:
            # Generate batch with only real data
            return real_batch_embeddings, real_batch_labels, fid_train

    def compute_end_accuracy_ci(self, accuracies):
        n_run = len(accuracies)
        t_coef = stats.t.ppf((1 + 0.95) / 2, n_run - 1)  # t coefficient for 95% CI
        avg_acc = (np.mean(accuracies), t_coef * stats.sem(accuracies))
        return avg_acc

    def compute_end_forgetting_ci(self, accuracies):
        forgetting_per_round = []
        for i in range(1, len(accuracies)):
            max_acc_previous = np.max(accuracies[:i], axis=0)
            forgetting_per_round.append(np.mean(max_acc_previous - accuracies[i]))
        t_coef = stats.t.ppf((1 + 0.95) / 2, len(forgetting_per_round) - 1)  # t coefficient for 95% CI
        avg_forgetting = (np.mean(forgetting_per_round), t_coef * stats.sem(forgetting_per_round))
        return avg_forgetting

    def compute_end_positive_backward_transfer_ci(self, accuracies):
        bwt_per_round = [max(accuracies[i - 1] - accuracies[i], 0) for i in range(1, len(accuracies))]
        t_coef = stats.t.ppf((1 + 0.95) / 2, len(bwt_per_round) - 1)  # t coefficient for 95% CI
        avg_bwtp = (np.mean(bwt_per_round), t_coef * stats.sem(bwt_per_round))

        return avg_bwtp

    def compute_end_forward_transfer_ci(self, accuracies):
        fwt_per_round = [max(accuracies[i] - accuracies[i - 1], 0) for i in range(1, len(accuracies))]
        t_coef = stats.t.ppf((1 + 0.95) / 2, len(fwt_per_round) - 1)  # t coefficient for 95% CI
        avg_fwt = (np.mean(fwt_per_round), t_coef * stats.sem(fwt_per_round))
        return avg_fwt

    def compute_end_performance_ci(self, accuracies):
        avg_acc_ci = self.compute_end_accuracy_ci(accuracies)
        avg_forgetting_ci = self.compute_end_forgetting_ci(accuracies)
        avg_bwtp_ci = self.compute_end_positive_backward_transfer_ci(accuracies)
        avg_fwt_ci = self.compute_end_forward_transfer_ci(accuracies)
        return avg_acc_ci, avg_forgetting_ci, avg_bwtp_ci, avg_fwt_ci

    def average_training_time(self, training_time_list):
        return np.mean(training_time_list)

    def average_inference_time(self, inference_time_list):
        return np.mean(inference_time_list)
