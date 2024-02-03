import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import textwrap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

plt.style.use('ggplot')

from utils import get_available_device, move_to_device


def save_plots(num_classes, current_datetime, model_name, epochs, results, optim_code='', plot_lrs=False):
    """
    Function to save the loss and accuracy plots to disk.
    """

    # Training Loss
    avg_train_loss = []
    avg_train_acc = []
    avg_val_loss = []
    avg_val_acc = []
    lrs = []

    for result in results:
        avg_train_loss.append(result['avg_train_loss'])
        avg_train_acc.append(result['avg_train_acc'] * 100)
        avg_val_loss.append(result['avg_valid_loss'])
        avg_val_acc.append(result['avg_val_acc'] * 100)
        lrs.append(result['lrs'])

    # Epochs count
    epoch_count = []
    for i in range(epochs):
        epoch_count.append(i + 1)

    # file name parameters
    if num_classes == 10:
        output_folder = 'cifar10'
    elif num_classes == 100:
        output_folder = 'cifar100'
    else:
        output_folder = None

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        epoch_count, avg_train_acc, color='green', linestyle='-',
        label='train'
    )
    plt.plot(
        epoch_count, avg_val_acc, color='blue', linestyle='-',
        label='validation'
    )
    plt.title("Accuracy per epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f'outputs/{output_folder}/accuracy_{model_name}{optim_code}_{current_datetime}.png')
    plt.show()

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        epoch_count, avg_train_loss, color='orange', linestyle='-',
        label='train'
    )
    plt.plot(
        epoch_count, avg_val_loss, color='red', linestyle='-',
        label='validation'
    )
    plt.title("Loss per epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/{output_folder}/loss_{model_name}{optim_code}_{current_datetime}.png')
    plt.show()

    # learning rate plots
    if plot_lrs:
        plt.figure(figsize=(10, 7))
        plt.plot(epoch_count, lrs, color='magenta', linestyle='-')
        plt.title("Learning rate per epoch")
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        plt.savefig(f'outputs/{output_folder}/lr_{model_name}{optim_code}_{current_datetime}.png')
        plt.show()


def denormalize(img, mean, std):
    """
    Helper function to denormalize an image.
    """
    for i in range(3):  # 3 color channels
        img[i] = img[i] * std[i] + mean[i]
    return img


def imshow(img, mean, std):
    """
    Helper function to display an image.
    """
    # Denormalize the image tensor back to the range [0, 1]
    img = denormalize(img, mean, std)
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


def visualize_knn_results(model, test_dl, knn_classifier, classes,
                          output_folder, model_name, cp_datetime, optim_code):
    """
    Function to visualize a test batch of images that are embedded by the specified model and classified by kNN.
    """
    model.eval()
    dataiter = iter(test_dl)
    images, labels = next(dataiter)
    device = get_available_device()
    images = move_to_device(images, device)

    # Calculate mean and std for denormalization
    mean = [round(m.item(), 4) for m in images.mean([0, 2, 3])]
    std = [round(s.item(), 4) for s in images.std([0, 2, 3])]

    with torch.no_grad():
        if hasattr(model, 'get_embeddings') and callable(getattr(model, 'get_embeddings')):
            embeddings = model.get_embeddings(images)
        else:
            get_embeddings = nn.Sequential(*list(model.children())[:-1])
            embeddings = get_embeddings(images)

        flattened_embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()

        # Predict labels using kNN classifier
        predictions = knn_classifier.predict(flattened_embeddings)

    # Visualize sample test results
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, int(20 / 2), idx + 1, xticks=[], yticks=[])
        image = images[idx].cpu().numpy()
        imshow(image, mean, std)
        ax.set_title("{} (True: {})".format(classes[predictions[idx]], classes[labels[idx]]),
                     color=("green" if predictions[idx] == labels[idx].item() else "red"))

    # Save the plot
    plt.savefig(f'outputs/{output_folder}/knn_sample_{model_name}{optim_code}_{cp_datetime}.png')
    plt.show()
    print('Visualization of kNN Sample Results Saved.')


def visualize_embeddings(train_embeddings, train_labels, test_embeddings, test_labels,
                         output_folder, model_name, cp_datetime, optim_code='',
                         anonymized=False, anonym_method='', noise_scale=None, n_components=None,
                         max_dist=None, min_samples=None,
                         n_clusters=None, assign_labels='',
                         method='t-SNE', plot_n_components=2):
    """
    Function to visualize train and test embedding databases.
    """
    if method == 't-SNE':
        tsne = TSNE(n_components=plot_n_components, perplexity=30, n_iter=300, verbose=0,
                    random_state=42)  # n_iter must be at least 250 (Default = 1000)
        train_reduced_embeddings = tsne.fit_transform(train_embeddings)
        test_reduced_embeddings = tsne.fit_transform(test_embeddings)
    elif method == 'PCA':
        pca = PCA(n_components=plot_n_components)
        train_reduced_embeddings = pca.fit_transform(train_embeddings)
        test_reduced_embeddings = pca.fit_transform(test_embeddings)
    else:
        raise ValueError(f"Unsupported visualization method: {method}")

    # Create subplots for train and test embeddings
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Define color palette
    if len(np.unique(train_labels)) == 10:  # CIFAR10
        palette = sns.color_palette("pastel")
    else:
        # Define custom palette
        pastel_hues = sns.color_palette("pastel", 10)
        dark_hues = sns.color_palette("dark", 10)
        husl_hues = sns.color_palette("husl", 10)
        bright_hues = sns.color_palette("bright", 10)
        tab10_hues = sns.color_palette("tab10", 10)
        deep_hues = sns.color_palette("deep", 10)
        paired_hues = sns.color_palette("Paired")
        colorcet_palette = sns.color_palette(cc.glasbey, n_colors=31)

        palette = pastel_hues + dark_hues + husl_hues + bright_hues + tab10_hues + deep_hues + paired_hues + colorcet_palette

    # Plot train embeddings
    sns.scatterplot(x=train_reduced_embeddings[:, 0], y=train_reduced_embeddings[:, 1], hue=train_labels,
                    palette=palette, alpha=0.5, ax=ax[0])
    ax[0].set_title(f'{method} Visualization of Train Embeddings')
    ax[0].set_xlabel(f'{method} Component 1')
    ax[0].set_ylabel(f'{method} Component 2')

    # Plot test embeddings
    sns.scatterplot(x=test_reduced_embeddings[:, 0], y=test_reduced_embeddings[:, 1], hue=test_labels,
                    palette=palette, alpha=0.5, ax=ax[1])
    ax[1].set_title(f'{method} Visualization of Test Embeddings')
    ax[1].set_xlabel(f'{method} Component 1')
    ax[1].set_ylabel(f'{method} Component 2')

    if anonymized:
        for axis in ax:
            axis.set_xlim(-13, 13)
            axis.set_ylim(-13, 13)

    # Format legends
    fontsize = 'small'
    title_fontsize = 'medium'
    if len(np.unique(train_labels)) == 10:  # CIFAR10
        num_columns_train = 1
        num_columns_test = 1
    else:
        num_columns_train = int(np.ceil(len(np.unique(train_labels)) / 25))
        num_columns_test = int(np.ceil(len(np.unique(test_labels)) / 25))
        fontsize = 'x-small'
        title_fontsize = 'small'

    ax[0].legend(title='Class labels', ncols=num_columns_train, fontsize=fontsize, title_fontsize=title_fontsize,
                 borderpad=0.2, labelspacing=0.3, columnspacing=0.4)
    ax[1].legend(title='Class labels', ncols=num_columns_test, fontsize=fontsize, title_fontsize=title_fontsize,
                 borderpad=0.2, labelspacing=0.3, columnspacing=0.4)

    # Save the plot
    plt.savefig(f'embeddings/{output_folder}/embeddings'
                + (f'_anonymized' if anonymized else '')
                + (f'_{anonym_method}' if anonym_method else '')
                + (f'_{n_components}' if n_components else '')
                + (f'_{max_dist}' if max_dist else '')
                + (f'_{min_samples}' if min_samples else '')
                + (f'_{n_clusters}' if n_clusters else '')
                + (f'_{assign_labels}' if assign_labels else '')
                + (f'_{noise_scale}' if noise_scale else '')
                + f'_{model_name}{optim_code}_{cp_datetime}.png')
    plt.show()
    print('Visualization of Embeddings Saved.')


def plot_anonymization_accuracy_vs_error(output_folder, model_name, cp_datetime, optim_code,
                                         anonym_method, noise_scale_tuning,
                                         max_dist_tuning, min_samples_tuning, n_clusters_tuning, assign_labels_tuning,
                                         reconstruction_errors_train,
                                         accuracy_losses, test_set='Anonymized Test Embeddings'):
    """
    Plot accuracy loss vs. reconstruction error with annotations.

    Parameters:
    - output_folder (str): cifar10 or cifar100
    - model_name (str): CNN model used to extract embeddings
    - cp_datetime (str): Datetime when the selected checkpoint of training the selected model was saved
    - optim_code (str): Code indicating all optimizations applied (compared to the selected default optimizations)
    - anonym_method (str): Anonymization method to be tested.
    - noise_scale_tuning (list): List of all noise_scale values.
    - n_components_tuning (list): List of all n_components values.
    - max_dist_tuning (list): List of all max_dist values.
    - min_samples_tuning (list): List of all min_samples values.
    - n_clusters_tuning (list): List of all n_clusters values.
    - assign_labels_tuning (list): List of all assign_labels values.
    - reconstruction_errors_train (list): List of reconstruction errors of anonymized train embeddings.
    - accuracy_losses (list): List of kNN accuracy losses on the selected test set.
    - test_set (str): Test set that the kNN classifier classifies
    """
    # Plotting for each combination
    plt.figure(figsize=(12, 6))
    # Plot reconstruction errors of train embeddings
    plt.plot(reconstruction_errors_train, accuracy_losses, marker='o', color='blue', label='Train Set')

    plt.xlabel('Reconstruction Error')
    plt.ylabel('Accuracy Loss (%)')
    plt.title(f'Accuracy Loss on {test_set} vs. Reconstruction Error')
    plt.suptitle(f'Anonymization Method: {anonym_method}, Model: {model_name}')

    if all(element is None for element in reconstruction_errors_train):
        print("This anonymization method does not support the reconstruction error metrics.")
        return

    # Add text annotations for each data point

    if anonym_method == "pca":
        print("This anonymization method does not support visualizing accuracy loss vs. reconstruction error.")
        return
    #     for error_train, error_test, acc_loss, n_components in zip(
    #             reconstruction_errors_train, reconstruction_errors_test, accuracy_losses, n_components_tuning):
    #         # Create annotation texts for data points
    #         plt.text(error_train, acc_loss, f'(n_components={n_components})', fontsize=7, ha='left', va='bottom')
    #         plt.text(error_test, acc_loss, f'(n_components={n_components})', fontsize=7, ha='left', va='bottom')

    # elif anonym_method == "density-based":
    if anonym_method == "density-based":
        tuning_parameters = []
        for max_dist in max_dist_tuning:
            for min_samples in min_samples_tuning:
                for noise_scale in noise_scale_tuning:
                    tuning_parameters.append((max_dist, min_samples, noise_scale))

        for error_train, acc_loss, (max_dist, min_samples, noise_scale) in zip(
                reconstruction_errors_train, accuracy_losses, tuning_parameters):
            # Create annotation texts for data points
            anno_text = textwrap.fill(f'(max_dist={max_dist}, min_samples={min_samples}, noise_scale={noise_scale})',
                                      width=30)
            plt.text(error_train, acc_loss, anno_text, fontsize=7, ha='left', va='bottom')

    elif anonym_method == "kmeans":
        tuning_parameters = []
        for n_clusters in n_clusters_tuning:
            for assign_labels in assign_labels_tuning:
                if noise_scale_tuning is None:
                    noise_scale_tuning = [0.0]
                for noise_scale in noise_scale_tuning:
                    tuning_parameters.append((n_clusters, assign_labels, noise_scale))

        for error_train, acc_loss, (n_clusters, assign_labels, noise_scale) in zip(
                reconstruction_errors_train, accuracy_losses, tuning_parameters):
            # Create annotation texts for data points
            anno_text = textwrap.fill(
                f'(n_clusters={n_clusters}, assign_labels={assign_labels}, noise_scale={noise_scale})',
                width=30)
            plt.text(error_train, acc_loss, anno_text, fontsize=7, ha='left', va='bottom')

    else:
        for error_train, acc_loss, noise_scale in zip(
                reconstruction_errors_train, accuracy_losses, noise_scale_tuning):
            # Create annotation texts for data points
            plt.text(error_train, acc_loss, f'(noise_scale={noise_scale})', fontsize=7, ha='left', va='bottom')

    # Save the plot
    plt.savefig(
        f'outputs/{output_folder}/anonymization_acc_vs_error_{anonym_method}_{test_set}_{model_name}{optim_code}_{cp_datetime}.png')
    print(f'Plot of KNN Classifier (trained on Anonymized Train Embeddings) Accuracy on {test_set} '
          f'vs. Reconstruction Error Saved.')

    plt.show()

