import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        avg_train_acc.append(result['avg_train_acc']*100)
        avg_val_loss.append(result['avg_valid_loss'])
        avg_val_acc.append(result['avg_val_acc']*100)
        lrs.append(result['lrs'])

    # Epochs count
    epoch_count = []
    for i in range(epochs):
        epoch_count.append(i+1)

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
    for i in range(3): # 3 color channels
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
        ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
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
                         method='t-SNE', plot_n_components=2):
    """
    Function to visualize train and test embedding databases.
    """
    if method == 't-SNE':
        tsne = TSNE(n_components=plot_n_components, perplexity=30, n_iter=300, verbose=0, random_state=42) # n_iter must be at least 250 (Default = 1000)
        train_reduced_embeddings = tsne.fit_transform(train_embeddings)
        test_reduced_embeddings = tsne.fit_transform(test_embeddings)
    elif method == 'PCA':
        pca = PCA(n_components=plot_n_components)
        train_reduced_embeddings = pca.fit_transform(train_embeddings)
        test_reduced_embeddings = pca.fit_transform(test_embeddings)
    else:
        raise ValueError(f"Unsupported visualization method: {method}")

    # Create subplots for train and test embeddings
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot train embeddings
    sns.scatterplot(x=train_reduced_embeddings[:, 0], y=train_reduced_embeddings[:, 1], hue=train_labels,
                    palette='pastel', alpha=0.5, ax=ax[0])
    ax[0].set_title(f'{method} Visualization of Train Embeddings')
    ax[0].set_xlabel(f'{method} Component 1')
    ax[0].set_ylabel(f'{method} Component 2')

    # Plot test embeddings
    sns.scatterplot(x=test_reduced_embeddings[:, 0], y=test_reduced_embeddings[:, 1], hue=test_labels,
                    palette='pastel', alpha=0.5, ax=ax[1])
    ax[1].set_title(f'{method} Visualization of Test Embeddings')
    ax[1].set_xlabel(f'{method} Component 1')
    ax[1].set_ylabel(f'{method} Component 2')

    if anonymized:
        for axis in ax:
            axis.set_xlim(-13, 13)
            axis.set_ylim(-13, 13)

    # Save the plot
    plt.savefig(f'embeddings/{output_folder}/embeddings'
                + (f'_anonymized' if anonymized else '')
                + (f'_{anonym_method}' if anonym_method else '')
                + (f'_{noise_scale}' if noise_scale else '')
                + (f'_{n_components}' if n_components else '')
                + f'_{model_name}{optim_code}_{cp_datetime}.png')
    plt.show()
    print('Visualization of Embeddings Saved.')


def plot_anonymization_accuracy_vs_error(output_folder, model_name, cp_datetime, optim_code,
                                         anonym_method, noise_scale_tuning, n_components_tuning,
                                         reconstruction_errors, accuracy_losses):
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
    - reconstruction_errors (list): List of reconstruction errors.
    - accuracy_losses (list): List of accuracy losses.
    """
    # Plotting for each combination
    plt.figure(figsize=(12, 6))
    plt.plot(reconstruction_errors, accuracy_losses, marker='o')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Accuracy Loss (%)')
    plt.title('Accuracy Loss vs. Reconstruction Error')
    plt.suptitle(f'Anonymization Method: {anonym_method}, Model: {model_name}')

    # Add text annotations for each point with epsilon, min_samples, and noise_scale values
    if "pca" in anonym_method:
        for i, (error, loss, n_components) in enumerate(
                zip(reconstruction_errors, accuracy_losses, n_components_tuning)):
            plt.text(error, loss, f'(n_components={n_components})', fontsize=8, ha='left',
                     va='top')
    #elif "density-based" in anonym_method:
    #    for i, (error, loss, epsilon, min_samples, noise_scale) in enumerate(zip(reconstruction_errors, accuracy_losses, all_epsilons, all_min_samples_values, all_noise_scale_values)):
    #        plt.text(error, loss, f'({epsilon=}, {min_samples=}, {noise_scale=})', fontsize=8, ha='middle', va='bottom')
    else:
        for i, (error, loss, noise_scale) in enumerate(
                zip(reconstruction_errors, accuracy_losses, noise_scale_tuning)):
            plt.text(error, loss, f'(noise_scale={noise_scale})', fontsize=8, ha='left',
                     va='top')

    # Save the plot
    plt.savefig(f'outputs/{output_folder}/anonymization_plot_{anonym_method}_{model_name}{optim_code}_{cp_datetime}.png')
    plt.show()
    print('Plot of Anonymization Accuracy vs Reconstruction Error Saved.')
