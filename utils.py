import torch
from torchinfo import summary

import numpy as np
import random
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'Random seed {seed} has been set.')


# Print a summary using torchinfo
def print_model_summary(model, batch_size):
    print(summary(model=model,
            input_size=(batch_size, 3, 32, 32), # make sure this is "input_size", not "input_shape"
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    ))


# Check if your system has cuda gpu or only cpu and get the currently available one
def get_available_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to transfer from CPU to GPU
def move_to_device(tensor, device):
    # Move all individual tensors to chosen device, e.g. from cpu to gpu
    if isinstance(tensor, (list, tuple)):
        return [move_to_device(element, device) for element in tensor]
    return tensor.to(device, non_blocking=True)


# Execute transfer from CPU to GPU for each device
class DeviceDataLoader():
    # Wrap a dataloader to move data to a device
    def __init__(self, dataloader, device):
        self.dl = dataloader
        self.device = device

    def __iter__(self):
        # Transfer each batch to chosen device and return
        for i in self.dl:
            yield move_to_device(i, self.device)

    def __len__(self):
        # Return the number of batches
        return len(self.dl)


class BestModelSaver:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least loss, then save the
    model state.
    """

    def __init__(
        self, epoch_val_loss_min=np.Inf
    ):
        self.epoch_val_loss_min = epoch_val_loss_min

    def __call__(
            self, epoch_val_loss,
            epoch, trained_on_cifar10, model, model_name, optimizer, loss_func
    ):
        output_folder = 'cifar10' if trained_on_cifar10 else 'cifar100'

        if epoch_val_loss <= self.epoch_val_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model for epoch {}...'.format(
                self.epoch_val_loss_min,
                epoch_val_loss,
                epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func,
            }, f'outputs/{output_folder}/best_model_{model_name}_{output_folder}.pth')
            self.epoch_val_loss_min = epoch_val_loss


def save_final_model(epochs, trained_on_cifar10, model, model_name, optimizer, loss_func):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")

    output_folder = 'cifar10' if trained_on_cifar10 else 'cifar100'

    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func,
                }, f'outputs/{output_folder}/final_model_{model_name}_{output_folder}.pth')


def save_plots(trained_on_cifar10, model_name, epochs, results):
    """
    Function to save the loss and accuracy plots to disk.
    """

    # Training Loss
    avg_train_loss = []
    for result in results:
        avg_train_loss.append(result['avg_train_loss'])

    # Training Accuracy
    avg_train_acc = []
    for result in results:
        avg_train_acc.append(result['avg_train_acc'])

    # Validation Loss
    avg_val_loss = []
    for result in results:
        avg_val_loss.append(result['avg_valid_loss'])

    # Validation Accuracy
    avg_val_acc = []
    for result in results:
        avg_val_acc.append(result['avg_val_acc'])

    # Epochs count
    epoch_count = []
    for i in range(epochs):
        epoch_count.append(i)

    output_folder = 'cifar10' if trained_on_cifar10 else 'cifar100'

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
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'outputs/{output_folder}/accuracy_{model_name}.png')
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
    plt.savefig(f'outputs/{output_folder}/loss_{model_name}.png')
    plt.show()
