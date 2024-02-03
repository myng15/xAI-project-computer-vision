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
                  input_size=(batch_size, 3, 32, 32),  # make sure the param name is "input_size", not "input_shape"
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
            self, num_classes, current_datetime, model, model_name, optimizer, loss_func, epoch_val_loss,
            epoch, optim_code=''
    ):

        if num_classes == 10:
            output_folder = 'cifar10'
        elif num_classes == 100:
            output_folder = 'cifar100'
        else:
            output_folder = None

        if epoch_val_loss <= self.epoch_val_loss_min:
            print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model for epoch {}...'.format(
                self.epoch_val_loss_min,
                epoch_val_loss,
                epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_func': loss_func,
            }, f'outputs/{output_folder}/best_model_{model_name}{optim_code}_{current_datetime}.pth')
            self.epoch_val_loss_min = epoch_val_loss


def save_final_model(num_classes, current_datetime, model, model_name, optimizer, loss_func, epochs, results, optim_code=''):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")

    if num_classes == 10:
        output_folder = 'cifar10'
    elif num_classes == 100:
        output_folder = 'cifar100'
    else:
        output_folder = None

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_func': loss_func,
        'results': results
    }, f'outputs/{output_folder}/final_model_{model_name}{optim_code}_{current_datetime}.pth')


