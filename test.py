import torch
import torchvision.transforms as transforms

import numpy as np
import argparse

from datasets import get_normalized_cifar_datasets, create_data_loaders
from utils import get_available_device, DeviceDataLoader, seed_all
from models import get_model

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
args = vars(parser.parse_args())

num_classes = args['num_classes']
model_name = args['model_name']
cp_datetime = args['cp_datetime']
if args['optim_code']:
    optim_code = args['optim_code']
else:
    optim_code = ''
batch_size = args['batch_size']
num_workers = args['num_workers']

# set seed
seed_all(42)

# get device
device = get_available_device()
print(f"Device: {device}\n")

# get the training, validation and test_datasets
train_data, test_data = get_normalized_cifar_datasets(num_classes)
# get the test data loader
_, _, test_loader = create_data_loaders(
    train_data, test_data, batch_size, num_workers
)

# Move all the tensors to GPU if available
test_dl = DeviceDataLoader(test_loader, device)
classes = train_data.classes

if num_classes == 10:
    output_folder = 'cifar10'
elif num_classes == 100:
    output_folder = 'cifar100'
else:
    output_folder = None


def compute_accuracy(predicted, labels):
    predictions, predicted_labels = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predicted_labels == labels).item() / len(predicted))


def evaluate(model, dl, loss_func):
    model.eval()

    loss_per_batch, accuracy_per_batch = [], []

    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    for images, labels in dl:
        # start loop
        with torch.no_grad():
            predicted = model(images)
        loss_per_batch.append(loss_func(predicted, labels))
        accuracy_per_batch.append(compute_accuracy(predicted, labels))

        # Calculate class-wise accuracy
        _, pred = torch.max(predicted, 1)
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(
            correct_tensor.cpu().numpy())

        for i in range(len(classes)):
            class_label = classes[i]
            class_correct[i] += np.sum(correct[labels.data.cpu().numpy() == i])
            class_total[i] += np.sum(labels.data.cpu().numpy() == i)

    test_loss = torch.stack(loss_per_batch).mean().item()
    # test_accuracy = torch.stack(accuracy_per_batch).mean().item()
    test_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

    print(
        f'Test Loss: {test_loss:.4f}, Test Accuracy (Overall): {test_accuracy:.2f}% ({np.sum(class_correct)}/{np.sum(class_total)})')  # {test_accuracy * 100:.2f}%

    for i in range(len(classes)):
        if class_total[i] > 0:
            print(f'Test Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}% '
                  f'({np.sum(class_correct[i])}/{np.sum(class_total[i])})')
        else:
            print(f'Test Accuracy of {classes[i]}: N/A (no training examples)')

    return test_loss, test_accuracy


def test_final_model(model, checkpoint, test_loader):
    """
    Test the last epoch saved model
    """
    print('Loading last epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_func = checkpoint['loss_func']

    with torch.set_grad_enabled(False):
        evaluate(model, test_loader, loss_func)


def test_best_model(model, checkpoint, test_loader):
    """
    Test the best epoch saved model
    """
    print('Loading best epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_func = checkpoint['loss_func']

    with torch.set_grad_enabled(False):
        evaluate(model, test_loader, loss_func)


if __name__ == '__main__':
    model = get_model(model_name, batch_size, num_classes)

    # load and test the final model checkpoint
    final_model_cp = torch.load(f'outputs/{output_folder}/final_model_{model_name}{optim_code}_{cp_datetime}.pth')
    final_model_epoch = final_model_cp['epoch']
    print(f"Final model was saved at {final_model_epoch} epochs\n")
    test_final_model(model, final_model_cp, test_dl)

    print('-' * 50)

    # load and test the best model checkpoint
    best_model_cp = torch.load(f'outputs/{output_folder}/best_model_{model_name}{optim_code}_{cp_datetime}.pth')
    best_model_epoch = best_model_cp['epoch']
    print(f"Best model was saved at {best_model_epoch} epochs\n")
    test_best_model(model, best_model_cp, test_dl)
