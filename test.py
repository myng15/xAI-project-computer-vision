import torch

import numpy as np
import argparse

from datasets import get_datasets, create_data_loaders
from utils import get_available_device, DeviceDataLoader, seed_all
from models import get_model, get_loss_func

parser = argparse.ArgumentParser()
parser.add_argument('-cifar10', '--trained_on_cifar10', type=bool,
    help='whether or not the model to be tested is trained on CIFAR-10')
parser.add_argument('-m', '--model_name', type=str,
    help='name of model to be tested')
parser.add_argument('-l', '--loss_func_name', type=str,
    help='name of loss function for model to be tested')
parser.add_argument('-b', '--batch_size', type=int,
    help='number of samples per batch to load')
parser.add_argument('-w', '--num_workers', type=int,
    help='number of subprocesses to use for data loading')
args = vars(parser.parse_args())

trained_on_cifar10 = args['trained_on_cifar10']
model_name = args['model_name']
loss_func_name = args['loss_func_name']
batch_size = args['batch_size']
num_workers = args['num_workers']

if trained_on_cifar10:
    num_classes = 10
else:
    num_classes = 100

seed_all(42)

# get device
device = get_available_device()
print(f"Device: {device}\n")

# get the training, validation and test_datasets
train_data, test_data = get_datasets(trained_on_cifar10)
# get the test data loader
_, _, test_loader = create_data_loaders(
    train_data, test_data, batch_size, num_workers
)

# Move all the tensors to GPU if available
test_dl = DeviceDataLoader(test_loader, device)
classes = train_data.classes

output_folder = 'cifar10' if trained_on_cifar10 else 'cifar100'

# load the best model checkpoint
best_model_cp = torch.load(f'outputs/{output_folder}/best_model_{model_name}_{output_folder}.pth')
best_model_epoch = best_model_cp['epoch']
print(f"Best model was saved at {best_model_epoch} epochs\n")


# load the final model checkpoint
final_model_cp = torch.load(f'outputs/{output_folder}/final_model_{model_name}_{output_folder}.pth')
final_model_epoch = final_model_cp['epoch']
print(f"Last model was saved at {final_model_epoch} epochs\n")


def compute_accuracy(predicted, labels):
    predictions, predicted_labels = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predicted_labels == labels).item()/len(predicted))


# def prepare_data_loaders(trained_on_cifar10, batch_size, num_workers):
#     # get the training, validation and test_datasets
#     train_data, test_data = get_datasets(trained_on_cifar10)
#     # get the test data loader
#     _, _, test_loader = create_data_loaders(
#         train_data, test_data, batch_size, num_workers
#     )
#
#     # Move all the tensors to GPU if available
#     test_dl = DeviceDataLoader(test_loader, device)
#     classes = train_data.classes
#
#     return test_dl, classes


def evaluate(model, dl, loss_func):
    model.eval()

    #_, classes = prepare_data_loaders(trained_on_cifar10)
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
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())

        for i in range(len(classes)):
            class_label = classes[i]
            class_correct[i] += np.sum(correct[labels.data.cpu().numpy() == i]) #np.sum(correct[labels.data == i]) if not torch.cuda.is_available() else
            class_total[i] += np.sum(labels.data.cpu().numpy() == i) #np.sum(labels.data == i)

    test_loss = torch.stack(loss_per_batch).mean().item()
    #test_accuracy = torch.stack(accuracy_per_batch).mean().item()
    test_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

    print(f'Test Loss: {test_loss:.6f}, Test Accuracy (Overall): {test_accuracy:.2f}% ({np.sum(class_correct)}/{np.sum(class_total)})') #{test_accuracy * 100:.2f}%

    for i in range(len(classes)):
        if class_total[i] > 0:
            print(f'Test Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}% '
                  f'({np.sum(class_correct[i])}/{np.sum(class_total[i])})')
        else:
            print(f'Test Accuracy of {classes[i]}: N/A (no training examples)')

    return test_loss, test_accuracy


# test the best epoch saved model
def test_best_model(model, checkpoint, test_loader):
    print('Loading best epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_func = get_loss_func(loss_func_name)

    with torch.set_grad_enabled(False):
        evaluate(model, test_loader, loss_func)
    #print(f"Best epoch saved model accuracy: {test_acc:.3f}")


# test the last epoch saved model
def test_final_model(model, checkpoint, test_loader):
    print('Loading last epoch saved model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_func = get_loss_func(loss_func_name)

    with torch.set_grad_enabled(False):
        evaluate(model, test_loader, loss_func)
    #print(f"Last epoch saved model accuracy: {test_acc:.3f}")


if __name__ == '__main__':
    model = get_model(model_name, batch_size, num_classes)
    test_final_model(model, final_model_cp, test_dl)
    test_best_model(model, best_model_cp, test_dl)
