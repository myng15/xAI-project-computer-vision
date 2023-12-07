import torch

import datetime

from datasets import get_datasets, create_data_loaders
from utils import seed_all, save_final_model, save_plots, BestModelSaver, get_available_device, DeviceDataLoader

# get device
device = get_available_device()
print(f"Device: {device}\n")

# construct the argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch_size', type=int, default=64,
#     help='number of samples per batch to load')
# parser.add_argument('-e', '--epochs', type=int, default=10,
#     help='number of epochs to train the model for')
# parser.add_argument('-lr', type=int, default=0.001,
#     help='learning rate')
# args = vars(parser.parse_args())
#
# batch_size = args['batch_size']
# epochs = args['epochs']
# lr = args['lr']


# LATER: Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# initialize SaveBestModel class
save_best_model = BestModelSaver()


def compute_accuracy(predicted, labels):
    predictions, predicted_labels = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predicted_labels == labels).item()/len(predicted))


def validate(model, dl, loss_func):
    model.eval()
    loss_per_batch, accuracy_per_batch = [], []
    for images, labels in dl:
        # start loop
        with torch.no_grad():
            predicted = model(images)
        loss_per_batch.append(loss_func(predicted, labels))
        accuracy_per_batch.append(compute_accuracy(predicted, labels))
    epoch_val_loss = torch.stack(loss_per_batch).mean().item()
    epoch_val_acc = torch.stack(accuracy_per_batch).mean().item()
    return epoch_val_loss, epoch_val_acc


def prepare_data_loaders(num_classes, batch_size, num_workers):
    # get the training, validation and test_datasets
    train_data, test_data = get_datasets(num_classes)
    # get the training and validation data loaders
    train_loader, valid_loader, _ = create_data_loaders(
        train_data, test_data, batch_size, num_workers
    )

    # Move all the tensors to GPU if available
    train_dl = DeviceDataLoader(train_loader, device)
    valid_dl = DeviceDataLoader(valid_loader, device)

    return train_dl, valid_dl


def train(num_classes, model, model_name, batch_size, num_workers, epochs, max_lr, loss_func, optimizer): # LATER: add parameter lr_scheduler
    # seed needed due to batch shuffling?
    seed_all(42)

    train_dl, valid_dl = prepare_data_loaders(num_classes, batch_size, num_workers)

    # start training
    results = []
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        lrs = []

        for batch_idx, (images, labels) in enumerate(train_dl):
            predicted = model(images)
            loss = loss_func(predicted, labels)
            train_losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep track of learning rate
            lrs.append(optimizer.param_groups[0]['lr'])
            train_accuracies.append(compute_accuracy(predicted, labels))

            # logging
            if not batch_idx % 120:
                print (f'Epoch: {epoch+1:03d}/{epochs:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_dl):03d} | '
                      f'Loss: {loss:.4f}')

        # LATER:
        #lr_scheduler.step()

        # no need to build the computation graph for backprop when computing accuracy
        with torch.set_grad_enabled(False):
            epoch_train_acc = torch.stack(train_accuracies).mean().item()
            epoch_train_loss = torch.stack(train_losses).mean().item()
            epoch_val_loss, epoch_val_acc = validate(model, valid_dl, loss_func)

        results.append({'avg_valid_loss': epoch_val_loss,
                        'avg_val_acc': epoch_val_acc,
                        'avg_train_loss': epoch_train_loss,
                        'avg_train_acc': epoch_train_acc,
                        'lrs' : lrs})

        # print training/validation statistics
        print(f'Epoch: {epoch+1:03d}/{epochs:03d}   Train Loss: {epoch_train_loss:.2f} | Train Acc.: {epoch_train_acc:.2f}%'
              f' | Validation Loss: {epoch_val_loss:.2f} | Validation Acc.: {epoch_val_acc:.2f}%')

        # save model if validation loss has decreased (validation loss ever increasing indicates possible overfitting.)
        save_best_model(
            num_classes, current_datetime, model, model_name, optimizer, loss_func, epoch_val_loss, epoch
        )
        print('-' * 50)

    # save the trained model weights for a final time
    save_final_model(num_classes, current_datetime, model, model_name, optimizer, loss_func, epochs)
    # save the loss and accuracy plots
    save_plots(num_classes, current_datetime, model_name, epochs, results)
    print('TRAINING FINISHED')

    return results


