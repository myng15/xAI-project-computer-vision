import torch
import torch.nn as nn

import datetime

from datasets import prepare_data_loaders
from utils import seed_all, save_final_model, BestModelSaver, get_available_device
from visualization import save_plots


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


def train(num_classes, model, model_name,
          batch_size, num_workers, epochs,
          lr_scheduler,
          loss_func, optimizer, grad_clip,
          train_transform, test_transform,
          patience=7, gamma=0.5, #or gamma=0.1,
          optim_code=''
          ):
    seed_all(42)

    train_dl, valid_dl = prepare_data_loaders(num_classes, batch_size, num_workers, train_transform, test_transform)

    plot_lrs = False

    # start training
    results = []
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # initialize SaveBestModel class
    save_best_model = BestModelSaver()

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

            # OPTIMIZATION: Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()

            # OPTIMIZATION: Keep track of learning rate
            lrs.append(optimizer.param_groups[0]['lr'])

            train_accuracies.append(compute_accuracy(predicted, labels))

            # logging
            if not batch_idx % 120:
                print (f'Epoch: {epoch+1:03d}/{epochs:03d} | '
                      f'Batch {batch_idx+1:03d}/{len(train_dl):03d} | '
                      f'Loss: {loss:.4f}')

        if lr_scheduler:
            lr_scheduler.step()
            plot_lrs = True

        # no need to build the computation graph for backprop when computing accuracy
        with torch.set_grad_enabled(False):
            epoch_train_acc = torch.stack(train_accuracies).mean().item()
            epoch_train_loss = torch.stack(train_losses).mean().item()
            epoch_val_loss, epoch_val_acc = validate(model, valid_dl, loss_func)

        results.append({'avg_valid_loss': epoch_val_loss,
                        'avg_val_acc': epoch_val_acc,
                        'avg_train_loss': epoch_train_loss,
                        'avg_train_acc': epoch_train_acc,
                        'lrs': lrs})

        # print training/validation statistics
        print(f'Epoch: {epoch+1:03d}/{epochs:03d}   Train Loss: {epoch_train_loss:.4f} | Train Acc.: {epoch_train_acc*100:.2f}%'
              f' | Validation Loss: {epoch_val_loss:.4f} | Validation Acc.: {epoch_val_acc*100:.2f}%')

        # save model if validation loss has decreased (validation loss ever increasing indicates possible overfitting.)
        save_best_model(
            num_classes, current_datetime, model, model_name, optimizer, loss_func, epoch_val_loss, epoch, optim_code
        )
        print('-' * 50)

    # save the trained model weights for a final time
    save_final_model(num_classes, current_datetime, model, model_name, optimizer, loss_func, epochs, results, optim_code)
    # save the loss and accuracy plots
    save_plots(num_classes, current_datetime, model_name, epochs, results, optim_code, plot_lrs)
    print('TRAINING FINISHED')



