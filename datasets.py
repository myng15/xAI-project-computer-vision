import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import argparse

from utils import seed_all

seed_all(42)

# construct the argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument('-b', '--batch_size', type=int, default=64,
#     help='number of samples per batch to load')
# parser.add_argument('-v', '--valid_size', type=float, default=0.2,
#     help='percentage of training set to use as validation')
# parser.add_argument('-w', '--num_workers', type=int, default=0,
#     help='number of subprocesses to use for data loading')
# args = vars(parser.parse_args())
#
# batch_size = args['batch_size']
# valid_size = args['valid_size']
# num_workers = args['num_workers']

# transform data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# # transforms and augmentations for training
# train_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# # transforms for validation and testing
# valid_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

train_data_cifar10 = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_data_cifar10 = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

train_data_cifar100 = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
test_data_cifar100 = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)


# function to create the datasets
def get_datasets(trained_on_cifar10):
    if trained_on_cifar10:
        train_data = train_data_cifar10
        test_data = test_data_cifar10
    else:
        train_data = train_data_cifar100
        test_data = test_data_cifar100

    return train_data, test_data


def create_data_loaders(train_data, test_data, batch_size, num_workers):
    """
    Function to build the data loaders.
    (Dataloader provides an iterable over the specified dataset by combining a dataset with a sampler)

    Parameters:
    :param train_data: The training dataset.
    :param valid_data: The validation dataset.
    :param test_data: The test dataset.
    """

    valid_size = 0.2

    # obtain training indices to be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers)  # shuffle=True? - #sampler option is mutually exclusive with shuffle (shuffle=True if remove sampler)
    valid_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers)  # shuffle=True? - #sampler option is mutually exclusive with shuffle (shuffle=True if remove sampler)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)  # shuffle=False?

    print(f"Number of training images: {len(train_loader.dataset)}")
    print(f"Number of validation images: {len(valid_loader.dataset)}")
    print(f"Number of test images: {len(test_loader.dataset)}")

    return train_loader, valid_loader, test_loader