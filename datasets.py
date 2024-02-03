from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

from utils import seed_all, get_available_device, DeviceDataLoader

# set seed
seed_all(42)


def get_cifar_datasets(num_classes, train_transform, test_transform):
    if num_classes == 10:
        train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
    elif num_classes == 100:
        train_data = datasets.CIFAR100(root='data', train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR100(root='data', train=False, download=True, transform=test_transform)
    else:
        train_data, test_data = None, None

    return train_data, test_data


# Function to calculate mean and std for normalization
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(loader))[0]  # Get all data from the first (and only) batch

    mean = [round(m.item(), 4) for m in data.mean([0, 2, 3])]
    std = [round(s.item(), 4) for s in data.std([0, 2, 3])]

    return mean, std


def get_normalized_cifar_datasets(num_classes):
    # Load CIFAR dataset without normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data, test_data = get_cifar_datasets(num_classes, train_transform=transform, test_transform=transform)

    # Calculate mean and std for normalization
    mean_train, std_train = calculate_mean_std(train_data)
    mean_test, std_test = calculate_mean_std(test_data)

    # Load CIFAR dataset with normalization
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_train, std_train)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_test, std_test)
    ])

    train_data, test_data = get_cifar_datasets(num_classes, train_transform=train_transform, test_transform=test_transform)
    return train_data, test_data


def create_data_loaders(train_data, test_data, batch_size, num_workers):
    """
    Function to build the data loaders.
    (Dataloader provides an iterable over the specified dataset by combining a dataset with a sampler)

    Parameters:
    :param train_data: The training dataset.
    :param test_data: The test dataset.
    """

    valid_size = 0.2

    # obtain training indices to be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    seed_all(42)
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
                            num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def prepare_data_loaders(num_classes, batch_size, num_workers, train_transform, test_transform):
    # get the training, validation and test_datasets
    train_data, test_data = get_cifar_datasets(num_classes, train_transform, test_transform)
    # get the training and validation data loaders
    train_loader, valid_loader, _ = create_data_loaders(
        train_data, test_data, batch_size, num_workers
    )

    # Move all the tensors to GPU if available
    device = get_available_device()
    train_dl = DeviceDataLoader(train_loader, device)
    valid_dl = DeviceDataLoader(valid_loader, device)

    return train_dl, valid_dl
