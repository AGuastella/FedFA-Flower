"""MNIST dataset utilities for federated learning."""
from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100

from omegaconf import DictConfig

# from FedFA.dataset_preparation import _partition_data


def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Download and transform CIFAR-100 (train and test)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjusted normalization values for CIFAR-10/CIFAR-100
    ])

    if config.dataset == 'cifar100':
        trainset = CIFAR100(root="./dataset", train=True, download=True, transform=transform)
        testset = CIFAR100(root="./dataset", train=False, download=True, transform=transform)
    else:
        trainset = CIFAR10(root="./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10(root="./dataset", train=False, download=True, transform=transform)

    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths)

    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = int(len(ds) * val_ratio)
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths)
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size, shuffle=True))
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders, testloader
