import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from typing import Optional, Tuple
from omegaconf import DictConfig

import random

class CustomDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {'image': self.x[idx], 'label': self.y[idx]}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

def generate_dirichlet_partitions(dataset, alpha, num_of_clients):
    folder = dataset + "_dirichlet"
    folder_path = os.path.join(folder, str(round(alpha, 2)))

    # Check if the partitioned dataset already exists
    if os.path.exists(os.path.join(folder_path, "train", "0_x.pt")):
                # Load and return the existing dataset
        print('Trovato dataset in', folder_path)
        datasets = []
        for c in range(num_of_clients):
            x = torch.load(os.path.join(folder_path, "train", f"{c}_x.pt"))
            y = torch.load(os.path.join(folder_path, "train", f"{c}_y.pt"))
            datasets.append((x, y))
        path = os.path.join(folder_path, "distribution.npy")
        smpls = np.load(path)
        return datasets, path

    print('Creo dataset in', folder_path)
    # If the dataset doesn't exist, create it
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, "train"), exist_ok=True)

    num_of_classes = 10 if dataset == "cifar10" else 100
    smpls = generate_dirichlet_samples(num_of_classes, alpha, num_of_clients, 
                                       5000 if dataset == "cifar10" else 500)
    datasets = []
    for c, per_client_sample in enumerate(smpls):
        x, y = load_data(dataset, per_client_sample)
        torch.save(x, os.path.join(folder_path, "train", f"{c}_x.pt"))
        torch.save(y, os.path.join(folder_path, "train", f"{c}_y.pt"))
        datasets.append((x, y))
    path = os.path.join(folder_path, "distribution.npy")
    np.save(path, smpls.numpy())

    return datasets, path



def generate_dirichlet_samples(num_of_classes, alpha, num_of_clients, num_of_examples_per_label):
    for _ in range(0, 10):
        alpha_tensor = torch.full((num_of_clients,), alpha)
        dist = torch.distributions.dirichlet.Dirichlet(alpha_tensor)
        samples = dist.sample([num_of_classes])
        int_samples = torch.round(samples * num_of_examples_per_label).int().t()
        correctly_generated = int_samples.sum(dim=1).min().item()
        if correctly_generated != 0:
            break
        print("Generated some clients without any examples. Retrying..")
    return int_samples


'''
def load_data(dataset, per_client_sample):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform) if dataset == "cifar10" \
        else CIFAR100("./dataset", train=True, download=True, transform=transform)

    indexes_of_labels = [torch.where(torch.tensor(trainset.targets) == label)[0] for label in range(len(trainset.classes))]

    x_data = []
    y_data = []

    for label, num_of_examples_per_label in enumerate(per_client_sample):
        available_indices = indexes_of_labels[label].numpy()
        if len(available_indices) < num_of_examples_per_label:
            raise ValueError("Not enough examples for label {}".format(label))
        
        extracted_indices = random.sample(list(available_indices), num_of_examples_per_label)
        x_data.append(torch.tensor(trainset.data[extracted_indices]))  # Convert to torch tensor
        y_data.append(torch.tensor(trainset.targets)[extracted_indices])  # Convert to torch tensor

    return torch.cat(x_data, dim=0), torch.cat(y_data, dim=0)
'''

def load_data(dataset, per_client_sample):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform) if dataset == "cifar10" \
        else CIFAR100("./dataset", train=True, download=True, transform=transform)

    indexes_of_labels = [torch.where(torch.tensor(trainset.targets) == label)[0] for label in range(len(trainset.classes))]

    x_data = []
    y_data = []

    for label, num_of_examples_per_label in enumerate(per_client_sample):
        available_indices = indexes_of_labels[label].numpy()
        if len(available_indices) < num_of_examples_per_label:
            raise ValueError("Not enough examples for label {}".format(label))
        
        extracted_indices = random.sample(list(available_indices), num_of_examples_per_label)
        x_data.append(torch.tensor(trainset.data[extracted_indices]))  # Convert to torch tensor
        y_data.append(torch.tensor(trainset.targets)[extracted_indices])  # Convert to torch tensor

    return torch.cat(x_data, dim=0), torch.cat(y_data, dim=0)






def load_generated_datasets(dataset_path, num_clients, val_ratio=0.1, batch_size=32):
    smpls_loaded = np.load(dataset_path)

    trainloaders = []
    valloaders = []
    for c in range(num_clients):
        x = torch.load(os.path.join(dataset_path, "train", f"{c}_x.pt"))
        y = torch.load(os.path.join(dataset_path, "train", f"{c}_y.pt"))

        len_val = int(len(x) * val_ratio)
        len_train = len(x) - len_val
        ds_train = CustomDataset(x[:len_train], y[:len_train])
        ds_val = CustomDataset(x[len_train:], y[len_train:])
        
        trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)
        
        trainloaders.append(trainloader)
        valloaders.append(valloader)

    return trainloaders, valloaders


def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    trainloaders = []
    valloaders = []
    testloaders = []


    datasets, dataset_path = generate_dirichlet_partitions(config.dataset, config.alpha_dirichlet, num_clients)


    for c in range(num_clients):
        x = datasets[c][0]
        y = datasets[c][1]

        len_val = int(len(x) * val_ratio)
        len_train = len(x) - len_val
        ds_train = CustomDataset(x[:len_train], y[:len_train])
        ds_val = CustomDataset(x[len_train:], y[len_train:])

        trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

        trainloaders.append(trainloader)
        valloaders.append(valloader)

    # Load or generate test dataset
    #test_x, test_y = load_data(dataset, 10)  # Replace 'test_samples' with the number of test samples per class
    #ds_test = CustomDataset(test_x, test_y)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds_test = CIFAR10("./dataset", train=False, download=True, transform=transform) if config.dataset == "cifar10" \
        else CIFAR100("./dataset", train=False, download=True, transform=transform)
    testloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    testloaders.append(testloader)

    return trainloaders, valloaders, testloaders
