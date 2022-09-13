import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import Sampler
from torchvision import transforms as T
#from torchvision.datasets.celeba import CelebA
from data.cxr import ChestDataset
from functools import reduce
import math
import warnings
import torchxrayvision as xrv

def get_confusion_matrix(num_classes, targets, biases):
    confusion_matrix_org = torch.zeros(num_classes, num_classes)
    confusion_matrix_org_by = torch.zeros(num_classes, num_classes)
    for t, p in zip(targets, biases):
        confusion_matrix_org[p.long(), t.long()] += 1
        confusion_matrix_org_by[t.long(), p.long()] += 1

    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    confusion_matrix_by = confusion_matrix_org_by / confusion_matrix_org_by.sum(1).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix, confusion_matrix_by

class ConfounderDataset(Dataset):
    def __init__(self, dataset, group_array):
        self.dataset = dataset
        self.group_array = torch.tensor(group_array)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], self.group_array[idx])

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

    
transforms = {
    "Gender_pneumothorax_case1": {
        "train": T.Compose(
            [
                #T.Resize((512, 512)),
                T.RandomResizedCrop((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        ),
    },
    "Gender_pneumothorax_case2": {
        "train": T.Compose(
            [
                #T.Resize((512, 512)),
                T.RandomResizedCrop((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        ),
    },
    "MIMIC_CXR_case1": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    },
    "MIMIC_CXR_case2": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    },
    "MIMIC_CXR_case3": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    },
}

def get_dataset(dataset_tag, data_dir, dataset_split, transform_split):
    dataset_category = dataset_tag.split("-")[0]
    transform = transforms[dataset_category][transform_split]
    dataset_split = "valid" if (dataset_split == "eval") else dataset_split
    if dataset_tag == "Gender_pneumothorax_case1":
        root = ''
        csv_dir = './dataset/csv/GbP-case1'
        dataset = ChestDataset(
            csv_dir=csv_dir,
            root=root,
            split=dataset_split,
            dataset_tag=dataset_tag,
            transform=transform,
        )
    elif dataset_tag == "Gender_pneumothorax_case2":
        root = ''
        csv_dir = './dataset/csv/GbP-case2'
        dataset = ChestDataset(
            csv_dir=csv_dir,
            root=root,
            split=dataset_split,
            dataset_tag=dataset_tag,
            transform=transform,
        )
    elif dataset_tag == "MIMIC_CXR_case1":
        root = ''
        csv_dir = './dataset/csv/SbP-bias90'
        dataset = ChestDataset(
            csv_dir=csv_dir,
            root=root,
            split=dataset_split,
            dataset_tag=dataset_tag,
            transform=transform,
        )
    elif dataset_tag == "MIMIC_CXR_case2":
        root = ''
        csv_dir = './dataset/csv/SbP-bias95'
        dataset = ChestDataset(
            csv_dir=csv_dir,
            root=root,
            split=dataset_split,
            dataset_tag=dataset_tag,
            transform=transform,
        )
    elif dataset_tag == "MIMIC_CXR_case3":
        root = ''
        csv_dir = './dataset/csv/SbP-bias99'
        dataset = ChestDataset(
            csv_dir=csv_dir,
            root=root,
            split=dataset_split,
            dataset_tag=dataset_tag,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return dataset

