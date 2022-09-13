import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import os

class ChestDataset(Dataset):
    def __init__(self, root, split, csv_dir, dataset_tag, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(ChestDataset, self).__init__()
        if split == 'train':
            csv_file = 'train.csv'
            print('Training with: ', csv_file)
        elif (split == 'eval') or (split == 'valid'):
            csv_file = 'valid.csv'
            print('Validation: ', csv_file)
        else:
            csv_file = 'test.csv'
            print('Testing: ', csv_file)

        if split == 'train':
            if 'Gender_pneumothorax' in dataset_tag:
                df = pd.read_csv(os.path.join(csv_dir, csv_file))
                pneumothorax = df['pneumothorax']
                gender = df['gender']
                attr = np.stack([np.array(pneumothorax), np.array(gender)], axis=1)
                self.filename = df['path']
            elif 'MIMIC' in dataset_tag:
                MIMIC_file = 'train_MIMIC.csv'
                NIH_file = 'train_NIH.csv'
                MIMIC_df = pd.read_csv(os.path.join(csv_dir, MIMIC_file))
                NIH_df = pd.read_csv(os.path.join(csv_dir, NIH_file))
                MIMIC_pneumonia = MIMIC_df['pneumonia']
                MIMIC_source = MIMIC_df['MIMIC']
                NIH_pneumonia = NIH_df['pneumonia']
                NIH_source = NIH_df['MIMIC']
                pneumonia = pd.concat([MIMIC_pneumonia, NIH_pneumonia])
                source = pd.concat([MIMIC_source, NIH_source])
                attr = np.stack([np.array(pneumonia), np.array(source)], axis=1)

                MIMIC_path = MIMIC_df['path'].tolist()
                NIH_path = NIH_df['path'].tolist()
                # print(type(NIH_path + MIMIC_path))
                # self.filename = pd.concat([MIMIC_path, NIH_path])
                self.filename = NIH_path + MIMIC_path
            else:
                raise NotImplementedError
        elif (split == 'eval') or (split == 'valid') or (split == 'test'):
            if 'Gender_pneumothorax' in dataset_tag:
                df = pd.read_csv(os.path.join(csv_dir, csv_file))
                pneumothorax = df['pneumothorax']
                gender = df['gender']
                attr = np.stack([np.array(pneumothorax), np.array(gender)], axis=1)
                self.filename = df['path']
            elif 'MIMIC' in dataset_tag:
                df = pd.read_csv(os.path.join(csv_dir, csv_file))
                pneumonia = df['pneumonia']
                source = df['MIMIC']
                attr = np.stack([np.array(pneumonia), np.array(source)], axis=1)
                self.filename = df['path']
            else:
                raise NotImplementedError

        self.attr = torch.LongTensor(attr)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        path = self.filename[index]


        path = path.replace('./dataset/GbP/', '/research/dept8/msc/dyxu21/NIH/images/')
        # print(path)

        image = Image.open(path).convert('L')
        attr = self.attr[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, attr

    def __len__(self):
        return len(self.filename)

