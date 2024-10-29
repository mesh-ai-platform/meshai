# meshai/data_handler.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initializes the text dataset.
        """
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves item at index idx.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.labels)

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Initializes the image dataset.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        """
        Retrieves item at index idx.
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return image, label

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.labels)

class NumericalDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initializes the numerical dataset.
        """
        self.features = torch.tensor(features).float()
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, idx):
        """
        Retrieves item at index idx.
        """
        return self.features[idx], self.labels[idx]

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.labels)
