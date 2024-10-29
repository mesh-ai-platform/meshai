# meshai/data_handler.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import PyPDF2

class BaseDataHandler:
    """
    Base class for data handlers.
    """
    def __init__(self):
        pass

    def load_data(self):
        """
        Loads data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def preprocess_data(self):
        """
        Preprocesses data.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initializes the text dataset.

        Args:
            texts (list): List of text samples.
            labels (list): List of labels corresponding to the texts.
            tokenizer: Tokenizer to preprocess the text data.
            max_length (int): Maximum length of tokenized sequences.
        """
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves item at index idx.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.labels)

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Initializes the image dataset.

        Args:
            image_paths (list): List of paths to image files.
            labels (list): List of labels corresponding to the images.
            transform: Transformations to apply to the images.
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

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing the image tensor and label.
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return image, label

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.labels)

class NumericalDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initializes the numerical dataset.

        Args:
            features (array-like): Feature matrix.
            labels (array-like): Target labels.
        """
        self.features = torch.tensor(features).float()
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, idx):
        """
        Retrieves item at index idx.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing features and label.
        """
        return self.features[idx], self.labels[idx]

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.labels)

def load_text_data_from_csv(csv_file, text_column='text', label_column='label'):
    """
    Loads text data and labels from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        text_column (str): Name of the column containing text data.
        label_column (str): Name of the column containing labels.

    Returns:
        tuple: Tuple containing a list of texts and a list of labels.
    """
    df = pd.read_csv(csv_file)
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist()
    return texts, labels

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.

    Args:
        pdf_file (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ''
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def load_image_paths_and_labels(image_dir, allowed_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """
    Loads image paths and corresponding labels from a directory.

    Args:
        image_dir (str): Path to the directory containing images organized in subdirectories by class.
        allowed_extensions (tuple): Allowed image file extensions.

    Returns:
        tuple: Tuple containing a list of image paths and a list of labels.
    """
    import os
    image_paths = []
    labels = []
    classes = sorted(os.listdir(image_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_dir = os.path.join(image_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for filename in os.listdir(cls_dir):
            if filename.lower().endswith(allowed_extensions):
                image_paths.append(os.path.join(cls_dir, filename))
                labels.append(class_to_idx[cls_name])
    return image_paths, labels

def load_numerical_data_from_csv(csv_file, feature_columns, label_column):
    """
    Loads numerical data and labels from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        feature_columns (list): List of column names to be used as features.
        label_column (str): Name of the column containing labels.

    Returns:
        tuple: Tuple containing feature matrix X and labels y.
    """
    df = pd.read_csv(csv_file)
    X = df[feature_columns].values
    y = df[label_column].values
    return X, y
