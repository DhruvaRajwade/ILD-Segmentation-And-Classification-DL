import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.transforms import Resize
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import torchvision.transforms as transforms


class MyDataset(data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

def get_data_loaders(images_path, labels_path, batch_size, train_split, val_split, img_dim):

    """

    Args:

    images_path (str): The file path to the .npy file containing the images data

    labels_path (str): The file path to the .npy file containing the labels data

    batch_size (int): The batch size for the data loader

    train_split (float): The percentage of the data to use for the training set, expressed as a float between 0 and 1

    val_split (float): The percentage of the data to use for the validation set, expressed as a float between 0 and 1

    img_dim (int): The target dimension of the image after resizing


    """


    # Load the images and labels
    images = np.load(images_path)
    labels = np.load(labels_path)

    # Normalize the pixel values to the range [0, 1]
    if np.max(images) > 1:
        images = images / 255.0
    images = images.astype(np.float32)

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_dim),
        # transforms.Normalize(( mean,), (std,))
    ])

    # Split the data into training, validation, and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        images, labels, test_size=1 - train_split, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(
        test_data, test_labels, test_size=val_split / (1 - train_split), random_state=42)

    # Define the dataset and data loader for the training set
    train_dataset = MyDataset(train_data, train_labels, transform=transform)
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # Define the dataset and data loader for the validation set
    val_dataset = MyDataset(val_data, val_labels, transform=transform)
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # Define the dataset and data loader for the testing set
    test_dataset = MyDataset(test_data, test_labels, transform=transform)
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
