import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import sys


"""This script implements the functions for data augmentation
and preprocessing.
"""

def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    
    if len(image.shape) != 1 or image.shape[0] != 3072:
        raise ValueError("Invalid image shape. Expected [3072], got {}".format(image.shape))


    image = image.reshape((3, 32, 32))
    image = np.transpose(image, [1, 2, 0])



    plt.imshow(image)
    plt.savefig(save_name)
    plt.close()  

    return image

    ### YOUR CODE HERE

# Other functions
### YOUR CODE HERE

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = img.reshape((3,32,32))
        img = np.transpose(img, [1,2,0])
        img = Image.fromarray(np.uint8(img))
        if self.transform:
            img = self.transform(img)
        return img, target

def transform_data(x_train, y_train, x_test, y_test):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CustomDataset(x_train, y_train, transform=transform_train)
    # print(x_train.shape)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = CustomDataset(x_test, y_test, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

def transform_val_data(x_val, y_val):
    """Transform the validation data and create a DataLoader object.

    Args:
        x_val: An array of shape [N, 3072].
        y_val: An array of shape [N,].

    Returns:
        valloader: DataLoader for validation data.
    """
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    valset = CustomDataset(x_val, y_val, transform=transform_val)
    valloader = DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

    return valloader

class CustomTestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = img.reshape((3,32,32))
        img = np.transpose(img, [1,2,0])
        img = Image.fromarray(np.uint8(img))
        if self.transform:
            img = self.transform(img)
        return img

def transform_test_data(x_test):
    """Transform the test data and create a DataLoader object.

    Args:
        x_test: An array of shape [N, 3072].

    Returns:
        testloader: DataLoader for test data.
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = CustomTestDataset(x_test, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return testloader


### END CODE HERE