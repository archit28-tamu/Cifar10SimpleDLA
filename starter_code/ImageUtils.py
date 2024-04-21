import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import sys


"""This script implements the functions for data augmentation
and preprocessing.
"""

# def parse_record(record, training):
#     """Parse a record to an image and perform data preprocessing.

#     Args:
#         record: An array of shape [3072,]. One row of the x_* matrix.
#         training: A boolean. Determine whether it is in training mode.

#     Returns:
#         image: An array of shape [3, 32, 32].
#     """
#     ### YOUR CODE HERE

#     img_reshape = record.reshape((3, 32, 32))

#     image = np.transpose(img_reshape, [1, 2, 0])

#     image = np.transpose(image, [2, 0, 1])

#     ### END CODE HERE

#     image = preprocess_image(image, training) # If any.

#     return image


# def preprocess_image(image, training):
#     """Preprocess a single image of shape [height, width, depth].

#     Args:
#         image: An array of shape [3, 32, 32].
#         training: A boolean. Determine whether it is in training mode.

#     Returns:
#         image: An array of shape [3, 32, 32]. The processed image.
#     """
#     ### YOUR CODE HERE

#     if training:
        
#         pad_width = ((4, 4), (4, 4), (0, 0))  
#         img = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)

#         crop_size = (32, 32) 
#         max_crop_y, max_crop_x = img.shape[1] - crop_size[0] + 1, img.shape[2] - crop_size[1] + 1
#         start_y = np.random.randint(0, max_crop_y)
#         start_x = np.random.randint(0, max_crop_x)
#         image = img[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1], :]

#         flip_or_not = np.random.randint(0, 2, dtype='bool')
#         if flip_or_not:
#             image = np.flip(image, axis=1)  

#     mean = np.mean(image, axis=(1, 2)) 
#     std = np.std(image, axis=(1, 2))   
#     image = (image - mean[:, None, None]) / std[:, None, None]  

#     ### END CODE HERE

#     return image


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

  
    image = image.reshape(32, 32, 3)

    img = np.clip(image, 0, 255).astype(np.uint8)

    plt.imshow(img)
    plt.savefig(save_name)
    plt.close()  

    return image

    ### YOUR CODE HERE
    
    # plt.imshow(image)
    # plt.savefig(save_name)
    # return image

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