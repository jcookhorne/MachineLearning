"""
* Torchvision - base domain library for PyTorch computer vision
* TorchVision.datasets - get datasets and data loading functions for computer vision here
* Torchvision.models - get pretrained computer vision models that you can leverage for your own problems
* torchvision.transforms - functions for manipulating your vision data(images) to be suitable for use with a ML model
* torch.utils.data.dataset - base dataset class for pytorch ( helps make custom data)\
* torch utils data.dataloader - creates a python iterable over a dataset
"""
# import pytorch
import torch
from torch import nn

# import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# check versions
# print(torch.__version__)
# print(torchvision.__version__)


# Setup training data
train_data = datasets.FashionMNIST(
    root="data",  # where to download data to?
    train=True,  # do we want the training dataset
    download=True,  # do we want to download yes/no
    transform=torchvision.transforms.ToTensor(),
    target_transform=None  # how do we want to transform the labels/targets?
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

# print(len(train_data), len(test_data))

# See the first training example
# image, label = train_data[0]
# print(image, label)

class_names = train_data.classes
# print(class_names)

class_to_idx = train_data.class_to_idx
# print(class_to_idx)

# print(train_data.targets)
# check the shape of our image
# print(f"image.shape: {image.shape} -> [color_channels, height,  width] \n image label: {class_names[label]}")

# Visualizing our data
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.show()

# Plot more images
# torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    print(rows, cols, i)
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(class_names[label])
    plt.axis(False)


plt.show()



# 2 PRepare DataLoader

'''
Right now our data is in the form of Pytorch Datasets.
Dataloader turns our dataset into a python iterable.
More specifically we want to turn our data into batches (or mini batches)
Why would we do this?
    1) it is more computationally efficient as in your computing hardware may not be able to look(store in memory) at 
    60,000 images in one hit. So we break it down to 32 images ata  time (batch size of 32)
    
    2) It gives our neural network more chances to update its gradients per epoch.
    
'''