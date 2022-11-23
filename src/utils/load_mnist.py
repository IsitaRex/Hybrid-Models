#import datasets from torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_mnist(batch_size):
    """Load MNIST dataset.
    
    Args:
        batch_size (int): Batch size.
    
    Returns:
        trainloader: A DataLoader for iterating over the train set MNIST dataset.
        testloader: A DataLoader for iterating over the test set MNIST dataset.
    """
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])
    
    trainset = datasets.MNIST('data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.MNIST('data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader