from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

class DataModule():

    def __init__(self):
        self.train_path = './dataset/images/train'
        self.val_path = './dataset/images/test'
        self.train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    

    def get_trainset(self):
        return datasets.ImageFolder(self.train_path, transform=self.train_transform)
    
    def get_valset(self):
        return datasets.ImageFolder(self.val_path, transform=self.val_transform)
            
