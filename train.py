import matplotlib.pyplot as plt
from pytorch_lightning import Trainer 
import torch
from datamodule import DataModule
from torch.utils.data import DataLoader
from model import Net



def main():
    model = Net()
    trainer = Trainer(max_epochs=1, gpus=1)
    trainer.fit(model)


if __name__ == "__main__":
    main()
