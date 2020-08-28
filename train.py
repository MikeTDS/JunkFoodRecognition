import matplotlib.pyplot as plt
from pytorch_lightning import Trainer 
import torch
from datamodule import DataModule
from torch.utils.data import DataLoader
from model import Net



def main():
    save_path = './checkpoints/model.pt'
    model = Net()
    model.load_state_dict(torch.load(save_path))
    trainer = Trainer(
        max_epochs=10, 
        gpus=1,  
        show_progress_bar=True
    )
    trainer.fit(model)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
