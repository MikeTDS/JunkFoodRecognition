import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from datamodule import DataModule
from torch.optim import Adam


class Net(LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)
        self.model.cuda()
        self.datamodule = DataModule()

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        data = data.cuda()
        target = target.cuda()
        output = self.forward(data)
        f = torch.nn.CrossEntropyLoss()
        loss = f(output, target)
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        data = data.cuda()
        target = target.cuda()
        output = self.forward(data)
        f = torch.nn.CrossEntropyLoss()
        loss = f(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'loss' : loss, 'correct' : correct}
    
    def train_dataloader(self):
        return DataLoader(self.datamodule.get_trainset(), batch_size=32, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.datamodule.get_valset(), batch_size=32, num_workers=4)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer