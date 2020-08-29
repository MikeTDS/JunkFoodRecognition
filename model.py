import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from datamodule import DataModule
from torch.optim import Adam


class Net(LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
        self.model.cuda()

        #freeze model
        for child in self.model.children():
            for param in child.parameters():
                param.requries_grad = False

        self.datamodule = DataModule()
        self.l5 = torch.nn.Linear(1000,2)
        #self.l2 = torch.nn.Linear(950,2)

    def forward(self, x):
        x = self.model.forward(x)
        #x = torch.relu(self.l1(x))
        return torch.log_softmax(self.l5(x), dim=1)

    def training_step(self, batch, batch_idx):
        data, target = batch
        data = data.cuda()
        target = target.cuda()
        output = self.forward(data)
        f = torch.nn.CrossEntropyLoss()
        loss = f(output, target)
        return {'loss':loss}
    


    def train_dataloader(self):
        return DataLoader(self.datamodule.get_trainset(), batch_size=32, shuffle=True, num_workers=4)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer