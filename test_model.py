import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from model import Net
from datamodule import DataModule

RANGE = 32
samples = 0
counter = 0

save_path = './checkpoints/model.pt'
model = Net()
model.load_state_dict(torch.load(save_path))
model.eval()
model.cuda()
data_module = DataModule()
data_loader = DataLoader(data_module.get_valset(), batch_size=32, num_workers=4, shuffle=True)


for x in iter(data_loader):
    data, target = x
    data = data.cuda()
    with torch.no_grad():
        output = model(data)
        for i in range(RANGE):
            samples += 1
            if output[i][0] == max(output[i]):
                out = 0
            else:
                out = 1
            #print(out, target[i].cpu().detach().numpy())
            if(out == target[i].cpu().detach().numpy()):
                counter += 1
    print("checked:", samples)

print("acc:", counter/samples)
    