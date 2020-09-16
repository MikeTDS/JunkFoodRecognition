import torch
from torch.utils.data import DataLoader

from datamodule import DataModule
from model import Net


def test(batch_size=32, path="./checkpoints/model.pt"):
    all_samples = 0
    counter = 0

    model = Net()
    model.load_state_dict(torch.load(path))
    model.eval()
    model.cuda()
    data_module = DataModule()
    data_loader = DataLoader(
        data_module.get_valset(), batch_size=batch_size, num_workers=4, shuffle=True
    )

    for x in iter(data_loader):
        data, target = x
        data = data.cuda()
        with torch.no_grad():
            output = model(data)
            for i in range(batch_size):
                all_samples += 1
                if output[i][0] == max(output[i]):
                    out = 0
                else:
                    out = 1
                if out == target[i].cpu().detach().numpy():
                    counter += 1
        print("checked:", all_samples)

    print("acc:", counter / all_samples)


if __name__ == "__main__":
    test()
