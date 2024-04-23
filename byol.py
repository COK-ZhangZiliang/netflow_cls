import time
import logging

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torch.optim.lr_scheduler import ExponentialLR

from byol_pytorch import BYOL
from DFNet.DFNet_H_V.DF_cls import info

class CustomDataset(Dataset):
    """
        define a custom dataset
    """
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        x1 = self.data[self.transform[idx]][idx]
        x2 = self.data[0][idx]
        return x1, x2


def load_and_transform_data(data_path):
    data = pd.read_csv(data_path)
    data = data.iloc[:len(data)//30, :]
    print(len(data))
    info(f"Loading data...")
    data = data.iloc[:, 0].apply(lambda x: eval(x))
    data = data.apply(lambda x: [[bytes//counts//10+1]*counts 
                                         for counts, bytes in x])
    data = data.apply(lambda x: [item for sublist in x for item in sublist])
    data = data.apply(lambda x: x[:500] if len(x) >= 500 else x + [0] * (500 - len(x)))
    data = torch.tensor(np.array(data.tolist()), dtype=torch.float32)
    data = data.reshape((len(data), 1, 1, -1))
    info(f"Data loaded and transformed!")

    print(data)
    return data


def pretrain(learner, data_loader, num_epochs, opt, device):
    for i in range(num_epochs):
        learner.train()
        tot_loss = 0
        for data1, data2 in data_loader:
            data1 = data1.to(device)
            data2 = data2.to(device)
            opt.zero_grad()
            loss = learner(data1, data2)
            loss.backward()
            opt.step()
            learner.update_moving_average()  # update moving average of target encoder
            with torch.no_grad():
                tot_loss += loss.item()
        scheduler.step()
        info(f"epoch {i+1}, loss: {tot_loss/len(data_loader.dataset):.8f}")

# pretrain the network
if __name__ == '__main__':
    timestamp = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=f"./results/pretrain_{timestamp}.log", )

    sample_rate = ['inf', 0.1, 1, 5, 10, 15, 20, 25, 30, 60, 120, 180]
    device = torch.device("cuda:1")

    # define the network
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # change the first layer to accept single channel input
    original_first_layer = resnet.conv1.weight
    new_first_layer_weight = original_first_layer.mean(dim=1, keepdim=True)
    new_first_layer = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, 
                                      stride=2, padding=3, bias=False)
    new_first_layer.weight = torch.nn.Parameter(new_first_layer_weight)
    resnet.conv1 = new_first_layer
    learner = BYOL(
        resnet,
        hidden_layer = 'avgpool'
    )
    learner = learner.to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=0.03)
    scheduler = ExponentialLR(opt, gamma=0.75)
    num_epochs = 40

    # the dataset
    data = []
    for rate in sample_rate:
        info(f"Loading data for rate {rate}")
        _data = load_and_transform_data(f'./datasets/pretrain/traces/traces_{rate}.csv')
        data.append(_data)
    transform = torch.randint(1, 12, size=(len(data[0]),))
    dataset = CustomDataset(data, transform)
    data_loader = DataLoader(dataset, batch_size=2048, shuffle=True)

    # pretrain the network
    pretrain(learner, data_loader, num_epochs, opt, device)

    # save improved network
    info("Saving improved network")
    torch.save(resnet.state_dict(), './models/improved-net.pt')
    