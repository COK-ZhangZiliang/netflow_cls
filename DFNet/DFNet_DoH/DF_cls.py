import sys
sys.path.append('/home/zhangziliang/netflow_cls/')

import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from DFNet.DFNet_torch import DFNet
from DFNet.utils import *
from utils.load_data import *

if __name__ == '__main__':
    timestamp = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                        filename=f"../results/DoH_{timestamp}.log", )

    
    act_timeout = [1, 5, 10, 15, 20, 25, 30, 60]
    device = torch.device("cuda:0")
    input_channels = 1
    num_classes = 4

    for t in act_timeout:
        # Load data
        data_for_cls, label_for_cls = load_avg_DoH(t, '../../datasets/DoH/nfv5')

        # Split data
        train_data, test_data, train_label, test_label \
            = train_test_split(data_for_cls, label_for_cls, test_size=0.2, random_state=42)
        
        # Create dataset and dataloader
        train_dataset_for_cls = TensorDataset(train_data, train_label)
        test_dataset_for_cls = TensorDataset(test_data, test_label)
        train_loader_for_cls = DataLoader(train_dataset_for_cls, batch_size=256, shuffle=True, num_workers=4)
        test_loader_for_cls = DataLoader(test_dataset_for_cls, batch_size=256, shuffle=False, num_workers=4)

        # Model, criterion, optimizer
        model_cls = DFNet(input_channels, num_classes)
        model_cls.to(device)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Adam(model_cls.parameters(), lr=0.01, weight_decay=1e-5)
        scheduler = ExponentialLR(optimizer, gamma=0.75)

        # Train and test
        train_model(model_cls, train_loader_for_cls, criterion, optimizer, epochs=40, device=device)
        info(f"======{t}======")
        test_model(model_cls, test_loader_for_cls, num_classes=num_classes, device=device)
