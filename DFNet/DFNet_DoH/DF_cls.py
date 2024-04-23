import sys
sys.path.append('/home/zhangziliang/netflow_cls/')

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from DFNet.DFNet_torch import DFNet, CustomDataset

import time

from DFNet.utils import *


def load_and_transform_data(bng_data_path, mal_data_path):
    bng_data, mal_data = pd.read_csv(bng_data_path), pd.read_csv(mal_data_path)
    info(f"Loading benign data...")
    bng_data = bng_data.iloc[:, 1].apply(lambda x: eval(x))
    bng_data = bng_data.apply(lambda x: [[bytes//counts//10+1]*counts 
                                         for counts, bytes in x])
    bng_data = bng_data.apply(lambda x: [item for sublist in x for item in sublist])
    bng_data = bng_data.apply(lambda x: x[:500] if len(x) >= 500 else x + [0] * (500 - len(x)))
    info(f"Loading malicious data...")
    mal_data = mal_data.iloc[:, 1].apply(lambda x: eval(x))
    mal_data = mal_data.apply(lambda x: [[bytes//counts//10+1]*counts 
                                         for counts, bytes in x])
    mal_data = mal_data.apply(lambda x: [item for sublist in x for item in sublist])
    mal_data = mal_data.apply(lambda x: x[:500] if len(x) >= 500 else x + [0] * (500 - len(x)))
    data_len = min(len(bng_data), len(mal_data))
    data_for_cls = pd.concat([bng_data.sample(data_len), mal_data.sample(data_len)], axis=0)
    data_for_cls = np.array(data_for_cls.tolist())
    data_for_cls = torch.tensor(data_for_cls, dtype=torch.float32)
    data_for_cls = data_for_cls.reshape((2 * data_len, 1, -1))
    label_for_cls = torch.tensor([0] * data_len + [1] * data_len, dtype=torch.long).reshape(-1, 1)
    info(f"Data loaded and transformed!")

    return data_for_cls, label_for_cls


if __name__ == '__main__':
    timestamp = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                        filename=f"../../results/DoH_{timestamp}.log", )

    # Load and transform data
    sample_rate = ['inf', 0.1, 1, 5, 10, 15, 20, 25, 30, 60, 120, 180]
    device = torch.device("cuda:0")
    for rate in sample_rate:
        data_for_cls, label_for_cls = load_and_transform_data(f'../../datasets/DoH/traces2/bng_{rate}.csv',
                                                        f'../../datasets/DoH/traces2/mal_{rate}.csv')

        # Convert to tensors and split
        train_data, test_data, train_label, test_label \
            = train_test_split(data_for_cls, label_for_cls, test_size=0.2, random_state=42)
        
        # Create dataset and dataloader
        train_dataset_for_cls = CustomDataset(train_data, train_label)
        test_dataset_for_cls = CustomDataset(test_data, test_label)
        train_loader_for_cls = DataLoader(train_dataset_for_cls, batch_size=256, shuffle=True, num_workers=4)
        test_loader_for_cls = DataLoader(test_dataset_for_cls, batch_size=256, shuffle=False, num_workers=4)

        # Model, criterion, optimizer
        model_cls = DFNet(1, 2)
        model_cls.to(device)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Adam(model_cls.parameters(), lr=0.0001, weight_decay=0.01)

        # Train and test
        train_model(model_cls, train_loader_for_cls, criterion, optimizer, epochs=20, device=device)
        info(f"======{rate}======")
        test_model(model_cls, test_loader_for_cls, device=device)
