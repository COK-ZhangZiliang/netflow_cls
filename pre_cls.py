import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pre import CustomDataset, DNN, preprocess_data
from cls import train_model, test_model

def load_and_transform_data(bng_data_path, mal_data_path):
    bng_data, mal_data = pd.read_csv(bng_data_path), pd.read_csv(mal_data_path)
    bng_num_for_cls = mal_data.shape[0] - mal_data.shape[0] // 2
    bng_data_idx, mal_data_idx = (
        bng_data.shape[0] - bng_num_for_cls, mal_data.shape[0] // 2)
    data_for_cls = pd.concat([bng_data.iloc[bng_data_idx:, :5], mal_data.iloc[mal_data_idx:, :5]], axis=0)
    label_for_cls = pd.concat([pd.DataFrame({'label': [0] * bng_num_for_cls}),
                               pd.DataFrame({'label': [1] * bng_num_for_cls})], axis=0)

    # Preprocess data
    data_for_cls = preprocess_data(data_for_cls)

    return data_for_cls, label_for_cls

if __name__ == '__main__':
    # Load and transform data
    data_for_cls, label_for_cls = load_and_transform_data('./datasets/DoH/bng_data.csv',
                                                          './datasets/DoH/mal_data.csv')

    # Convert to tensors and split
    model_pre = DNN(14, 10)
    model_pre.load_state_dict(torch.load('./models/pre.pth'))
    model_pre.eval()
    data = torch.tensor(data_for_cls.values.astype(np.float32))
    data = model_pre(data).detach()
    label = torch.tensor(label_for_cls.values.astype(np.float32), dtype=torch.long)
    train_data, test_data, train_label, test_label \
        = train_test_split(data, label, test_size=0.2, random_state=42)

    # Create dataset and dataloader
    train_dataset_for_cls = CustomDataset(train_data, train_label)
    test_dataset_for_cls = CustomDataset(test_data, test_label)
    train_loader_for_cls = DataLoader(train_dataset_for_cls, batch_size=32, shuffle=True, num_workers=4)
    test_loader_for_cls = DataLoader(test_dataset_for_cls, batch_size=32, shuffle=False, num_workers=4)

    # Model, criterion, optimizer
    model_cls = DNN(10, 2)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model_cls.parameters(), lr=0.001, weight_decay=0.0001)

    # Train and test
    train_model(model_cls, train_loader_for_cls, criterion, optimizer, epochs=20)
    test_model(model_cls, test_loader_for_cls)
