import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pre import CustomDataset, DNN, preprocess_data


def load_and_transform_data(bng_data_path, mal_data_path):
    bng_data, mal_data = pd.read_csv(bng_data_path), pd.read_csv(mal_data_path)
    data_for_cls = pd.concat([bng_data.sample(len(mal_data)).iloc[:, :5], mal_data.iloc[:, :5]], axis=0)
    label_for_cls = pd.concat([pd.DataFrame({'label': [0] * len(mal_data)}),
                               pd.DataFrame({'label': [1] * len(mal_data)})], axis=0)

    # Preprocess data
    data_for_cls = preprocess_data(data_for_cls)

    return data_for_cls, label_for_cls

def train_model(model, train_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(data)
            correct += (torch.max(output, 1)[1] == target.view(-1)).sum().item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.4f}, '
              f'Accuracy: {100*correct/len(train_loader.dataset):.4f}%')


def test_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            correct += (torch.max(output, 1)[1] == target.view(-1)).sum().item()
    print(f'Accuracy: {100*correct/len(test_loader.dataset):.4f}%')


if __name__ == '__main__':
    # Load and transform data
    data_for_cls, label_for_cls = load_and_transform_data('./datasets/DoH/bng_data.csv',
                                                          './datasets/DoH/mal_data.csv')

    # Convert to tensors and split
    data = torch.tensor(data_for_cls.values.astype(np.float32))
    label = torch.tensor(label_for_cls.values.astype(np.float32), dtype=torch.long)
    train_data, test_data, train_label, test_label \
        = train_test_split(data, label, test_size=0.2, random_state=42)

    # Create dataset and dataloader
    train_dataset_for_cls = CustomDataset(train_data, train_label)
    test_dataset_for_cls = CustomDataset(test_data, test_label)
    train_loader_for_cls = DataLoader(train_dataset_for_cls, batch_size=32, shuffle=True, num_workers=4)
    test_loader_for_cls = DataLoader(test_dataset_for_cls, batch_size=32, shuffle=False, num_workers=4)

    # Model, criterion, optimizer
    model_cls = DNN(14, 2)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model_cls.parameters(), lr=0.001, weight_decay=0.0001)

    # Train and test
    train_model(model_cls, train_loader_for_cls, criterion, optimizer, epochs=20)
    test_model(model_cls, test_loader_for_cls)
