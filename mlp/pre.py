import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init


class CustomDataset(Dataset):
    """
        define a custom dataset
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y


class DNN(nn.Module):
    """
        define a simple DNN
    """
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc4(x)
        return x


def preprocess_data(data):
    data.reset_index(drop=True, inplace=True)
    data['tcp_flags'] = data['tcp_flags'].apply(eval)
    expanded_tcp_flags = pd.DataFrame(data['tcp_flags'].tolist(), columns=['tcp_flags_' + str(i) for i in range(10)])
    data = pd.concat([data, expanded_tcp_flags], axis=1)
    data.drop('tcp_flags', axis=1, inplace=True)
    data['ip_protocols'] = data['ip_protocols'].apply(eval)
    data['ip_protocols'] = data['ip_protocols'].apply(lambda x: ','.join(map(str, x)))
    return data


def load_and_transform_data(bng_data_path, mal_data_path, bng_label_path, mal_label_path):
    bng_data, mal_data = pd.read_csv(bng_data_path), pd.read_csv(mal_data_path)
    bng_label, mal_label = pd.read_csv(bng_label_path), pd.read_csv(mal_label_path)
    bng_num_for_pre, mal_num_for_pre = bng_data.shape[0] // 2, mal_data.shape[0] // 2
    data_for_pre = pd.concat([bng_data.iloc[:bng_num_for_pre, :5], mal_data.iloc[:mal_num_for_pre, :5]], axis=0)
    label_for_pre = pd.concat([bng_label.iloc[:bng_num_for_pre, :10], mal_label.iloc[:mal_num_for_pre, :10]], axis=0)

    # Preprocess data
    data_for_pre = preprocess_data(data_for_pre)

    return data_for_pre, label_for_pre


def train_model(model, train_loader, criterion, optimizer, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss/len(train_loader.dataset):.4f}")


def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
    print(f"Test Loss: {test_loss/len(test_loader.dataset):.4f}")


if __name__ == '__main__':
    # Load and preprocess data
    data_for_pre, label_for_pre = load_and_transform_data('./datasets/DoH/bng_data.csv',
                                                          './datasets/DoH/mal_data.csv',
                                                          './datasets/DoH/bng_label.csv',
                                                          './datasets/DoH/mal_label.csv')

    # Convert to tensors and split
    data = torch.tensor(data_for_pre.values.astype(np.float32))
    label = torch.tensor(label_for_pre.values.astype(np.float32))
    train_data, test_data, train_label, test_label \
        = train_test_split(data, label, test_size=0.2, random_state=42)

    # Create dataset and dataloader
    train_dataset_for_pre = CustomDataset(train_data, train_label)
    test_dataset_for_pre = CustomDataset(test_data, test_label)
    train_loader_for_pre = DataLoader(train_dataset_for_pre, batch_size=32, shuffle=True, num_workers=4)
    test_loader_for_pre = DataLoader(test_dataset_for_pre, batch_size=32, shuffle=False, num_workers=4)

    # Model, criterion, optimizer
    model_pre = DNN(14, 10)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model_pre.parameters(), lr=0.001, weight_decay=0.0001)

    # Train and test
    train_model(model_pre, train_loader_for_pre, criterion, optimizer, num_epochs=30)
    test_model(model_pre, test_loader_for_pre, criterion)

    # Save model
    torch.save(model_pre.state_dict(), './models/pre.pth')
