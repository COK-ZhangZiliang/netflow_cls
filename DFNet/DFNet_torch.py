import torch.nn as nn
from torch.utils.data import Dataset


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


class ConvBlock(nn.Module):
    """
        define a convolutional block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_kernel_size, pool_stride,
                 pool_padding, dropout_rate):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class DFNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DFNet, self).__init__()
        # Define the hyperparameters for the convolutional blocks
        filter_nums = ['None', 32, 64, 128, 256]
        kernel_size = ['None', 8, 8, 8, 8]
        stride_size = ['None', 1, 1, 1, 1]
        padding = 4
        pool_kernel_size = ['None', 8, 8, 8, 8]
        pool_stride_size = ['None', 4, 4, 4, 4]
        pool_padding = 4
        dropout_rate = ['None', 0.1, 0.1, 0.1, 0.1]

        self.block1 = ConvBlock(input_channels, filter_nums[1], kernel_size[1], stride_size[1], padding,
                                pool_kernel_size[1], pool_stride_size[1], pool_padding, dropout_rate[1])
        self.block2 = ConvBlock(filter_nums[1], filter_nums[2], kernel_size[2], stride_size[2], padding,
                                pool_kernel_size[2], pool_stride_size[2], pool_padding, dropout_rate[2])
        self.block3 = ConvBlock(filter_nums[2], filter_nums[3], kernel_size[3], stride_size[3], padding,
                                pool_kernel_size[3], pool_stride_size[3], pool_padding, dropout_rate[3])
        self.block4 = ConvBlock(filter_nums[3], filter_nums[4], kernel_size[4], stride_size[4], padding,
                                pool_kernel_size[4], pool_stride_size[4], pool_padding, dropout_rate[4])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x
