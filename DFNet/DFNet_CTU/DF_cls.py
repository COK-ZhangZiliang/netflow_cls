import sys
sys.path.append('/home/zhangziliang/netflow_cls/')

from DFNet.DFNet_DoH.DF_cls import *

if __name__ == '__main__':
    timestamp = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                        filename=f"../results/CTU_{timestamp}.log", )

    device = torch.device("cuda:0")
    input_channels = 1
    num_classes = 2
    lr = 0.0003
    epochs = 40
    logging.info(f'lr: {lr}, num_epochs: {num_epochs}')

    for i in range(13):
        # Load data
        data_for_cls, label_for_cls = load_avg_CTU(f'../../datasets/CTU-13-Dataset/proc/{i+1}')

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
        optimizer = optim.Adam(model_cls.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ExponentialLR(optimizer, gamma=0.75)

        # Train and test
        train_model(model_cls, train_loader_for_cls, criterion, optimizer, epochs=epochs, device=device)
        logging.info(f"==================== CTU-{i+1} ====================")
        test_model(model_cls, test_loader_for_cls, num_classes=num_classes, device=device)
