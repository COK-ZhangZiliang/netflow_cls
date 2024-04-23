from DF_cls import *


def load_and_transform_data(data_path):
    data = pd.read_csv(data_path)
    print(len(data))
    info(f"Loading data...")
    data = data.iloc[:, 1].apply(lambda x: eval(x))
    bng_data = data[data.apply(lambda x: x[1] == 0)]
    mal_data = data[data.apply(lambda x: x[1] == 1)]
    data_len = min(len(bng_data), len(mal_data))
    data = pd.concat([bng_data.sample(data_len), mal_data.sample(data_len)])
    print(len(data))
    data_for_cls = data.apply(lambda x: [bytes for _, bytes in x[0]])
    data_for_cls = data_for_cls.apply(lambda x: x[:100] if len(x) >= 100 else x + [0] * (100 - len(x)))
    data_for_cls = np.array(data_for_cls.tolist())
    data_for_cls = torch.tensor(data_for_cls, dtype=torch.float32)
    data_for_cls = data_for_cls.reshape((len(data_for_cls), 1, -1))
    label_for_cls = data.apply(lambda x: x[1]).tolist()
    label_for_cls = torch.tensor(label_for_cls, dtype=torch.long).reshape(-1, 1)
    info(f"Data loaded and transformed!")

    print(data_for_cls, label_for_cls)
    return data_for_cls, label_for_cls


if __name__ == '__main__':
    timestamp = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                        filename=f"../../results/H_V2_{timestamp}.log", )

    # Load and transform data
    sample_rate = [0.1, 1, 5, 10, 15, 20, 25, 30]
    device = torch.device("cuda:1")
    for rate in sample_rate:
        data_for_cls, label_for_cls = load_and_transform_data(f'../../datasets/H_V/traces/traces_{rate}.csv')

        # Convert to tensors and split
        train_data, test_data, train_label, test_label \
            = train_test_split(data_for_cls, label_for_cls, test_size=0.2, random_state=42)
        
        # Create dataset and dataloader
        train_dataset_for_cls = CustomDataset(train_data, train_label)
        test_dataset_for_cls = CustomDataset(test_data, test_label)
        train_loader_for_cls = DataLoader(train_dataset_for_cls, batch_size=256, shuffle=True, num_workers=4, drop_last=True)
        test_loader_for_cls = DataLoader(test_dataset_for_cls, batch_size=256, shuffle=False, num_workers=4, drop_last=True)

        # Model, criterion, optimizer
        model_cls = DFNet(1, 2)
        model_cls.to(device)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Adam(model_cls.parameters(), lr=0.00003, weight_decay=0.000001)

        # Train and test
        train_model(model_cls, train_loader_for_cls, criterion, optimizer, epochs=40, device=device)
        info(f"======{rate}======")
        test_model(model_cls, test_loader_for_cls, device=device)
