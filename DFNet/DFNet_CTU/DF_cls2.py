from DF_cls import *


def load_and_transform_data(data_path):
    data = pd.read_csv(data_path)
    info(f"Loading data from {data_path}...")
    data = data.map(lambda x: eval(x))
    data['Label'] = data['Label'].apply(lambda x: x[0])
    data = data[data['Label'].str.contains('From-Botnet|From-Normal|To-Botnet|To-Normal')]
    data['Label'] = data['Label'].apply(lambda x: 1 if x.startswith('flow=From-Botnet') else 0)
    # data_len = min(len(data[data['Label']==1]), len(data[data['Label']==0]))
    # data = pd.concat([data[data['Label']==1].sample(data_len), data[data['Label']==0].sample(data_len)])
    print(data)
    data_for_cls = data.apply(lambda x: [[bytes, counts]
                                         for counts, bytes in zip(x.iloc[0], x.iloc[1])], axis=1)
    data_for_cls = data_for_cls.apply(lambda x: x[:10] if len(x) >= 10 else x + [[0, 0]] * (10 - len(x)))
    data_for_cls = np.array(data_for_cls.tolist())
    data_for_cls = torch.tensor(data_for_cls, dtype=torch.float32)
    data_for_cls = data_for_cls.reshape((len(data_for_cls), 10, -1))
    label_for_cls = data.iloc[:, 2].tolist()
    label_for_cls = torch.tensor(label_for_cls, dtype=torch.long).reshape(-1, 1)
    info(f"Data loaded and transformed!")

    print(data_for_cls, label_for_cls)
    return data_for_cls, label_for_cls


if __name__ == '__main__':
    timestamp = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=f"../../results/CTU2_{timestamp}.log", )

    device = torch.device("cuda:0")
    for i in range(13):
        # Load and transform data
        data_for_cls, label_for_cls = load_and_transform_data(f'../../datasets/CTU-13-Dataset/{i+1}/traces.csv')

        # Convert to tensors and split
        train_data, test_data, train_label, test_label \
            = train_test_split(data_for_cls, label_for_cls, test_size=0.2, random_state=42)
        
        # Create dataset and dataloader
        train_dataset_for_cls = CustomDataset(train_data, train_label)
        test_dataset_for_cls = CustomDataset(test_data, test_label)
        train_loader_for_cls = DataLoader(train_dataset_for_cls, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
        test_loader_for_cls = DataLoader(test_dataset_for_cls, batch_size=64, shuffle=False, num_workers=4, drop_last=True)

        # Model, criterion, optimizer
        model_cls = DFNet(10, 2)
        model_cls.to(device)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Adam(model_cls.parameters(), lr=0.03, weight_decay=0.00001)
        scheduler = ExponentialLR(optimizer, gamma=0.8)
        

        # Train and test
        train_model(model_cls, train_loader_for_cls, criterion, optimizer, epochs=40, device=device, scheduler=scheduler)
        info(f"======{i+1}======")
        test_model(model_cls, test_loader_for_cls, device=device)