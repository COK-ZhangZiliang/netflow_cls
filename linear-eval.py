from torchvision import models

from mlp.mlp import *
from DFNet_H_V.DF_cls import *


if __name__ == "__main__":
    timestamp = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        filename=f"results/DoH4_{timestamp}.log", )
    # Load and transform data
    sample_rate = [20, 25, 30]
    device = torch.device("cuda:0")

    # Load pretrained model
    pretrained_model_path = 'models/improved-net.pt'
    resnet = models.resnet50()
    resnet.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
    resnet.load_state_dict(torch.load(pretrained_model_path))
    resnet.to(device)
    resnet.eval()

    for rate in sample_rate:
        data_for_cls, label_for_cls = load_and_transform_data(f'datasets/H_V/traces/traces_{rate}.csv',)
        batch_size = 256
        data_for_cls_rep = torch.zeros((data_for_cls.shape[0], 1000))
        for i in range(0, data_for_cls.shape[0], batch_size):
            data_for_cls_rep[i:i + batch_size] = resnet(
                data_for_cls[i:i + batch_size].reshape(-1, 1, 1, 500).to(device)).detach()

        # Convert to tensors and split
        train_data, test_data, train_label, test_label \
            = train_test_split(data_for_cls_rep, label_for_cls, test_size=0.2, random_state=42)

        # Create dataset and dataloader
        train_dataset_for_cls = CustomDataset(train_data, train_label)
        test_dataset_for_cls = CustomDataset(test_data, test_label)
        train_loader_for_cls = DataLoader(train_dataset_for_cls, batch_size=64, shuffle=True, num_workers=4)
        test_loader_for_cls = DataLoader(test_dataset_for_cls, batch_size=64, shuffle=False, num_workers=4)

        # Define the model
        model_cls = MLPClassifier(1000, [512, 256], 2)
        model_cls.to(device)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(model_cls.parameters(), lr=0.003, weight_decay=0.0001)

        # Train and test
        train_model(model_cls, train_loader_for_cls, criterion, optimizer, epochs=20, device=device)
        info(f"======{rate}======")
        test_model(model_cls, test_loader_for_cls, device=device)
