import torch

import logging


def info(msg):
    logging.info(msg)
    print(msg)


def train_model(model, train_loader, criterion, optimizer, epochs=20, device=torch.device("cuda:1"), scheduler=None):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                total_loss += loss.item() * len(data)
                correct += (torch.max(output, 1)[1] == target.view(-1)).sum().item()
        if scheduler:
            scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.4f}, '
                            f'Accuracy: {100*correct/len(train_loader.dataset):.4f}%')
        if (epoch + 1) % 5 == 0:
            info(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.4f}, '
                            f'Accuracy: {100*correct/len(train_loader.dataset):.4f}%')
            

def test_model(model, test_loader, device=torch.device("cuda:1")):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            correct += (torch.max(output, 1)[1] == target.view(-1)).sum().item()
    info(f'Accuracy: {100*correct/len(test_loader.dataset):.4f}%')
