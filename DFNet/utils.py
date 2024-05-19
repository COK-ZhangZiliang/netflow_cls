import torch
import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def info(msg):
    logging.info(msg)
    print(msg)


def train_model(model, train_loader, criterion, optimizer, epochs=20, device=torch.device("cuda"), scheduler=None):
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
            

def test_model(model, test_loader, num_classes, device=torch.device("cuda")):
    model.eval()
    output = torch.zeros(0, num_classes).to(device)
    label = torch.zeros(0,).to(device)
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = torch.cat((output, model(data)), dim=0)
            label = torch.cat((label, target.to(device)), dim=0)

        pred = torch.max(output, 1)[1].cpu().numpy()
        label = label.cpu().numpy()

        # Calculate metrics
        acc = accuracy_score(label, pred)
        pre= precision_score(label, pred, average=None)
        rec = recall_score(label, pred, average=None)
        f1 = f1_score(label, pred, average=None)
        mac_pre = precision_score(label, pred, average='macro')
        mac_rec = recall_score(label, pred, average='macro')
        mac_f1 = f1_score(label, pred, average='macro')
        mic_pre = precision_score(label, pred, average='micro')
        mic_rec = recall_score(label, pred, average='micro')
        mic_f1 = f1_score(label, pred, average='micro')

        info(f'Accuracy: {acc:.4f}')
        info(f'Precision: {pre}')
        info(f'Recall: {rec}')
        info(f'F1: {f1}')
        info(f'Macro Precision: {mac_pre:.4f}')
        info(f'Macro Recall: {mac_rec:.4f}')
        info(f'Macro F1: {mac_f1:.4f}')
        info(f'Micro Precision: {mic_pre:.4f}')
        info(f'Micro Recall: {mic_rec:.4f}')
        info(f'Micro F1: {mic_f1:.4f}')