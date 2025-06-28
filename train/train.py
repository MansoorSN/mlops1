import torch
def train(model, dataloader, loss_criterion, optimizer, device):
    model.train()
    total_loss=0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch=x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output=model(x_batch)
        loss=loss_criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    return total_loss/len(dataloader)
