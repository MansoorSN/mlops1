import torch
from torch.utils.tensorboard import SummaryWriter

def train(model, dataloader, loss_criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss=0

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch=x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output=model(x_batch)
        loss=loss_criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

        writer.add_scalar('Train_Loss/train_batch', loss.item(), epoch*len(dataloader)+batch_idx)

    avg_loss=total_loss/len(dataloader)
    writer.add_scalar('Train_Loss/train_epoch', avg_loss, epoch)
    return avg_loss
