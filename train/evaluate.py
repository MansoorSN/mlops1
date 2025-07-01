import torch
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, dataloader, device, writer=None, epoch=None):
    model.eval()
    correct=0
    total=0
    loss_score=0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch= x_batch.to(device), y_batch.to(device)

            outputs=model(x_batch)
            score, predicted=outputs.max(1)
            loss_score+=score.sum().item()
            correct+=(predicted==y_batch).sum().item()
            total+=y_batch.size(0)

    accuracy=correct/total
    eval_loss=loss_score/total

    if writer and epoch is not None:
        writer.add_scalar('Eval/Accuracy', accuracy, epoch)
        writer.add_scalar('Eval/Loss', eval_loss, epoch)
    
    return (eval_loss, accuracy)

