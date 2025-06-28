import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import yaml

from data.load_data import get_data
from models.model import SimpleNN
from train.train import train
from train.evaluate import evaluate
from utils.logger import log


def main():
    with open("config/config.yaml", 'r')as f:
        cfg=yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_data(cfg["batch_size"])
    val_loader = get_data(cfg["batch_size"])

    model=SimpleNN(cfg["input_dim"], cfg["hidden_dim"], cfg["output_dim"]).to(device)
    loss_criterion=nn.CrossEntropyLoss()
    optimizer=Adam(model.parameters(), lr=cfg["lr"])

    best_acc=0.0

    for epoch in range(cfg["epochs"]):
        loss=train(model, train_loader, loss_criterion, optimizer, device)
        eval_loss,acc=evaluate(model, val_loader, device)

        log(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}, Eval_loss={eval_loss:.4f}")

        if acc>best_acc:
            torch.save(model.state_dict(), cfg["model_path"])
            best_acc=acc
            log("✅ Best model saved!")


if __name__ == "__main__":
    main()



