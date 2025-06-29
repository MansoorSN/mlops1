import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data(batch_size):
    x=torch.randn(1000, 10)
    y=torch.randint(0,2,(1000,))
    dataset=TensorDataset(x,y)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader