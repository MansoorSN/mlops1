import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data(batch_size):
    x=torch.randn(10000, 10)
    y=torch.randint(0,2,(10000,))
    dataset=TensorDataset(x,y)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader