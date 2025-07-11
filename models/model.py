import torch
import torch.nn as nn



class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):

        super(SimpleNN, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input):
        return self.net(input)
    