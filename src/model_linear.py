import torch
import torch.nn as nn

class LinearPredictor(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        return self.fc(x)
