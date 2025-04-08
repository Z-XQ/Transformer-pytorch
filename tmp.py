import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.ln1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(self.relu(self.ln1(x)))
        x = self.ln2(x)

        return x