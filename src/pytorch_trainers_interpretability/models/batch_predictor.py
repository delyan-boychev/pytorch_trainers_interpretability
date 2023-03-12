import torch.nn as nn
import torch

class BatchPredictor(nn.Module):
    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
    def forward(self, x):
        out = []
        if x.shape[0] == self.batch_size:
            return self.model(x)
        else:
            with torch.no_grad():
                for i in range(0, x.shape[0], self.batch_size):
                    out.append(self.model(x[i:i+self.batch_size]))
                return torch.concat(out, dim=0)
        