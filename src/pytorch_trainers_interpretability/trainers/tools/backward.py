import torch.nn as nn


class Backward:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, X, y):
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, y)
        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        length = y.size(0)
        curr_acc = (predictions == y).sum().item()
        curr_loss = loss.item()
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return curr_loss, curr_acc, length
