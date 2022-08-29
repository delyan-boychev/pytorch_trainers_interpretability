from .AttackSteps import *
import torch.nn as nn
import torch


class Attacker:
    def __init__(self, model, num_iter=20, epsilon=0.5, attack_step=L2Step, lr=0.1, restart=True):
        if not isinstance(model, nn.Module):
            raise("Not valid model")
        self.model = model
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.attack_step = attack_step
        self.lr = lr
        self.restart = restart
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)
    def __call__(self, X, y):
        attack_step = self.attack_step(epsilon=self.epsilon, orig_X=X, lr=self.lr)
        adv_X = attack_step.random_restart(X).to(self.device).requires_grad_(True)

        best_loss = None
        best_X = None
        iter_no_change = -1

        for i in range(self.num_iter):
            if iter_no_change > 1:
                adv_X = attack_step.random_restart(adv_X)
            adv_X = adv_X.detach().clone().requires_grad_(True)
            output = self.model(adv_X)
            loss = self.criterion(output, y)
            loss.backward()
            if best_loss is None or best_loss < loss.item():
                best_loss = loss.item()
                best_X = adv_X.clone().detach()
                iter_no_change = -1
            adv_X = attack_step.step(adv_X, adv_X.grad.detach().clone())
            adv_X = attack_step.project(adv_X)
            iter_no_change +=1
        return best_X
