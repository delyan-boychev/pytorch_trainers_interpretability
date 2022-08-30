from . import AttackSteps
import torch.nn as nn
import torch
from tqdm import tqdm

class Attacker:
    def __init__(self, model, num_iter=20, epsilon=8/255, attack_step=AttackSteps.LinfStep, lr=0.1, normalizer=None, restart=True, tqdm=True):
        if not isinstance(model, nn.Module):
            raise("Not valid model")
        self.model = model
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.attack_step = attack_step
        self.lr = lr
        self.tqdm = tqdm
        self.restart = restart
        self.normalizer = normalizer
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)
    def __call__(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)
        attack_step = self.attack_step(epsilon=self.epsilon, orig_X=X.clone().detach(), lr=self.lr, device=self.device)
        adv_X = attack_step.random_restart(X).to(self.device).requires_grad_(True)
        adv_X = adv_X.to(self.device)
        best_loss = None
        best_X = None 
        iter_no_change = 0
        iterat = range(self.num_iter)
        if self.tqdm is True:
            iterat = tqdm(iterat)
        for i in range(self.num_iter):
            if iter_no_change > 10 and self.restart is True:
                adv_X = attack_step.random_restart(adv_X)
            adv_X = adv_X.detach().clone().requires_grad_(True)
            if self.normalizer is not None:
                output = self.model(self.normalizer(adv_X))
            else:
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