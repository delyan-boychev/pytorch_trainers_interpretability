# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1706.06083.pdf                    #
# for Projected Gradient Descent                                               #
# ---------------------------------------------------------------------------- #
from . import AttackSteps
import torch.nn as nn
import torch
from tqdm import tqdm

class Attacker:
    def __init__(self, model, num_iter=20, epsilon=8/255, attack_step=AttackSteps.LinfStep, lr=0.01, criterion=None, normalizer=lambda x: x, restart=False, tqdm=True):
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
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.custom_loss = False
        else:
            self.criterion = criterion
            self.custom_loss = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)
    def __call__(self, X, y, targeted=False, fake_relu=False, use_best=True, random_start=False):
        X = X.to(self.device)
        attack_step = self.attack_step(epsilon=self.epsilon, orig_X=X.clone().detach(), lr=self.lr, device=self.device)
        m = -1 if targeted else 1
        if random_start:
            X = attack_step.random_restart(X)
        adv_X = X.requires_grad_(True).to(self.device)
        best_loss = None
        best_X = None 
        iter_no_change = 0
        iterat = range(self.num_iter)
        if self.tqdm is True:
            iterat = tqdm(iterat)
        rn_loss = 0.0
        for i in iterat:
            if iter_no_change > 10 and self.restart is True:
                adv_X = attack_step.random_restart(adv_X)
            adv_X = adv_X.detach().clone().requires_grad_(True)
            loss = None
            if self.custom_loss:
                loss = self.criterion(self.model, self.normalizer(adv_X), y)
            else:
                output = self.model(self.normalizer(adv_X), fake_relu=fake_relu)
                loss = self.criterion(output, y)
            loss = torch.mean(loss)
            rn_loss += loss.item()
            loss = m * loss
            grad, = torch.autograd.grad(loss, [adv_X])
            if best_loss is None or best_loss < loss.item():
                best_loss = loss.item()
                best_X = adv_X.clone().detach()
                iter_no_change = -1
            if use_best is False:
                best_X = adv_X.clone().detach()
            if self.tqdm:
                iterat.set_postfix(loss=m*best_loss)
            adv_X = attack_step.step(adv_X, grad)
            adv_X = attack_step.project(adv_X)
            iter_no_change +=1
        return best_X