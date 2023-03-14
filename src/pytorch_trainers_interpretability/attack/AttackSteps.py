# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1706.06083.pdf                    #
# attack steps for Projected Gradient Descent                                  #
# ---------------------------------------------------------------------------- #

import torch
import numpy as np


class AttackStep:
    def __init__(self, epsilon, orig_X, lr, device):
        self.epsilon = epsilon
        self.orig_X = orig_X
        self.lr = lr
        self.device = device

    def step(self, X, grad):
        raise NotImplementedError()

    def project(self, X):
        raise NotImplementedError()

    def random_restart(self, X):
        raise NotImplementedError()


class L2Step(AttackStep):
    def step(self, X, grad):
        l = len(X.shape) - 1
        g_norm = torch.norm(
            grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = grad / (g_norm + 1e-10)
        return X + scaled_g * self.lr

    def project(self, X):
        delta = X - self.orig_X
        delta_norm = delta.renorm(p=2, dim=0, maxnorm=self.epsilon)
        new_X = torch.clamp(self.orig_X+delta_norm, 0, 1)
        return new_X

    def random_restart(self, X):
        l = len(X.shape) - 1
        rp = torch.randn_like(X)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return torch.clamp(X + self.epsilon * rp / (rp_norm + 1e-10), 0, 1)


class LinfStep(AttackStep):
    def project(self, X):
        delta = X - self.orig_X
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        new_X = torch.clamp(self.orig_X+delta, 0, 1)
        return new_X

    def step(self, X, grad):
        step = torch.sign(grad)*self.lr
        new_X = X + step
        return new_X

    def random_restart(self, X):
        pert = torch.rand_like(X)
        new_X = self.project(self.orig_X + pert)
        return new_X
