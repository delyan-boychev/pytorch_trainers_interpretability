import torch
import numpy as np

class AttackStep:
    def __init__(self, epsilon, orig_X, lr):
        self.epsilon = epsilon
        self.orig_X = orig_X
        self.lr = lr
    def step(self, X, grad):
        raise NotImplementedError()
    def project(self, X):
        raise NotImplementedError()
    def random_restart(self, X):
        raise NotImplementedError()

class L2Step(AttackStep):
    def step(self, X, grad):
        norm = torch.norm(grad, p=2, keepdim=True).detach()
        grad_normed = grad.div(norm)
        new_X = X + grad_normed*self.epsilon
        return new_X
    def project(self, X):
        delta = X - self.orig_X
        delta_norm = delta.renorm(p=2, dim=0, maxnorm=self.epsilon)
        new_X = torch.clamp(X+delta_norm, 0, 1)
        return new_X
    def random_restart(self, X):
        new_X = X + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, X.shape)).float()
        new_X = torch.clamp(new_X, 0, 1)
        return new_X
class LinfStep(AttackStep):
    def project(self, X):
        delta = X - self.orig_X
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        new_X = torch.clamp(X+delta, 0, 1)
        return new_X
    def step(self, X, grad):
        step = torch.sign(grad)*self.lr
        new_X = X + step
        return new_X
    def random_restart(self, X):
        new_X = X + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, X.shape)).float()
        new_X = torch.clamp(new_X, 0, 1)
        return new_X

