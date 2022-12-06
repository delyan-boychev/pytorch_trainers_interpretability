import torch


"""
Method which is used to for activation maximization. It is ReLU in pass forward, but in backward pass it return the gradients directly. 
We want not to have zero gradients because we analyze the activations.
"""


class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs
