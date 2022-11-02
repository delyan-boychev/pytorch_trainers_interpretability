import torch

class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs
