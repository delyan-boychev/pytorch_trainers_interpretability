import torch

"""
Sequential with arguments, which is applied to the last layer of the Sequential module.
"""


class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input
