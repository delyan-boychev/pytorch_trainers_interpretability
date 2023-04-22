import torch

"""
Sequential with arguments, which is applied to the last layer of the Sequential module.
"""


class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, inp, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                inp = vs[i](inp, *args, **kwargs)
            else:
                inp = vs[i](inp)
        return inp
