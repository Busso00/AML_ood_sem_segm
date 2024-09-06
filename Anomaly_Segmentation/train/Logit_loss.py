import torch.nn as nn
import torch
import torch.nn.functional as F


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


class LogitNormLoss(nn.Module):

    def __init__(self, device, weight=None, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.ce = CrossEntropyLoss2d(weight)
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return self.ce(logit_norm, target)


