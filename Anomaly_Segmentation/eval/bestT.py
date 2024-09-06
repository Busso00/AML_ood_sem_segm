import torch
import torch.nn as nn 

class FocalLoss(nn.Module):

    def __init__(self, gamma = 2):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs, targets):

        probs = torch.nn.functional.softmax(outputs, dim=1)
        target_probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)

        return (-((1-target_probs)**self.gamma)*torch.log(target_probs)).mean()
