# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import torch
from torch.nn.modules.loss import _Loss

alpha = 0


# log sum exponential pairwise loss
class LSEPLoss(_Loss):
    def __init__(self):
        super(LSEPLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = 0
        for batch_idx in range(targets.size(0)):
            t = targets[batch_idx]
            o = outputs[batch_idx]
            positive = [np.argwhere(t.detach().cpu().numpy() == 1)]
            negative = [np.argwhere(t.detach().cpu().numpy() == 0)]
            pos_exms = o[positive]
            neg_exms = o[negative].reshape(-1)
            loss += torch.log(1 +
                              torch.sum(torch.exp(neg_exms - pos_exms + alpha)))
        return loss / targets.size(0)
