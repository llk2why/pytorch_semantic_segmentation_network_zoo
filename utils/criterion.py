import torch
import torch.nn as nn
import torch.nn.functional as F

# class CrossEntropyLoss2d(nn.Module):

#     def __init__(self, weight=None):
#         super().__init__()

#         self.loss = nn.NLLLoss2d(weight)

#     def forward(self, outputs, targets):
#         return self.loss(F.log_softmax(outputs), targets)

class NLLLOSS2d_logSoftmax(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs,dim=1), targets)

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)