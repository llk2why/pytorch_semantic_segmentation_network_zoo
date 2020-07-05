import torch
import numpy as np

class ToLabel:
    def __call__(self, mat):
        return torch.from_numpy(np.array(mat)).long()
        # return torch.from_numpy(np.array(mat)).long().unsqueeze(0)
