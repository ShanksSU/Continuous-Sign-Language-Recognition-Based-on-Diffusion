import torch
import torch.nn as nn
import numpy as np

class Glossloss(nn.Module):
    def __init__(self, tau=0.15):
        super(Glossloss, self).__init__()
        self.tau = tau
        self.celoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, g):
        matrix = torch.matmul(x, g.transpose(1, 2)) # [Batch, N_video, N_gloss]
        max_sim, index = torch.max(matrix, 2) 
        valid_mask = (max_sim > self.tau).float() 
        flat_matrix = matrix.view(-1, matrix.size(-1)) # [B*N, N_gloss]
        flat_index = index.view(-1)                    # [B*N]
        flat_mask = valid_mask.view(-1)                # [B*N]

        raw_loss = self.celoss(flat_matrix, flat_index)
        loss = (raw_loss * flat_mask).sum() / (flat_mask.sum() + 1e-6)
        
        return loss