import torch.nn as nn
import torch


class ParwiseLoss(nn.Module):

    def __init__(self):
        super(ParwiseLoss, self).__init__()
        self.margin = 1

    def forward(self, x, x1, y):
        pdist = nn.PairwiseDistance()
        distances = pdist(x, x1)
        distances = distances.cpu()
        y = y.cpu()
        zeros = torch.zeros(distances.size()[0])
        losses = 0.5 * (
                (1 - y) * distances.pow(2) +
                y * torch.max(zeros, self.margin - distances).pow(2))

        return losses.mean()
