import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = torch.cdist(
            anchor.unsqueeze(0), positive.unsqueeze(0), p=2)
        distance_negative = torch.cdist(
            anchor.unsqueeze(0), negative.unsqueeze(0), p=2)

        losses = torch.log(
            1 + torch.exp(distance_positive - distance_negative))

        return losses.mean() if size_average else losses.sum()
