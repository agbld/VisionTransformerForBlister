import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

class KoLeoLoss_Triplet(nn.Module):
    def __init__(self):
        super(KoLeoLoss_Triplet, self).__init__()

    def forward(self, anchor, positive, negative):
        d_anchor_anchor = torch.cdist(anchor, anchor)
        d_anchor_positive = torch.cdist(anchor, positive)
        d_anchor_negative = torch.cdist(anchor, negative)
        d = torch.cat((d_anchor_anchor, d_anchor_positive, d_anchor_negative), dim=1)
        d_remove_self = torch.where(d == 0, d.max(), d)
        min_d = torch.min(d_remove_self, dim=1)[0]
        
        n = anchor.shape[0]
        
        loss_koleo = -(1/n) * torch.sum(torch.log(min_d))
        
        return loss_koleo
    
class KoLeoLoss_Contrastive(nn.Module):
    def __init__(self):
        super(KoLeoLoss_Contrastive, self).__init__()

    def forward(self, anchor, sample):
        d_anchor_anchor = torch.cdist(anchor, anchor)
        d_anchor_sample = torch.cdist(anchor, sample)
        d = torch.cat((d_anchor_anchor, d_anchor_sample), dim=1)
        d_remove_self = torch.where(d == 0, d.max(), d)
        min_d = torch.min(d_remove_self, dim=1)[0]
        
        n = anchor.shape[0]
        
        loss_koleo = -(1/n) * torch.sum(torch.log(min_d))
        
        return loss_koleo

class TripletLoss_bug(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(TripletLoss_bug, self).__init__()

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = torch.cdist(
            anchor.unsqueeze(0), positive.unsqueeze(0), p=2)
        distance_negative = torch.cdist(
            anchor.unsqueeze(0), negative.unsqueeze(0), p=2)

        losses = torch.log(
            1 + torch.exp(distance_positive - distance_negative))

        return losses.mean() if size_average else losses.sum()

# my triplet loss fix, using in this project
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = torch.cdist(
            anchor.unsqueeze(1), positive.unsqueeze(1), p=2)
        distance_negative = torch.cdist(
            anchor.unsqueeze(1), negative.unsqueeze(1), p=2)

        losses = torch.log(
            1 + torch.exp(distance_positive - distance_negative))

        return losses.mean() if size_average else losses.sum()

# triplet loss fix solution from lab 
class TripletLoss_fix2(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(TripletLoss_fix2, self).__init__()

    def forward(self, anchor, positive, negative, size_average=True):
        # cos_sim = torch.nn.CosineSimilarity(dim=1)

        # CosineSimilarity
        # cos = nn.CosineSimilarity(dim=0)
        # distance_positive = 1 - cos(anchor, positive)
        # distance_negative = 1 - cos(anchor, negative)

        # Euclidean distance
        # distance_positive = np.linalg.norm(anchor-positive)
        # distance_negative = np.linalg.norm(anchor-negative)
        total_loss = []
        for a, p, n in zip(anchor, positive, negative):
            # distance_positive = torch.cdist(anchor.unsqueeze(0), positive.unsqueeze(0), p=2)
            # distance_negative = torch.cdist(anchor.unsqueeze(0), negative.unsqueeze(0), p=2)

            distance_positive = torch.cdist(a.unsqueeze(0), p.unsqueeze(0), p=2)
            distance_negative = torch.cdist(a.unsqueeze(0), n.unsqueeze(0), p=2)

            # losses = F.relu(distance_negative - distance_positive + self.margin)
            # print(distance_positive.item(), distance_negative.item())
            # original
            # distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
            # distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
            # losses = F.relu(distance_positive - distance_negative + self.margin)

            losses = torch.log(1 + torch.exp(distance_positive.squeeze() - distance_negative.squeeze()))
            total_loss.append(losses)
        
        return torch.stack(total_loss).mean()