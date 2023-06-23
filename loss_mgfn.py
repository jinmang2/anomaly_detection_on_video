from typing import Optional

import torch
from torch import nn


class TemporalSmoothnessLoss(nn.Module):
    def __init__(self, lambda1: float = 8e-4):
        super().__init__()
        self.lambda1 = lambda1

    def forward(self, x, lambda1: Optional[float] = None):
        if lambda1 is None:
            lambda1 = self.lambda1

        x_except_last, x_except_first = x[:, :-1, :], x[:, 1:, :]
        loss = torch.sum((x_except_first - x_except_last) ** 2)
        return lambda1 * loss


class SparsityLoss(nn.Module):
    def __init__(self, lambda2: float = 8e-3):
        super().__init__()
        self.lambda2 = lambda2

    def forward(self, x, lambda2: Optional[float] = None):
        if lambda2 is None:
            lambda2 = self.lambda2

        loss = torch.mean(torch.norm(x, dim=0))
        return lambda2 * loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=200.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


class MGFNLoss(nn.Module):
    def __init__(self, alpha: float = 0.001):
        super().__init__()
        self.alpha = alpha
        self.criterion = nn.BCELoss()
        self.contrastive = ContrastiveLoss()

    def forward(
        self,
        abnormal_scores,
        normal_scores,
        a_feat_magnitude,
        n_feat_magnitude,
        abnormal_labels,
        normal_labels,
    ):
        labels = torch.cat((normal_labels, abnormal_labels), 0)
        scores = torch.cat((normal_scores, abnormal_scores), 0).squeeze()
        seperate = len(n_feat_magnitude) / 2

        loss_cls = self.criterion(scores, labels)
        loss_con = self.contrastive(
            torch.norm(a_feat_magnitude, p=1, dim=2),
            torch.norm(n_feat_magnitude, p=1, dim=2),
            1,
        )  # try tp separate normal and abnormal
        loss_con_n = self.contrastive(
            torch.norm(n_feat_magnitude[int(seperate) :], p=1, dim=2),
            torch.norm(n_feat_magnitude[: int(seperate)], p=1, dim=2),
            0,
        )  # try to cluster the same class
        loss_con_a = self.contrastive(
            torch.norm(a_feat_magnitude[int(seperate) :], p=1, dim=2),
            torch.norm(a_feat_magnitude[: int(seperate)], p=1, dim=2),
            0,
        )

        loss_contrastive = self.alpha * loss_con + loss_con_a + loss_con_n
        loss_total = loss_cls + self.alpha * loss_contrastive

        return loss_total
