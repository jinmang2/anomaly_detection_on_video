import torch
from torch import nn

from . import ContrastiveLoss


class MGFNLoss(nn.Module):
    def __init__(self, alpha: float = 0.001):
        super().__init__()
        self.alpha = alpha
        self.criterion = nn.BCELoss()
        self.contrastive = ContrastiveLoss()

    def forward(
        self,
        abnormal_scores: torch.Tensor,
        normal_scores: torch.Tensor,
        a_feat_magnitude: torch.Tensor,
        n_feat_magnitude: torch.Tensor,
        abnormal_labels: torch.Tensor,
        normal_labels: torch.Tensor,
    ) -> torch.Tensor:
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
