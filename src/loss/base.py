from typing import Optional

import torch
from torch import nn


class TemporalSmoothnessLoss(nn.Module):
    def __init__(self, lambda1: float = 8e-4):
        super().__init__()
        self.lambda1 = lambda1

    def forward(self, x: torch.Tensor, lambda1: Optional[float] = None) -> torch.Tensor:
        if lambda1 is None:
            lambda1 = self.lambda1

        x_except_last, x_except_first = x[:, :-1, :], x[:, 1:, :]
        loss = torch.sum((x_except_first - x_except_last) ** 2)
        return lambda1 * loss


class SparsityLoss(nn.Module):
    def __init__(self, lambda2: float = 8e-3):
        super().__init__()
        self.lambda2 = lambda2

    def forward(self, x: torch.Tensor, lambda2: Optional[float] = None) -> torch.Tensor:
        if lambda2 is None:
            lambda2 = self.lambda2

        loss = torch.mean(torch.norm(x, dim=0))
        return lambda2 * loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 200.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        euclidean_distance = torch.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive
