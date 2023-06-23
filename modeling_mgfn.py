import torch
from torch import nn
from torch import nn, einsum
from einops import rearrange

import transformers
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from configuration_mgfn import MGFNConfig


class MGFNLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class MGFNFeedForward(nn.Module):
    def __init__(self, dim, repe=4, dropout=0.0):
        super().__init__()
        self.layer_norm = MGFNLayerNorm(dim)
        self.in_conv = nn.Conv1d(dim, dim * repe, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_conv = nn.Conv1d(dim * repe, dim, 1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.in_conv(x)
        x = self.gelu(x)
        x = self.dropout(x)
        out = self.out_conv(x)
        return out


class MGFNFeatureAmplifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        init_dim = config.dims[0]
        self.mag_ratio = config.mag_ratio
        self.to_tokens = nn.Conv1d(
            config.channels,
            init_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # bs: batch_size
        # ncrops: number of crops in each video clip,
        #        `P` in paper. (Sultani, Chen, and Shah 2018)
        # t: clips
        # c: feature dimension, `C` in paper.
        bs, ncrops, t, c = x.size()
        x = x.view(bs * ncrops, t, c).permute(0, 2, 1)
        x_f, x_m = x[:, :2048, :], x[:, 2048:, :]
        x_f = self.to_tokens(x_f)
        x_m = self.to_mag(x_m)  # eq (1)
        x_f = x_f + self.mag_ratio * x_m  # eq (2)
        return x_f


class GlanceAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads
        self.norm = MGFNLayerNorm(dim)
        # construct a video clip-level transformer (VCT)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        shape, h = x.shape, self.heads
        x = rearrange(x, "b c ... -> b c (...)")
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h d) n -> b h n d", h=h), (q, k, v))
        q = q * self.scale
        # correlate the different temporal clips
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        # softmax normalization
        attn = sim.softmax(dim=-1)
        # weighted average of all clips in the long video containing
        # both normal and abnormal (if exists) ones
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b (h d) n", h=h)
        out = self.to_out(out)
        return out.view(*shape)


class GlanceBlock(nn.Module):
    """MGFN component to learn the global correlatoin among clips."""

    def __init__(self, config, dim, heads):
        super().__init__()
        # shortcut convolution
        self.scc = nn.Conv1d(dim, dim, 3, padding=1)
        # video clip-level transformer
        self.attention = GlanceAttention(
            dim=dim,
            heads=heads,
            dim_head=config.dim_head,
        )
        # additional feed-forward network
        # To further improve the model's representation capability.
        self.ffn = MGFNFeedForward(dim, repe=config.ff_repe, dropout=config.dropout)

    def forward(self, x):
        x = self.scc(x) + x
        x = self.attention(x) + x
        x = self.ffn(x) + x
        return x


class FocusAttention(nn.Module):
    """
    MHRAs (multi-head relation aggregators):
          Like self-attention, SAC allows each channel to get access
        to the nearby channels to learn the channel-wise correlation
        without any learnable weights
    """

    def __init__(self, dim, heads, dim_head, local_aggr_kernel):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.BatchNorm1d(dim)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, bias=False)
        self.rel_pos = nn.Conv1d(
            heads,
            heads,
            local_aggr_kernel,
            padding=local_aggr_kernel // 2,
            groups=heads,
        )
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)  # (b*crop,c,t)
        b, c, *_, h = *x.shape, self.heads
        v = self.to_v(x)  # (b*crop,c,t)
        v = rearrange(v, "b (c h) ... -> (b c) h ...", h=h)  # (b*ten*64,c/64,32)
        out = self.rel_pos(v)
        out = rearrange(out, "(b c) h ... -> b (c h) ...", b=b)
        return self.to_out(out)


class FocusBlock(nn.Module):
    """MGFN component to enhance the feature learning in each video clip."""

    def __init__(self, config, dim, heads):
        super().__init__()
        # shortcut convolution
        self.scc = nn.Conv1d(dim, dim, 3, padding=1)
        # self-attentional convolution (SAC)
        self.attention = FocusAttention(
            dim=dim,
            heads=heads,
            dim_head=config.dim_head,
            local_aggr_kernel=config.local_aggr_kernel,
        )
        # additional feed-forward network
        # To further improve the model's representation capability.
        self.ffn = MGFNFeedForward(dim, repe=config.ff_repe, dropout=config.dropout)

    def forward(self, x):
        x = self.scc(x) + x
        x = self.attention(x) + x
        x = self.ffn(x) + x
        return x


class MGFNIntermediate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer_norm = MGFNLayerNorm(in_dim)
        self.conv = nn.Conv1d(in_dim, out_dim, 1, stride=1)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.conv(x)


class MGFNPreTrainedModel(PreTrainedModel):
    config_class = MGFNConfig
    base_model_prefix = "backbone"

    def _init_weights(self, module):
        return NotImplementedError

    @property
    def dummy_inputs(self):
        # @TODO: fixed tensor
        # (batch_size, n_crops, n_clips, feature_dim)
        # ncrops == 10
        return torch.randn(32, 10, 32, 2049)


class MGFNModel(MGFNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        depths = config.depths
        mgfn_types = config.mgfn_types

        self.amplifier = MGFNFeatureAmplifier(config)

        blocks = []
        for ind, (depth, mgfn_type) in enumerate(zip(depths, mgfn_types)):
            stage_dim = config.dims[ind]
            heads = stage_dim // config.dim_head

            if mgfn_type == "gb":
                block_cls = GlanceBlock
            elif mgfn_type == "fb":
                block_cls = FocusBlock
            else:
                raise AttributeError(
                    "The type of mgfn block must be either `gb` or `fb`."
                )

            block = block_cls(config, dim=stage_dim, heads=heads)
            blocks.append(block)

            if ind != len(depths) - 1:
                intermediate = MGFNIntermediate(stage_dim, config.dims[ind + 1])
                blocks.append(intermediate)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.amplifier(x)
        out = self.blocks(x)
        return out


class MGFNForVideoAnomalyDedection(MGFNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.k = config.k
        self.batch_size = config.batch_size
        last_dim = config.dims[-1]

        self.backbone = MGFNModel(config)

        self.layer_norm = nn.LayerNorm(last_dim)
        self.fc = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout_rate)

    def magnitude_selection_and_score_prediction(
        self, features, scores, bs, ncrops,
    ):
        # @TODO: batch inference, abn/nor split, code 중복 해소
        device = features.device
        bc, t, f = features.size()

        scores = scores.view(bs, ncrops, -1).mean(dim=1)
        scores = scores.unsqueeze(dim=2)

        normal_features = features[0:self.batch_size * 10]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size * 10:]
        abnormal_scores = scores[self.batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(dim=1)
        n_feat_magnitudes = feat_magnitudes[0:self.batch_size]
        a_feat_magnitudes = feat_magnitudes[self.batch_size:]
        n_size = n_feat_magnitudes.shape[0]

        if n_feat_magnitudes.shape[0] == 1:  # this is for inference
            a_feat_magnitudes = n_feat_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(n_feat_magnitudes, device=device)
        select_idx = self.dropout(select_idx)

        a_feat_magnitudes_drop = a_feat_magnitudes * select_idx
        idx_abn = torch.topk(a_feat_magnitudes_drop, self.k, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(dim=2).expand([-1, -1, abnormal_features.shape[2]])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)

        total_select_abn_feature = torch.zeros(0, device=device)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(
                abnormal_feature, 1, idx_abn_feat,
            )
            total_select_abn_feature = torch.cat(
                [total_select_abn_feature, feat_select_abn]
            )
        idx_abn_score = idx_abn.unsqueeze(dim=2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(
            torch.gather(abnormal_scores, 1, idx_abn_score),
            dim=1,
        )

        select_idx_normal = torch.ones_like(n_feat_magnitudes, device=device)
        select_idx_normal = self.dropout(select_idx_normal)
        
        n_feat_magnitudes_drop = n_feat_magnitudes * select_idx_normal
        idx_normal = torch.topk(n_feat_magnitudes_drop, self.k, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(dim=2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0, device=device)
        for normal_feature in normal_features:
            feat_select_normal = torch.gather(
                normal_feature, 1, idx_normal_feat,
            )
            total_select_nor_feature = torch.cat(
                [total_select_nor_feature, feat_select_normal]
            )
        idx_normal_score = idx_normal.unsqueeze(dim=2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(
            torch.gather(normal_scores, 1, idx_normal_score),
            dim=1,
        )

        abn_feamagnitude = total_select_abn_feature
        nor_feamagnitude = total_select_nor_feature

        return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores

    def forward(self, video):
        # @TODO: calc_loss, output format, msnsd method
        bs, ncrops = video.size()[:2]
        x_f = self.backbone(video).permute(0, 2, 1)
        x = self.layer_norm(x_f)
        score = self.sigmoid(self.fc(x))
        
        (
            score_abnormal,
            score_normal,
            abn_feamagnitude,
            nor_feamagnitude,
            scores,
        ) = self.magnitude_selection_and_score_prediction(
            x, score, bs, self.batch_size, ncrops,
        )

        return (
            score_abnormal,
            score_normal,
            abn_feamagnitude,
            nor_feamagnitude,
            scores,
        )