import os
import wget
from typing import Callable, Tuple
import torch
from torch import nn
from pytorchvideo.models.resnet import create_resnet


model_zoo = {
    "i3d_8x8_r50": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D\_8x8\_R50.pyth",
}


class ResNetHead(nn.Module):
    def __init__(self, pool: nn.Module = None, output_pool: nn.Module = None):
        super().__init__()
        self.pool = pool
        self.output_pool = output_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = self.pool(x)
        if self.output_pool is not None:
            x = self.output_pool(x)
        return x


def create_res_pooler(direct_pool: bool = False):

    def _create_res_pool(
        *,
        pool: Callable = nn.AvgPool3d,
        pool_kernel_size: Tuple[int] = (1, 7, 7),
        pool_stride: Tuple[int] = (1, 1, 1),
        pool_padding: Tuple[int] = (0, 0, 0),
        **kwargs,
    ) -> nn.Module:
        if direct_pool:
            return nn.AdaptiveAvgPool3d((1, 1, 1))

        return ResNetHead(
            pool=pool(
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding,
            ),
            # output_with_global_average
            output_pool=nn.AdaptiveAvgPool3d(1)
        )

    return _create_res_pool


def build_i3d_feature_extractor(model_path: str = "~/content/", check_model_size: bool = True):
    model = create_resnet(
        stem_conv_kernel_size=(5, 7, 7),
        stage1_pool=nn.MaxPool3d,
        stage_conv_a_kernel_size=(
            (3, 1, 1),
            [(3, 1, 1), (1, 1, 1)],
            [(3, 1, 1), (1, 1, 1)],
            [(1, 1, 1), (3, 1, 1)],
        ),
        head=create_res_pooler(direct_pool=False),
    )

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(os.path.join(model_path, "I3D_8x8_R50.pyth")):
        wget.download(model_zoo["i3d_8x8_r50"], out=model_path)

    checkpoint = torch.load(os.path.join(model_path, "I3D_8x8_R50.pyth"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=False)

    if check_model_size:
        # https://discuss.pytorch.org/t/pytorch-model-size-in-mbs/149002
        size_model = 0
        for param in model.parameters():
            if param.data.is_floating_point():
                size_model += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.data.dtype).gits
        print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

    return model