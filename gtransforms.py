from PIL import Image
from typing import Union, List, Tuple

import torch
from torchvision import transforms


class GroupResize:
    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = 256,
        resample: int = Image.BILINEAR,
    ):
        self.worker = transforms.Resize(size, interpolation=resample)

    def __call__(self, img_group: List["PIL.Image"]) -> List["PIL.Image"]:
        return [self.worker(img) for img in img_group]


class GroupTenCrop:
    def __init__(self, size: int = 224):
        tencrop = transforms.TenCrop(size)
        pil_to_tensor = transforms.Lambda(
            lambda crops: torch.stack(
                [transforms.PILToTensor()(crop) for crop in crops]
            )
        )
        self.worker = transforms.Compose([tencrop, pil_to_tensor])

    def __call__(self, img_group: List["PIL.Image"]) -> List[torch.Tensor]:
        return [self.worker(img) for img in img_group]


class ToTensorTenCrop:
    def __call__(self, img_group: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(img_group, dim=0).float()


class GroupNormalizeTenCrop:
    def __init__(
        self,
        mean: List[float] = [114.75, 114.75, 114.75],
        std: List[float] = [57.375, 57.375, 57.375],
    ):
        # default mean, str:
        # Single channel mean/stev on kinetics (unlike pytorch Imagenet)
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor.shape == (frames_per_clip, ncrops, n_channel, heights, widths)
        # frames_per_clip
        for b in range(tensor.size(0)):
            # ncrops
            for c in range(tensor.size(1)):
                for t, m, s in zip(tensor[b][c], self.mean, self.std):
                    t.sub_(m).div_(s)
        return tensor


class LoopPad:
    def __init__(self, max_len: int = 16):
        self.max_len = max_len

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        length = tensor.size(0)

        if length != self.max_len:
            # repeat the clip as many times as is necessary
            n_pad = self.max_len - length
            pad = [tensor] * (n_pad // length)

            if n_pad % length > 0:
                pad += [tensor[0 : n_pad % length]]

            tensor = torch.cat([tensor] + pad, dim=0)

        return tensor