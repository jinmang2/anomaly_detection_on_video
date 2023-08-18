from abc import abstractmethod
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

    def __call__(self, img_group: List[Image.Image]) -> List[Image.Image]:
        return [self.worker(img) for img in img_group]


class GroupTenCrop:
    def __init__(self, size: int = 224):
        self.worker = transforms.TenCrop(size)

    def __call__(self, img_group: List[Image.Image]) -> List[List[Image.Image]]:
        return [self.worker(img) for img in img_group]


class ToTensorTenCrop:
    def __init__(self):
        pil_to_tensor = transforms.PILToTensor()
        self.worker = transforms.Lambda(
            lambda crops: torch.stack([pil_to_tensor(crop) for crop in crops])
        )

    def __call__(self, img_group: List[List[Image.Image]]) -> torch.Tensor:
        tensor_group = [self.worker(img) for img in img_group]
        return torch.stack(tensor_group, dim=0).float()


class _GroupNormalizeTenCropBase:
    @abstractmethod
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor.shape == (frames_per_clip, ncrops, n_channel, heights, widths)
        # frames_per_clip
        for b in range(tensor.size(0)):
            # ncrops
            for c in range(tensor.size(1)):
                normalized = self.normalize(tensor[b, c])
                tensor[b, c] = normalized
        return tensor


class GroupStandardizationTenCrop(_GroupNormalizeTenCropBase):
    def __init__(
        self,
        mean: Union[float, List[float]] = 114.75,
        std: Union[float, List[float]] = 57.375,
    ):
        if isinstance(mean, float):
            mean = [mean] * 3
        if isinstance(std, float):
            std = [std] * 3
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class GroupPixelMinmaxTenCrop(_GroupNormalizeTenCropBase):
    def __init__(self, min: float = 0.0, max: float = 1.0):
        assert min < max
        self.min = min
        self.max = max

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        min, max = tensor.min(), tensor.max()
        new_min, new_max = self.min, self.max
        tensor = (tensor - min) / (max - min)
        return tensor * (new_max - new_min) + new_min


class GroupRGBChannelMinmaxTenCrop(_GroupNormalizeTenCropBase):
    def __init__(
        self,
        min: Union[float, List[float]] = 0.0,
        max: Union[float, List[float]] = 1.0,
    ):
        if isinstance(min, float):
            min = [min] * 3
        if isinstance(max, float):
            max = [max] * 3

        assert any(map(lambda x, y: x < y, min, max))

        self.min = torch.FloatTensor(min).unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.max = torch.FloatTensor(max).unsqueeze(dim=-1).unsqueeze(dim=-1)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        min = tensor.min(dim=-1)[0].min(dim=-1)[0]
        min = min.unsqueeze(dim=-1).unsqueeze(dim=-1)
        max = tensor.max(dim=-1)[0].max(dim=-1)[0]
        max = max.unsqueeze(dim=-1).unsqueeze(dim=-1)
        new_min, new_max = self.min, self.max
        tensor = (tensor - min) / (max - min)
        return tensor * (new_max - new_min) + new_min


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
