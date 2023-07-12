import os
from PIL import Image
from typing import Union, Tuple, List

import decord

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from . import gtransforms


class FeatureDataset(Dataset):
    def __init__(
        self,
        rgb_list,
        modality: str = "RGB",
        is_normal: bool = True,
        transform: Union[torch.nn.Module, "transforms.Compose"] = None,
        test_mode: bool = False,
        is_preprocessed: bool = False,
        seg_length: int = 32,
    ):
        self.modality = modality
        self.is_normal = is_normal
        self.rgb_list_file = rgb_list
        self.transform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.is_preprocessed = is_preprocessed
        self.seg_length = seg_length

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            # only UCF
            if self.is_normal:
                self.list = self.list[810:]  # ucf 810; sht63; xd 9525
            else:
                self.list = self.list[:810]  # ucf 810; sht 63; 9525

    @staticmethod
    def process_feat(feat, length):
        new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)  # UCF(32,2048)
        r = np.linspace(0, len(feat), length + 1, dtype=int)  # (33,)
        for i in range(length):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]
        return new_feat

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = self.get_label(index)  # get video level label 0/1

        # Only UCF
        features = np.load(self.list[index].strip("\n"), allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        name = self.list[index].split("/")[-1].strip("\n")[:-4]

        if self.transform is not None:
            features = self.tranform(features)

        if self.test_mode:
            # only UCF
            mag = np.linalg.norm(features, axis=2)[:, :, np.newaxis]
            features = np.concatenate((features, mag), axis=2)
            return features, name
        else:
            # only UCF
            if self.is_preprocessed:
                return features, label
            features = features.transpose(1, 0, 2)  # [10, T, F]
            divided_features = []

            divided_mag = []
            for feature in features:
                feature = self.process_feat(feature, self.seg_length)  # ucf(32,2048)
                divided_features.append(feature)
                divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
            divided_features = np.array(divided_features, dtype=np.float32)
            divided_mag = np.array(divided_mag, dtype=np.float32)
            divided_features = np.concatenate((divided_features, divided_mag), axis=2)
            return divided_features, label

    def get_label(self, index: int) -> torch.Tensor:
        label = 0.0 if self.is_normal else 1.0
        return torch.tensor(label)

    def __len__(self) -> int:
        return len(self.list)

    def get_num_frames(self) -> int:
        return self.num_frame


class TenCropVideoFrameDataset(Dataset):
    def __init__(
        self,
        video_path_or_images: Union[str, List[Image.Image]],
        frames_per_clip: int = 16,
        resize: int = 256,
        cropsize: int = 224,
        resample: int = Image.BILINEAR,
    ):
        if isinstance(video_path_or_images, str):
            vr = decord.VideoReader(uri=video_path_or_images)
            self.images = []
            for i in range(len(vr)):
                arr = vr[i].asnumpy()
                self.images.append(Image.fromarray(arr))
        elif isinstance(video_path_or_images, list) and isinstance(
            video_path_or_images[0], Image.Image
        ):
            self.images = video_path_or_images
        else:
            raise ValueError(
                "The type of `video_path_or_images` must be either `str` "
                "or `List[PIL.Image.Image]`. "
                f"The type of your input is {type(video_path_or_images)}."
            )

        self.frames_per_clip = frames_per_clip
        n_frames = len(self.images)
        self.indices = list(range((n_frames - 1) // frames_per_clip + 1))

        self.transform = transforms.Compose(
            [
                gtransforms.GroupResize(size=resize, resample=resample),
                gtransforms.GroupTenCrop(size=cropsize),
                gtransforms.ToTensorTenCrop(),
                gtransforms.GroupStandardizationTenCrop(),
                gtransforms.LoopPad(max_len=frames_per_clip),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx = idx * self.frames_per_clip
        end_idx = (idx + 1) * self.frames_per_clip
        image = self.images[start_idx:end_idx]
        # (clip_len, ncrops, n_channel, heights, widths)
        tensor = self.transform(image)
        # (ncrops, clip_len, n_channel, heights, widths)
        return tensor.permute(1, 0, 2, 3, 4)
