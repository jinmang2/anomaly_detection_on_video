import os
import zipfile
from PIL import Image
from typing import Union, List, Optional, Callable, Dict

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import decord

from huggingface_hub import hf_hub_url
from datasets import DownloadManager, DownloadConfig

from . import gtransforms


DEFAULT_FEATURE_HUB = "jinmang2/ucf_crime_tencrop_i3d_seg32"
DEFAULT_FILENAMES = {"train": "train.zip", "test": "test.zip"}


def _build_feature_dataset(
    filepath: str,
    mode: str,
    cache_dir: str
) -> Union[Dataset, Dict[str, Dataset]]:
    assert mode in ("train", "test")

    dl_config = DownloadConfig(cache_dir=cache_dir)
    dl_manager = DownloadManager(download_config=dl_config)
    archive = dl_manager.download(filepath)
    zipf = zipfile.ZipFile(archive)

    filenames = []
    zipinfos = []
    labels = []
    for member in zipf.infolist():
        filenames.append(member.filename.split("/")[-1])
        zipinfos.append(member)
        label = 0 if "Normal" in member.filename else 1
        labels.append(label)

    if mode == "test":
        return FeatureDataset(
            filenames=filenames,
            zipinfos=zipinfos,
            labels=labels,
            open_func=zipf.open,
        )

    return {
        "normal": FeatureDataset(
            filenames=[name for i, name in enumerate(filenames) if labels[i] == 0],
            zipinfos=[info for i, info in enumerate(zipinfos) if labels[i] == 0],
            labels=[i for i in labels if i == 0],
            open_func=zipf.open,
        ),
        "abnormal": FeatureDataset(
            filenames=[name for i, name in enumerate(filenames) if labels[i] == 1],
            zipinfos=[info for i, info in enumerate(zipinfos) if labels[i] == 1],
            labels=[i for i in labels if i == 1],
            open_func=zipf.open,
        ),
    }


def get_hf_hub_url(filename: str) -> str:
    return hf_hub_url(DEFAULT_FEATURE_HUB, filename, repo_type="dataset")


def build_feature_dataset(
    mode: str = "train",
    local_path: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Union[Dataset, Dict[str, Dataset]]:
    assert mode in ("train", "test")
    if local_path is None:
        if filename is not None:
            raise ValueError
        filepath = get_hf_hub_url(DEFAULT_FILENAMES[mode])
    else:
        if filename is None:
            raise ValueError
        filepath = os.path.join(local_path, filename)

    return _build_feature_dataset(filepath, mode, cache_dir)


class FeatureDataset(Dataset):
    def __init__(
        self,
        filenames: List[str],
        zipinfos: List[zipfile.ZipInfo],
        labels: List[int],
        open_func: Callable,
    ):
        assert len(filenames) == len(zipinfos)
        assert len(zipinfos) == len(labels)

        self.filenames = filenames
        self.zipinfos = zipinfos
        self.labels = labels

        self.open = open_func

    def __len__(self) -> int:
        return len(self.zipinfos)

    def __getitem__(self, idx: int) -> torch.Tensor:
        zipinfo = self.zipinfos[idx]
        feature = np.load(self.open(zipinfo))
        label = self.labels[idx]
        return feature, label
    
    def get_filename(self, idx: int) -> str:
        return self.filenames[idx]


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
