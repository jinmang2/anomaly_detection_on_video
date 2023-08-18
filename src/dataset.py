import os
import json
import zipfile
from PIL import Image
from typing import Union, List, Optional, Callable, Dict, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import decord

from huggingface_hub import hf_hub_download

from . import gtransforms


DEFAULT_FEATURE_HUB = "jinmang2/ucf_crime_tencrop_i3d_seg32"
DEFAULT_FILENAMES = {"train": "train.zip", "test": "test.zip"}


def _build_feature_dataset(
    filepath: str, mode: str, dynamic_load: bool
) -> Union[Dataset, Dict[str, Dataset]]:
    assert mode in ("train", "test")

    zipf = zipfile.ZipFile(filepath)

    filenames = []
    values = {}
    for member in zipf.infolist():
        filename = member.filename.split("/")[-1]
        filenames.append(filename)
        value = np.load(zipf.open(member)) if not dynamic_load else member
        values[filename] = value

    if mode == "test":
        gt_path = hf_hub_download(
            repo_id=DEFAULT_FEATURE_HUB,
            filename="ground_truth.json",
            repo_type="dataset",
            force_download=True,
        )
        gt = json.load(open(gt_path))
        return FeatureDataset(
            filenames=filenames,
            values=values,
            labels=gt,
            open_func=zipf.open if dynamic_load else None,
        )

    normal_filenames = [fname for fname in filenames if "Normal" in fname]
    normal_kwargs = {
        "filenames": normal_filenames,
        "values": {fname: values[fname] for fname in normal_filenames},
        "open_func": zipf.open if dynamic_load else None,
    }
    abnormal_filenames = [fname for fname in filenames if "Normal" not in fname]
    abnormal_kwargs = {
        "filenames": abnormal_filenames,
        "values": {fname: values[fname] for fname in abnormal_filenames},
        "open_func": zipf.open if dynamic_load else None,
    }

    return {
        "normal": FeatureDataset(**normal_kwargs),
        "abnormal": FeatureDataset(**abnormal_kwargs),
    }


def build_feature_dataset(
    mode: str = "train",
    local_path: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    dynamic_load: bool = True,
) -> Union[Dataset, Dict[str, Dataset]]:
    assert mode in ("train", "test")
    assert sum([local_path is None, filename is None]) != 1

    if local_path is None:  # filename is also None
        filepath = hf_hub_download(
            repo_id=DEFAULT_FEATURE_HUB,
            filename=DEFAULT_FILENAMES[mode],
            cache_dir=cache_dir,
            revision=revision,
            repo_type="dataset",
        )
    else:  # local_path and filename aren't None
        filepath = os.path.join(local_path, filename)

    return _build_feature_dataset(filepath, mode, dynamic_load)


class FeatureDataset(Dataset):
    def __init__(
        self,
        filenames: List[str],
        values: Dict[str, Union[zipfile.ZipInfo, np.ndarray]],
        labels: Optional[Dict[str, float]] = None,
        open_func: Optional[Callable] = None,
    ):
        self.filenames = filenames
        self.values = values
        self.labels = labels

        self.open_func = open_func

    def __len__(self) -> int:
        return len(self.values)

    def open(self, value: Union[zipfile.ZipInfo, np.ndarray]) -> np.ndarray:
        if self.open_func is None:
            return value
        # dynamic loading
        return np.load(self.open_func(value))

    def add_magnitude(self, feature: np.ndarray) -> np.ndarray:
        magnitude = np.linalg.norm(feature, axis=2)[:, :, np.newaxis]
        feature = np.concatenate((feature, magnitude), axis=2)
        return feature

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        fname = self.get_filename(idx)
        feature = self.open(self.values[fname])
        anomaly = 0.0 if "Normal" in fname else 1.0
        outputs = {
            "feature": self.add_magnitude(feature),
            "anomaly": np.array(anomaly, dtype=np.float32),
        }

        if self.labels is not None:
            label = np.array(self.labels[fname], dtype=np.float32)
            outputs.update({"label": label})

        return outputs

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
