import os
from tqdm import tqdm
from typing import Union

import datasets
from datasets import load_dataset

import numpy as np

import decord
from PIL import Image

import torch
from torch.utils.data import DataLoader

from src.i3d import build_i3d_feature_extractor
from src.dataset import TenCropVideoFrameDataset


def load_ucf_crime_dataset(
    repo_id: str = "jinmang2/ucf_crime",
    cache_dir: str = "/content/drive/MyDrive/ucf_crime",
) -> datasets.DatasetDict:
    return load_dataset(repo_id, "anomaly", cache_dir=cache_dir)


def load_feature_extraction_model(model_name: str = "i3d_8x8_r50") -> torch.nn.Module:
    model = build_i3d_feature_extractor(model_name=model_name)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print(next(model.parameters()).device)
    return model


def main(outdir: str = "/content/drive/MyDrive/ucf_crime"):
    anomaly = load_ucf_crime_dataset()
    model = load_feature_extraction_model()
    device = next(model.parameters()).device
    outpath = os.path.join(outdir, "anomaly_features")

    extract(anomaly, model, device, outpath)


def extract(
    dataset: Union[datasets.Dataset, datasets.DatasetDict],
    model: torch.nn.Module,
    device: torch.device,
    outpath: str,
):
    if isinstance(dataset, datasets.DatasetDict):
        for mode, dset in dataset.items():
            new_outpath = os.path.join(outpath, mode)

            extract(dset, model, device, new_outpath)

        return None

    assert isinstance(dataset, datasets.Dataset), (
        "The type of dataset argument must be `datasets.Dataset` or"
        f"`datasets.DatasetDict`. Your input's type is {type(dataset)}."
    )

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    def _extract(video_dataset: torch.utils.data.Dataset) -> torch.Tensor:
        outputs = []
        dataloader = DataLoader(video_dataset, batch_size=16, shuffle=False)
        for _, inputs in enumerate(dataloader):
            # Unlike Tushar-N's B which is `n_videos`, our B is `n_clips`.
            # (B, 10, 16, 3, 224, 224) -> (B, 10, 3, 16, 224, 224)
            inputs = inputs.permute(0, 1, 3, 2, 4, 5)
            crops = []
            for crop_idx in range(inputs.shape[1]):
                crop = inputs[:, crop_idx]
                crop = crop.to(device)
                # (B, 3, 16, 224, 224) -> (B, 2048, 1, 1, 1)
                crop = model(crop)
                crops.append(crop.detach().cpu().numpy())
            outputs.append(crops)

        # stack
        _outputs = []
        for output in outputs:
            # [(B, 2048, 1, 1, 1)] * 10 -> (B, 10, 2048, 1, 1, 1)
            _outputs.append(np.stack(output, axis=1))
        # [(B, 10, 2048, 1, 1, 1)] * T -> (n_clips, 10, 2048, 1, 1, 1)
        # T = n_clips / B
        _outputs = np.vstack(_outputs)
        outputs = np.squeeze(_outputs)  # (n_clips, 10, 2048)

        return outputs

    for sample in tqdm(dataset):
        # check existence
        filename = sample["video_path"].split(os.sep)[-1].split(".")[0]
        savepath = os.path.join(outpath, filename + "_i3d.npy")

        if os.path.exists(savepath):
            continue

        # If the size of the video is larger than 1GB, divide it by the
        # segment length. After that, upload it to RAM and receive the
        # tencrop result from the model.
        if sample["size"] > 1024**2:
            # The fps of the video in `ucf_crime` dataset is 30. Therefore, 3,000 video frames
            # are about 100 seconds long, which is a good video length for inference in colab pro+.
            # Among the transforms of `TenCropVideoFrameDataset`, `LoopPad` forcibly pads clips lower than
            # `frames_per_clip`(default: 16), so segment_length is designated as a multiple of 16.
            seg_len = 16 * 188  # 3,008
            # read video frames
            vr = decord.VideoReader(uri=sample["video_path"])
            segments = []
            for seg in tqdm(range(len(vr) // seg_len + 1)):
                seg_folder = os.path.join(outpath, filename)

                if not os.path.exists(seg_folder):
                    os.makedirs(seg_folder)

                seg_savepath = os.path.join(seg_folder, filename + f"_{seg}.npy")

                if os.path.exists(seg_savepath):
                    outputs = np.load(seg_savepath)
                else:
                    images = []
                    for i in [seg * seg_len + i for i in range(seg_len)]:
                        if i == len(vr):
                            break
                        arr = vr[i].asnumpy()
                        images.append(Image.fromarray(arr))
                    video_dataset = TenCropVideoFrameDataset(images)
                    # inference
                    outputs = _extract(video_dataset)
                    np.save(seg_savepath, outputs)

                segments.append(outputs)
            outputs = np.vstack(segments)
        else:
            # read video frames
            video_dataset = TenCropVideoFrameDataset(sample["video_path"])
            # inference
            outputs = _extract(video_dataset)

        # save
        np.save(savepath, outputs)


if __name__ == "__main__":
    main()
