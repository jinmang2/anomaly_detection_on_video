import os
import datasets
from datasets import load_dataset

import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

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
    train_path = os.path.join(outpath, "train")
    test_path = os.path.join(outpath, "test")

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    extract(anomaly, model, device, train_path, mode="train")
    extract(anomaly, model, device, test_path, mode="test")


def extract(
    dataset: datasets.DatasetDict,
    model: torch.nn.Module,
    device: torch.device,
    outpath: str,
    mode: str = "train",
):
    for sample in tqdm(dataset[mode]):
        # check existence
        filename = sample["video_path"].split(os.sep)[-1].split(".")[0]
        filename += "_i3d.npy"
        savepath = os.path.join(outpath, filename)

        if os.path.exists(savepath):
            continue

        # read video frames
        video_dataset = TenCropVideoFrameDataset(sample["video_path"])
        dataloader = DataLoader(video_dataset, batch_size=16, shuffle=False)

        # inference
        outputs = []
        for _, inputs in enumerate(dataloader):
            # Unlike Tushar-N's B which is `n_videos`, our B is `n_clips`.
            # (B, 10, 16, 3, 224, 224) -> (B, 10, 3, 16, 224, 224)
            inputs = inputs.permute(0, 1, 3, 2, 4, 5)
            crops = []
            for crop_idx in range(inputs.shape[1]):
                crop = inputs[:, crop_idx]
                crop = crop.to(device)
                crop = model(crop)
                crops.append(crop.detach().cpu().numpy())
            outputs.append(crops)

        # concat
        _outputs = []
        for output in outputs:
            outs = []
            for out in output:
                outs.append(out)
            _outputs.append(np.stack(outs, axis=1))
        _outputs = np.vstack(_outputs)
        outputs = np.squeeze(_outputs)  # (n_clips, 10, 2048)

        # save
        np.save(savepath, outputs)


if __name__ == "__main__":
    main()
