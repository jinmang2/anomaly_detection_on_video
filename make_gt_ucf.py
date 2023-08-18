import os
import json
import zipfile
import numpy as np
from tqdm import tqdm
from datasets import DownloadManager, DownloadConfig
from huggingface_hub import hf_hub_download, hf_hub_url


temporal_annots_path = hf_hub_download(
    repo_id="jinmang2/ucf_crime",
    filename="Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
    subfolder="UCF_Crimes-Train-Test-Split",
    repo_type="dataset",
    force_download=True,
)
with open(temporal_annots_path, "r") as f:
    temporal_annots = {}
    for line in f.readlines():
        filename, _, s1, e1, s2, e2 = line.strip().split("  ")
        s1, e1, s2, e2 = map(lambda x: int(x), (s1, e1, s2, e2))
        temporal_annots[filename.split(".")[0]] = {
            "first_event": (s1, e1),
            "second_event": (s2, e2),
        }

repo_id = "jinmang2/ucf_crime_tencrop_i3d_seg32"
dl_config = DownloadConfig()
dl_manager = DownloadManager(record_checksums=False, download_config=dl_config)
archive = dl_manager.download(hf_hub_url(repo_id, "test.zip", repo_type="dataset"))
zipf = zipfile.ZipFile(archive)

ground_truths = {}
for member in tqdm(zipf.infolist()):
    features = np.load(zipf.open(member))
    num_frame = features.shape[0] * 16

    file = member.filename.split("/")[-1].replace("_i3d.npy", "")
    annots_idx = temporal_annots[file]
    first_event = annots_idx["first_event"]
    second_event = annots_idx["second_event"]
    gt = [0.0] * num_frame

    if first_event[0] > 0 and first_event[0] > 0:
        for i in range(first_event[0], min(first_event[1] + 1, num_frame)):
            gt[i] = 1.0

    if second_event[0] > 0 and second_event[1] > 0:
        for i in range(second_event[0], min(second_event[1] + 1, num_frame)):
            gt[i] = 1.0

    ground_truths[file] = gt

savefolder = "/content/drive/MyDrive/ucf_crime/ground_truth"
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

savepath = os.path.join(savefolder, "ground_truth_ucf_crime.json")
with open(savepath, "w") as f:
    json.dump(ground_truths, f)
