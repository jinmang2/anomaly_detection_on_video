import os
import json
import decord
from datasets import load_dataset
from huggingface_hub import hf_hub_download


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

# decord.VideoReader 때문에 동작이 느림 (대략 6분)
# n_frames 정보만 필요하기 때문에 이를 따로 저장해두면 될 것으로 보임
drive_path = "/content/drive/MyDrive/ucf_crime"
repo_id = "jinmang2/ucf_crime"
anomaly = load_dataset(repo_id, "anomaly", cache_dir=drive_path)

ground_truths = {}
for sample in anomaly["test"]:
    n_frames = len(decord.VideoReader(sample["video_path"]))
    num_frame = ((n_frames - 1) // 16 + 1) * 16  # padded

    file = sample["video_path"].split("/")[-1].replace(".mp4", "")
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


output_path = "/content/drive/MyDrive/ucf_crime/ground_truth"

if not os.path.exists(output_path):
    os.makedirs(output_path)

outpath = os.path.join(output_path, "ground_truth_ucf_crime.json")
with open(outpath, "w") as f:
    json.dump(ground_truths, f)
