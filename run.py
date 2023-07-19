import os
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve

import torch
from torch.utils.data import DataLoader

from src.dataset import build_feature_dataset
from src.models import MGFNForVideoAnomalyDetection, MGFNConfig


def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write(f"epoch: {test_info['epoch'][-1]}\n")
    fo.write(f" AUC = {test_info['test_AUC'][-1]}\n")
    fo.write(f" PR = {test_info['test_PR'][-1]}\n")
    fo.close()


batch_size = 8
max_epochs = 1000
cache_dir = "/content/drive/MyDrive/ucf_crime/i3d_seg/"
checkpoint_path = "/content/drive/MyDrive/ucf_crime/0713"

train_dataset = build_feature_dataset(mode="train", cache_dir=cache_dir)
normal_dataset = train_dataset["normal"]
abnormal_dataset = train_dataset["abnormal"]
test_dataset = build_feature_dataset(mode="test", cache_dir=cache_dir)

train_nloader = DataLoader(
    normal_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)
train_aloader = DataLoader(
    abnormal_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = MGFNForVideoAnomalyDetection(MGFNConfig())

if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0005,
)
device = next(model.parameters()).device

test_info = {"epoch": [], "test_AUC": [], "test_PR": []}
best_AUC = -1
for epoch in tqdm(range(max_epochs), desc="Epochs"):
    with torch.set_grad_enabled(True):
        model.train()

        for step, (ninputs, ainputs) in enumerate(
            tqdm(
                zip(train_nloader, train_aloader),
                total=min(len(train_nloader), len(train_aloader)),
                desc="Iteration",
                leave=False,
            )
        ):
            inputs = torch.cat((ninputs["feature"], ainputs["feature"]), 0).to(device)
            nlabels = ninputs["anomaly"].float().to(device)
            alabels = ainputs["anomaly"].float().to(device)
            outputs = model(
                video=inputs, abnormal_labels=alabels, normal_labels=nlabels
            )

            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()

    with torch.no_grad():
        model.eval()
        all_preds = []
        all_labels = []
        for inputs in tqdm(test_loader, desc="Evaluation", leave=False):
            scores = model(inputs["feature"].to(device)).scores
            preds = scores.squeeze(0).squeeze(-1)
            all_preds += [preds.cpu().detach().numpy()]
            all_labels += list(map(lambda x: x.item(), inputs["label"]))
        all_preds = np.concatenate(all_preds)
        all_preds = np.repeat(all_preds, 16)

        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        rec_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)

    test_info["epoch"] += [epoch]
    test_info["test_AUC"] += [rec_auc]
    test_info["test_PR"] += [pr_auc]

    print(f"Epoch: {epoch} AUC={rec_auc} PR={pr_auc}")
    if test_info["test_AUC"][-1] > best_AUC:
        best_AUC = test_info["test_AUC"][-1]
        savepath = os.path.join(checkpoint_path, f"Epoch{epoch:04}")
        model.save_pretrained(savepath)
        save_best_record(test_info, os.path.join(savepath, "history.txt"))
