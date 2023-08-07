import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve

import torch
from torch.utils.data import DataLoader

from src.dataset import build_feature_dataset
from src.models import MGFNForVideoAnomalyDetection, MGFNConfig


def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write(f"epoch: {test_info['epoch'][-1]}\n")
    fo.write(f" AvgLoss = {test_info['AvgLoss'][-1]}\n")
    fo.write(f" StdLoss = {test_info['StdLoss'][-1]}\n")
    fo.write(f" AUC = {test_info['test_AUC'][-1]}\n")
    fo.write(f"  PR = {test_info['test_PR'][-1]}\n")
    fo.close()


batch_size = 16
max_epochs = 1000
revision = "tushar-n"
checkpoint_path = "/content/drive/MyDrive/ucf_crime/0807"
cache_dir = "/content/drive/MyDrive/ucf_crime/cache/"

# @TODO: alignment
train_dataset = build_feature_dataset(
    mode="train", revision=revision, cache_dir=cache_dir
)
normal_dataset, abnormal_dataset = train_dataset["normal"], train_dataset["abnormal"]
test_dataset = build_feature_dataset(
    mode="test", revision=revision, cache_dir=cache_dir
)

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

test_info = {
    "epoch": [],
    "test_AUC": [],
    "test_PR": [],
    "AvgLoss": [],
    "StdLoss": [],
}

best_AUC = -1
pred_host = []
label_host = []
loss_host = []

for epoch in tqdm(range(max_epochs), desc="Epochs"):
    with torch.set_grad_enabled(True):
        model.train()
        losses = []

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
            losses.append(outputs.loss.detach().cpu().item())
            optimizer.step()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        model.eval()
        all_preds = []
        all_labels = []
        for inputs in tqdm(test_loader, desc="Evaluation", leave=False):
            features = inputs["feature"].permute(0, 2, 1, 3)
            scores = model(features.to(device)).scores
            preds = scores.squeeze(0).squeeze(-1)
            all_preds += [preds.cpu().detach().numpy()]
            all_labels += list(map(lambda x: x.item(), inputs["label"]))
        all_preds = np.concatenate(all_preds)
        all_preds = np.repeat(all_preds, 16)

        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        rec_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)

    pred_host += [all_preds]
    label_host += [all_labels]

    test_info["epoch"] += [step]
    test_info["test_AUC"] += [rec_auc]
    test_info["test_PR"] += [pr_auc]
    avg_loss = np.mean(losses)
    test_info["AvgLoss"] += [avg_loss]
    std_loss = np.std(losses)
    test_info["StdLoss"] += [std_loss]

    print(
        f"Epoch:{epoch} AUC={rec_auc} PR={pr_auc} "
        f"\n\t1stLoss={losses[0]:.6f} lastLoss={losses[-1]:.6f} minLoss={np.min(losses):.6f} "
        f"\n\tAvgLoss={avg_loss:.6f} StdLoss={std_loss:.6f} maxLoss={np.max(losses):.6f}"
    )
    fig = plt.figure(figsize=(16, 4))
    fig.set_facecolor("white")
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(all_preds)
    ax1.plot(np.array(all_labels), alpha=0.5)
    ax2.plot(fpr, tpr)
    plt.show()

    if test_info["test_AUC"][-1] > best_AUC:
        best_AUC = test_info["test_AUC"][-1]
        savepath = os.path.join(checkpoint_path, f"Epoch{epoch:04}")
        model.save_pretrained(savepath)
        save_best_record(test_info, os.path.join(savepath, "history.txt"))
