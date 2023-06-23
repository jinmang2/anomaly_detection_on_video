"""
@TODO: 클래스화 시켜서 in-out 확인하기
"""
import os

from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import datetime

import option
from config import *

from dataset import Dataset
from utils import save_best_record

args = option.parse_args()
config = Config(args)

normal_dataset = Dataset(args, test_mode=False, is_normal=True)

normal_dataset.list = [
    "data/ucf_crime/" + "/".join(i.strip().split("/")[-2:])
    for i in normal_dataset.list
]
print(len(normal_dataset))
normal_dataset.list = [i for i in normal_dataset.list if os.path.exists(i)]
print(len(normal_dataset))

abnormal_dataset = Dataset(args, test_mode=False, is_normal=False)

abnormal_dataset.list = [
    "data/ucf_crime/" + "/".join(i.strip().split("/")[-2:])
    for i in abnormal_dataset.list
]
print(len(abnormal_dataset))
abnormal_dataset.list = [i for i in abnormal_dataset.list if os.path.exists(i)]
print(len(abnormal_dataset))

train_nloader = DataLoader(
    normal_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=False, drop_last=True,
)
train_aloader = DataLoader(
    abnormal_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=False, drop_last=True,
)

for (ninput, nlabel), (ainput, alabel) in zip(train_nloader, train_aloader):
    break

print(ninput.shape, nlabel.shape, ainput.shape, alabel.shape)

inputs = torch.cat((ninput, ainput), dim=0)  # torch.Size([32, 10, 32, 2049])

from loss_official import sparsity, smooth, mgfn_loss

from mgfn import mgfn

official = mgfn()
state_dict = torch.load("mgfn_ucf.pkl", map_location="cpu")
official.load_state_dict(state_dict)

from modeling_mgfn import MGFNForVideoAnomalyDedection, MGFNConfig
from convert_official_to_hf import convert

model = MGFNForVideoAnomalyDedection(MGFNConfig())
new_state_dict = convert(state_dict)
model.load_state_dict(new_state_dict)

model.eval()
model.force_split = True
official.eval()
print("eval mode")

result = True
for step, (tensor1, tensor2) in enumerate(zip(model(inputs).to_tuple(), official(inputs))):
    if not torch.isclose(tensor1, tensor2).all().item():
        result = False
print(result)  # True

score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = official(inputs)

loss_sparse = sparsity(scores[:args.batch_size,:,:].view(-1), args.batch_size, 8e-3)

loss_smooth = smooth(scores,8e-4)

scores = scores.view(args.batch_size * 32 * 2, -1)
scores = scores.squeeze()

nlabel = nlabel[0:args.batch_size]
alabel = alabel[0:args.batch_size]

loss_criterion = mgfn_loss(0.0001)

loss_mgfn = loss_criterion(score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude)
cost = loss_mgfn + loss_smooth + loss_sparse
print(loss_mgfn, loss_sparse, loss_smooth)

outputs = model(inputs, alabel, nlabel)

print(torch.isclose(cost, outputs.loss).all().item())  # True
