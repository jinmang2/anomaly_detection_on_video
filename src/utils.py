import numpy as np


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)  # UCF(32,2048)
    r = np.linspace(0, len(feat), length + 1, dtype=int)  # (33,)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat
