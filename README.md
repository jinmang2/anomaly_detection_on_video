# Anomaly Detection on Video

## 개발 중인 파일
- `modeling_mgfn.py`
- `configuration_mgfn.py`
- `loss_mgfn.py`
- `convert_official_to_hf.py`
- `test.py`

현재는 mgfn 모델을 기반으로 개발에 착수. gradio로 huggingface space에 demo를 올린 이후에는 Reference에 있는 다른 모델들에 대한 코드와 `lightning` 기반 학습 스크립트 작성 예정.

## Deprecated 파일
- `config.py`
- `dataset.py`
- `loss_official.py`
- `mgfn.py`
- `option.py`
- `utils.py`

위 파일들은 [original repo](https://github.com/carolchenyx/MGFN.)의 code들을 모델 재구현 비교를 위해 임시로 commit한 파일들. (+ error 발생하는 부분들 수정)

demo 구현 및 `test.py`가 class화 된 이후엔 전부 삭제 예정

## Feature Extraction Demo code

```python
!pip install torch av pytorchvideo transformers

import av
import numpy as np
from torch.utils.data import DataLoader
from transformers.trainer import nested_concat

from dataset import TenCropVideoFrameDataset
from i3d import build_i3d_feature_extractor

sample_video_path = "/content/Abuse001_x264.mp4"
dataset = TenCropVideoFrameDataset(sample_video_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = build_i3d_feature_extractor()
model.eval()
model.cuda()

outputs = None
for step, inputs in enumerate(dataloader):
    inputs = inputs.view(-1, 16, 3, 224, 224).permute(0, 2, 1, 3, 4).cuda()
    output = model(inputs)
    output = output.detach().cpu().numpy()
    outputs = output if outputs is None else nested_concat(outputs, output)

print(output.shape)
```
```
# 171 * ten_crops, feature_dimension, 1, 1
# `Abuse001_x264.mp4` video's n_frames == 2729
# frames_per_clip == 16
# 2729 // 16 + 1 == 171
(1710, 2048, 1, 1)
```

# Reference
- https://github.com/carolchenyx/MGFN./tree/main
    - feature extraction code: https://github.com/carolchenyx/MGFN./issues/6
- https://github.com/louisYen/S3R
- https://github.com/ktr-hubrt/WSAL
- https://github.com/tianyu0207/RTFM
- https://github.com/fjchange/MIST_VAD
- https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
- https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet