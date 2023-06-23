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

# Reference
- https://github.com/carolchenyx/MGFN./tree/main
    - feature extraction code: https://github.com/carolchenyx/MGFN./issues/6
- https://github.com/louisYen/S3R
- https://github.com/ktr-hubrt/WSAL
- https://github.com/tianyu0207/RTFM
- https://github.com/fjchange/MIST_VAD
- https://github.com/WaqasSultani/AnomalyDetectionCVPR2018
- https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet