# MobileViT
MobileViT 방식으로 PyTorch 및 PyTorch framework 사용하여 Vision Task 진행하는 프로젝트입니다.

## Implementations

- MobileViT
- Feature Extractor 
- Data (Prepare)
- Data Augmentations (Albumentations)
- Multi GPU Training

## 프로젝트 구조
```
MobileViT
├─ .gitignore
├─ __README.md
├─ configs # 학습 시 사용할 하이퍼 파라미터, 데이터셋 설정 등 Configuration을 위한 yaml 파일 경로
├─ dataset # Image Data Generator 모듈
├─ models # Network, Convolution Module 등 구현
│  ├─ backbone
│  └─ layers
├─ module # 학습을 위한 Pytorch Lightning 모듈
├─ onnx_module # onnx 변형을 위한 모듈
├─ train_classifier.py # classification 학습 스크립트
└─ utils

```

## Requirements
`PyYaml`
`PyTorch`
`PyTorch Lightning`


## Config Train Parameters

기본 설정값은 ./configs/default_settings.yaml에 정의됩니다.  
Train 스크립트 실행 시 입력되는 CFG 파일로 하이퍼파라미터 및 학습 기법을 설정할 수 있습니다.

[default_settings.yaml](./configs/default_settings.yaml)

    // ./configs/*.yaml 파일 수정
    // ex) cls_frostnet -> default_settings 파라미터를 업데이트 해서 사용
    model : 'MobileViT'
    dataset_name : ImageNet_1k
    classes : 1000
    epochs: 500
    data_path : '/mnt/'
    save_dir : './saved'
    workers: 8
    ...

## Train Detector

Object Classification 모델 Train 스크립트 입니다.

    python train_classifier.py --cfg configs/cls_mobilevit.yaml

## Test MobileViT Classification

## TODOs
- Backbone Networks Customizing
- Deployment를 위한 ONNX Conversion Script, Torch Script 추가
- mAP Callback method 추가
- QAT, Grad Clip, SWA, FP16 등 학습 기법 추가 및 테스트
- Detection, Segmentation 추가
