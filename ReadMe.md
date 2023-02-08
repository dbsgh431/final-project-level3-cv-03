![header](https://capsule-render.vercel.app/api?type=rect&color=gradient&text=아스팔트%20도로%20유지보수를%20위한%20실시간%20도로%20파손%20감지&fontSize=32)
<div align="left">
	<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white" />
	<img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat&logo=Pytorch&logoColor=white" />
	<img src="https://img.shields.io/badge/OpenMMLab-181717?style=flat&logo=Github&logoColor=white" />
</div>
&nbsp;

# Members
- **김도윤**  : Classification Baseline 작성 및 실험, 데이터 전처리 및 Annotation 작업, Object Detection Model 실험, 서비스 배포를 위한 Backend
- **김윤호**  : Classification Baseline 작성 및 실험, 데이터 전처리 및 Annotation 작업, Object Detection Model 실험, 실시간 이미지 전송을 위한 Streamlit Frontend 모듈 구현
- **김종해**  : Frontend 및 Backend Pipeline 구현, 데이터 전처리 및 Annotation 작업, Object Detection Model 실험, GCS 기반 DB 구축, Google Bigquery를 활용한 데이터 Log 생성, Streamlit을 활용한 실시간 데이터 시각화
- **조재효**  : Frontend 및 Backend Pipeline 구현, 데이터 전처리 및 Annotation 작업, Object Detection Model 실험
- **허진녕**  : Classification Baseline 작성 및 실험, 데이터 전처리 및 Annotation 작업, Object Detection Model 실험 

&nbsp;

# 프로젝트 개요
> 폭설, 폭우, 도로의 노후화로 생긴 포트홀은 수많은 운전자를 위협하고 있습니다. 아스팔트 도로 위의 포트홀은 차량을 파손할 뿐만 아니라, 운전자 역시 크게 다치거나 사망하는 인명사고로 이어질 수 있습니다.
하지만 지구 둘레의 약 2.8배인 아스팔트 도로 위에서, 포트홀이 어디에 있는지 알기는 쉽지 않습니다. 이러한 문제를 해결하고자, 인공지능을 활용하여 방대한 아스팔트 도로를 빠르게 점검하고 포트홀을 감지하는 서비스를 개발하며, 결과적으로 신속한 도로 유지보수를 가능하게 할 것입니다.

&nbsp;


# 데이터셋 구조
```
├─ input
│  ├─ train
│  │  ├─ images
│  │  └─ annotations
│  ├─ valid
│  │  ├─ images
│  │  └─ annotations
│  └─ test
│     └─ images
│
└─ final_project
   ├─ .git  
   ├─ mmdetection
   ├─ notebook
   ├─ script
   └─ .gitignore
```	


&nbsp;

# 프로젝트 수행 절차


&nbsp;

# Dataset
### **AIHub 개방 데이터셋**
<img width="500" alt="image" src="https://user-images.githubusercontent.com/69153087/217477704-96233c3b-fc30-4ec2-ba02-fa47bee23d59.png">

&nbsp;

# Annotation Tool
<img width="400" alt="image" src="https://user-images.githubusercontent.com/69153087/217476587-2eccb51c-c5c3-436a-bd8a-7f217fdcc14b.png">

&nbsp;

# Image Classification 모델 실험 결과
|모델|Best Accuracy|
|------|---|
|EfficentNet_b1|90.25|
|EfficentNet_b2|91.35|
|MobileNet_small|73.94|
|MobileNet_large|87.05|
|ResNet50|87.98|
|ResNeSt|90.02|

&nbsp;

# Object Detection 모델 실험 결과
|1 stage 모델|Best mAP50|
|------|---|
|YOLOF|0.3250|
|YOLOX|0.4610|
|YOLOV7|0.5293|

|2 stage 모델|Best mAP50|
|------|---|
|Cascade_rcnn_r101|0.4590|
|Cascade_ConvNeXt|0.4800|
|Cascade_SwinL|0.5860|


# Reference
