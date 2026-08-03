# -*- coding: utf-8 -*-
"""
CVAT에서 export한 51장짜리 세그멘테이션 데이터로 미니 모델을 학습하고,
그 모델로 나머지 이미지들을 예측(자동 라벨링)해서
CVAT에 다시 올릴 pre-annotation을 만드는 전체 파이프라인.

코랩 셀에 순서대로 나눠서 붙여넣어 실행하세요.
"""

# ============================================================
# [셀 1] 드라이브 마운트 & CVAT export zip 압축 해제
# ============================================================
"""
from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os

# 아래 경로는 실제 zip 파일 위치로 수정
CVAT_ZIP = '/content/drive/MyDrive/cvat_export_51.zip'
EXTRACT_DIR = '/content/labeled_51'

os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(CVAT_ZIP, 'r') as z:
    z.extractall(EXTRACT_DIR)

print(os.listdir(EXTRACT_DIR))
# 보통 이런 구조로 나옵니다:
# labeled_51/
#   data.yaml
#   images/
#     Train/ (또는 train/)
#   labels/
#     Train/ (또는 train/)
"""

# ============================================================
# [셀 2] (Roboflow 사용 시 스킵) Roboflow가 이미 train/valid/test로 나눠주므로
# 별도 분리 작업 불필요. 대신 압축 푼 구조와 data.yaml 내용만 확인.
# ============================================================
"""
import os

DATASET_DIR = '/content/labeled_51'  # 실제 압축 해제 경로로 수정

print("=== 폴더 구조 ===")
for root, dirs, files in os.walk(DATASET_DIR):
    level = root.replace(DATASET_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    for f in files[:3]:
        print(f'{indent}  {f}')

print("\\n=== Roboflow가 만든 data.yaml 내용 ===")
print(open(f'{DATASET_DIR}/data.yaml').read())
"""

# ============================================================
# [셀 3] data.yaml의 names 순서 검증 (Roboflow 게 그대로 써도 되는지 확인)
# ============================================================
"""
import yaml

with open(f'{DATASET_DIR}/data.yaml') as f:
    yml = yaml.safe_load(f)

print("Roboflow가 준 클래스 순서:", yml['names'])
print("실제 CVAT/라벨링에서 쓴 순서와 같은지 직접 대조하세요: ['open', 'close', 'food', 'table']")

# 순서가 다르면 아래처럼 덮어써서 새 파일로 저장 (원본은 보존하고 별도 저장 권장)
# yml['names'] = ['open', 'close', 'food', 'table']
# with open(f'{DATASET_DIR}/data_fixed.yaml', 'w') as f:
#     yaml.dump(yml, f, allow_unicode=True)
"""

# ============================================================
# [셀 4] Ultralytics 설치 & 미니 모델 학습
# ============================================================
"""
!pip install ultralytics -q

from ultralytics import YOLO

model = YOLO('yolo11s-seg.pt')   # YOLO11 small 세그멘테이션 모델

results = model.train(
    data=f'{DATASET_DIR}/data.yaml',   # Roboflow가 만든 data.yaml 그대로 사용 (또는 위에서 수정한 data_fixed.yaml)
    imgsz=640,
    epochs=100,      # 51장뿐이라 과적합 위험 있음, 너무 크게 잡지 않기
    batch=8,
    name='mini_seg_v1',
)
"""

# ============================================================
# [셀 5] 학습된 미니 모델로 '나머지' 라벨 안 된 이미지들 예측
# ============================================================
"""
from ultralytics import YOLO

best_model = YOLO('/content/runs/segment/mini_seg_v1/weights/best.pt')  # YOLO11도 동일하게 runs/segment/ 밑에 저장됨

UNLABELED_DIR = '/content/drive/MyDrive/Image/unlabeled_279'  # 나머지 이미지 폴더 경로로 수정

results = best_model.predict(
    source=UNLABELED_DIR,
    save=True,        # 시각화 결과 이미지도 저장 (눈으로 확인용)
    save_txt=True,    # YOLO 세그멘테이션 형식 txt 라벨 자동 생성
    conf=0.4,         # confidence 임계값, 너무 낮으면 오탐 많아짐
    exist_ok=True,
)

print("예측 완료! 결과 위치:")
print(results[0].save_dir)
# 예: runs/segment/predict/labels/ 안에 txt 파일들이 생김
"""

# ============================================================
# [셀 6] 예측 결과를 Roboflow에 pre-annotation으로 업로드
# (CVAT가 아니라 Roboflow를 쓰는 경우 이 버전 사용)
# ============================================================
"""
!pip install roboflow -q

from roboflow import Roboflow
import glob, os

rf = Roboflow(api_key="여기에_본인_API_KEY")   # Roboflow 계정 설정 > API Key 에서 확인
project = rf.workspace("워크스페이스명").project("프로젝트명")

PRED_LABEL_DIR = '/content/runs/segment/predict/labels'   # 셀 5에서 예측된 txt 폴더
IMG_DIR = UNLABELED_DIR                                    # 셀 5에서 예측한 원본 이미지 폴더

image_paths = glob.glob(f'{IMG_DIR}/*.jpg') + glob.glob(f'{IMG_DIR}/*.png')

for img_path in image_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = f'{PRED_LABEL_DIR}/{base}.txt'

    if not os.path.exists(ann_path):
        # 모델이 아무것도 예측 못한 이미지 (라벨 없음) -> 이미지만 업로드
        project.upload(image_path=img_path, split='train')
        continue

    project.single_upload(
        image_path=img_path,
        annotation_path=ann_path,          # YOLO 세그멘테이션 포맷(class x1 y1 x2 y2 ...) 그대로 인식됨
        is_prediction=True,                # '모델이 예측한 라벨'이라는 표시 -> Annotate 화면에서 검수 대상으로 뜸
        split='train',
    )

print("업로드 완료! Roboflow 프로젝트에서 'Annotate' 화면 열어서 예측된 폴리곤 중 틀린 것만 수정하면 됩니다.")
"""
