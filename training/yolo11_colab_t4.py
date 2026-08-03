# -*- coding: utf-8 -*-
"""
yolo11 - 코랩(T4 GPU) 실행용

로컬 버전에서 아래를 코랩에 맞게 되돌림:
  - 드라이브 마운트 추가
  - 경로를 /content/... 로 변경
  - !pip install 로 패키지 설치
  - 한글 폰트를 나눔고딕(apt 설치)으로 변경 (맑은고딕은 코랩에 없음)
  - Windows 전용이었던 KMP_DUPLICATE_LIB_OK, workers=0 은 불필요해서 제거
    (Linux는 fork 방식이라 멀티프로세싱 충돌 문제 자체가 없음)
  - if __name__ == '__main__': 가드도 노트북 셀 방식이라 불필요, 제거
    (섹션별로 셀 나눠서 순서대로 실행)
"""

# ============================================================
# 0. 한글 폰트 설치 (세션 새로 열 때마다 제일 먼저 1회)
# ============================================================
!apt-get -qq install fonts-nanum > /dev/null
!fc-cache -fv > /dev/null

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 패키지 설치
# ============================================================
!pip install ultralytics roboflow -q


# ============================================================
# 2. 드라이브 마운트 & 압축 해제
# ============================================================
import os
import shutil
import zipfile
import glob

import torch
import pandas as pd
import yaml
from ultralytics import YOLO
from google.colab import drive

drive.mount('/content/drive')

# GPU 확인
print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")

ZIP_PATH = '/content/drive/MyDrive/Image/Image_03.yolov11.zip'  # 실제 경로로 수정
EXTRACT_DIR = '/content/yolo11_v4'

if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)

print("압축 해제 완료:", EXTRACT_DIR)


# ============================================================
# 3. data.yaml 확인
# ============================================================
with open(os.path.join(EXTRACT_DIR, 'data.yaml'), encoding='utf-8') as f:
    yml = yaml.safe_load(f)
print(yml)
print("클래스 순서 확인:", yml['names'])


# ============================================================
# 4. 학습 (T4는 VRAM 16GB라 로컬 GTX1060 3GB보다 훨씬 여유 있음)
# ============================================================
ROUND_NAME = 'mini_seg_v4'
RUNS_ROOT = '/content/runs'

prev_run_dir = os.path.join(RUNS_ROOT, 'segment', ROUND_NAME)
if os.path.exists(prev_run_dir):
    shutil.rmtree(prev_run_dir)

model = YOLO('yolo11s-seg.pt')

results = model.train(
    data=os.path.join(EXTRACT_DIR, 'data.yaml'),
    imgsz=640,       # T4는 VRAM 여유 있으니 640으로 (로컬에선 416으로 낮췄었음)
    epochs=100,
    batch=16,        # T4 기준으로 배치도 키움 (로컬 GTX1060은 4였음)
    patience=20,
    name=ROUND_NAME,
    project=os.path.join(RUNS_ROOT, 'segment'),
    exist_ok=True,
    device=0,
)

RUN_DIR = os.path.join(RUNS_ROOT, 'segment', ROUND_NAME)
print("학습 결과 경로:", RUN_DIR)


# ============================================================
# 5. 백업 (드라이브에 저장 - 세션 끊겨도 안전)
# ============================================================
BACKUP_DIR = '/content/drive/MyDrive/model_weights'
LOG_DIR = '/content/drive/MyDrive/training_logs'
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

shutil.copy(os.path.join(RUN_DIR, 'weights', 'best.pt'), os.path.join(BACKUP_DIR, f'{ROUND_NAME}_best.pt'))
shutil.copy(os.path.join(RUN_DIR, 'results.csv'), os.path.join(LOG_DIR, f'{ROUND_NAME}_results.csv'))
shutil.copy(os.path.join(RUN_DIR, 'results.png'), os.path.join(LOG_DIR, f'{ROUND_NAME}_results.png'))
print(f"{ROUND_NAME} 백업 완료")


# ============================================================
# 6. test셋 최종 평가
# ============================================================
best_model = YOLO(os.path.join(RUN_DIR, 'weights', 'best.pt'))

test_metrics = best_model.val(
    data=os.path.join(EXTRACT_DIR, 'data.yaml'),
    split='test',
    device=0,
)

print("=== Test셋 평가 결과 ===")
print("Test mAP50:", test_metrics.seg.map50)
print("Test mAP50-95:", test_metrics.seg.map)


# ============================================================
# 7. 나머지 이미지 예측
# ============================================================
UNLABELED_DIR = '/content/drive/MyDrive/Image/unlabeled'  # 실제 경로로 수정

image_files = glob.glob(os.path.join(UNLABELED_DIR, '*.jpg')) + glob.glob(os.path.join(UNLABELED_DIR, '*.png'))
print(f"예측할 이미지 수: {len(image_files)}")

predict_results = best_model.predict(
    source=image_files,
    save=True,
    save_txt=True,
    conf=0.4,
    exist_ok=True,
    device=0,
)
print("예측 완료! 결과 위치:", predict_results[0].save_dir)


# ============================================================
# 8. Roboflow에 pre-annotation 업로드
# ============================================================
from roboflow import Roboflow
from getpass import getpass

names = yml['names']
with open('/content/labelmap.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(names))

api_key = getpass("Roboflow API Key 입력: ")
rf = Roboflow(api_key=api_key)
project = rf.workspace("s-workspace-boumy").project("find-box_open-food-box_close-table-s8p8s")

PRED_LABEL_DIR = os.path.join(predict_results[0].save_dir, 'labels')

# 먼저 5장만 테스트
for img_path in image_files[:5]:
    base = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = os.path.join(PRED_LABEL_DIR, f'{base}.txt')

    if not os.path.exists(ann_path):
        project.upload(image_path=img_path, split='train')
        continue

    result = project.single_upload(
        image_path=img_path,
        annotation_path=ann_path,
        annotation_labelmap='/content/labelmap.txt',
        is_prediction=True,
        split='train',
    )
    print(base, "→", result)

# ---- 5장 테스트 success=True 확인 후 아래 실행 (나머지 전체) ----
"""
for img_path in image_files[5:]:
    base = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = os.path.join(PRED_LABEL_DIR, f'{base}.txt')

    if not os.path.exists(ann_path):
        project.upload(image_path=img_path, split='train')
        continue

    project.single_upload(
        image_path=img_path,
        annotation_path=ann_path,
        annotation_labelmap='/content/labelmap.txt',
        is_prediction=True,
        split='train',
    )

print("전체 업로드 완료!")
"""
