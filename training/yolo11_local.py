# -*- coding: utf-8 -*-
"""
yolo11 - 로컬(VSCode + CUDA) 실행용

사전 준비 (터미널에서 한 번만):
    pip install ultralytics roboflow pandas matplotlib

실행 전 확인:
    - CUDA_ZIP, EXTRACT_DIR, UNLABELED_DIR 등 경로를 본인 PC 경로로 수정
    - 아래 GPU 확인 코드 실행해서 True 나오는지 먼저 체크
"""

import os
import shutil
import zipfile
import glob

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import yaml
from ultralytics import YOLO


# ============================================================
# 0. GPU 확인 (제일 먼저 실행해서 CUDA 잘 잡히는지 체크)
# ============================================================
print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음 (CPU로 실행됨)")


# ============================================================
# 0-2. 한글 폰트 설정 (Windows는 보통 맑은고딕이 이미 있음)
# ============================================================
# Windows: 맑은 고딕 (기본 설치되어 있음)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Mac이라면 위 줄 대신 아래 사용:
# plt.rcParams['font.family'] = 'AppleGothic'

# 위 폰트가 없다는 에러 뜨면, 아래로 설치된 폰트 목록 확인해서 이름 맞추기
# for f in fm.fontManager.ttflist:
#     if 'Gothic' in f.name or '고딕' in f.name:
#         print(f.name)


# ============================================================
# 1. 압축 해제 (드라이브 마운트 불필요, 로컬 경로 바로 사용)
# ============================================================
ZIP_PATH = r'C:\Users\Woori\Documents\chicken_project\Image_03.yolov11.zip'  # 실제 경로로 수정
EXTRACT_DIR = r'C:\Users\Woori\Documents\chicken_project\labeled_298'        # 실제 경로로 수정

if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)

print("압축 해제 완료:", EXTRACT_DIR)


# ============================================================
# 2. data.yaml 확인
# ============================================================
with open(os.path.join(EXTRACT_DIR, 'data.yaml'), encoding='utf-8') as f:
    yml = yaml.safe_load(f)
print(yml)
print("클래스 순서 확인:", yml['names'])


# ============================================================
# 3. 학습 (device=0 으로 GPU 명시 지정)
# ============================================================
ROUND_NAME = 'mini_seg_v4'
RUNS_ROOT = r'C:\Users\Woori\Documents\chicken_project\runs'  # 코랩의 /content/runs 대신 로컬 경로

prev_run_dir = os.path.join(RUNS_ROOT, 'segment', ROUND_NAME)
if os.path.exists(prev_run_dir):
    shutil.rmtree(prev_run_dir)

model = YOLO('yolo11s-seg.pt')  # 처음 실행 시 자동 다운로드됨

results = model.train(
    data=os.path.join(EXTRACT_DIR, 'data.yaml'),
    imgsz=640,
    epochs=100,
    batch=8,
    patience=20,
    name=ROUND_NAME,
    project=os.path.join(RUNS_ROOT, 'segment'),  # 로컬에선 project 경로를 명시하는 게 안전
    exist_ok=True,
    device=0,   # GPU 0번 사용 명시 (CPU로 돌리려면 device='cpu')
)

RUN_DIR = os.path.join(RUNS_ROOT, 'segment', ROUND_NAME)
print("학습 결과 경로:", RUN_DIR)


# ============================================================
# 3-2. 백업 (로컬이니 드라이브 대신 별도 백업 폴더로)
# ============================================================
BACKUP_DIR = r'C:\Users\Woori\Documents\chicken_project\model_weights'
LOG_DIR = r'C:\Users\Woori\Documents\chicken_project\training_logs'
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

shutil.copy(os.path.join(RUN_DIR, 'weights', 'best.pt'), os.path.join(BACKUP_DIR, f'{ROUND_NAME}_best.pt'))
shutil.copy(os.path.join(RUN_DIR, 'results.csv'), os.path.join(LOG_DIR, f'{ROUND_NAME}_results.csv'))
shutil.copy(os.path.join(RUN_DIR, 'results.png'), os.path.join(LOG_DIR, f'{ROUND_NAME}_results.png'))
print(f"{ROUND_NAME} 백업 완료")


# ============================================================
# 4. test셋 최종 평가
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
# 5. 나머지 이미지 예측
# ============================================================
UNLABELED_DIR = r'C:\Users\Woori\Documents\chicken_project\unlabeled'  # 실제 경로로 수정

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
# 6. Roboflow 업로드 (로컬에서도 동일하게 동작)
# ============================================================
from roboflow import Roboflow
from getpass import getpass

names = yml['names']
with open('labelmap.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(names))

api_key = getpass("Roboflow API Key 입력: ")
rf = Roboflow(api_key=api_key)
project = rf.workspace("s-workspace-boumy").project("find-box_open-food-box_close-table-s8p8s")

PRED_LABEL_DIR = os.path.join(predict_results[0].save_dir, 'labels')

for img_path in image_files[:5]:
    base = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = os.path.join(PRED_LABEL_DIR, f'{base}.txt')

    if not os.path.exists(ann_path):
        project.upload(image_path=img_path, split='train')
        continue

    result = project.single_upload(
        image_path=img_path,
        annotation_path=ann_path,
        annotation_labelmap='labelmap.txt',
        is_prediction=True,
        split='train',
    )
    print(base, "→", result)
