# -*- coding: utf-8 -*-
"""
v3 최종본 재현 학습 - 298장(Image_03.yolov11.zip) 데이터로 재학습해서
최종 pt를 확실하게 드라이브에 백업하는 스크립트.

seed=0, deterministic=True가 기본 설정이라 이전 v3 결과와 거의 동일하게 재현됩니다.
"""

# ============================================================
# 0. 한글 폰트 설치
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
!pip install ultralytics -q

# ============================================================
# 2. 드라이브 마운트 & 압축 해제
# ============================================================
import os
import shutil
import zipfile

import torch
import yaml
from ultralytics import YOLO
from google.colab import drive

drive.mount('/content/drive')

print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")

ZIP_PATH = '/content/drive/MyDrive/Image/Image_03.yolov11.zip'  # v3(298장) zip 경로 - 실제 위치로 확인 필요
EXTRACT_DIR = '/content/yolo11_v3_final'

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
# 4. 학습 (v3 원본 설정 그대로 재현)
# ============================================================
ROUND_NAME = 'mini_seg_v3_final'
RUNS_ROOT = '/content/runs'

prev_run_dir = os.path.join(RUNS_ROOT, 'segment', ROUND_NAME)
if os.path.exists(prev_run_dir):
    shutil.rmtree(prev_run_dir)

model = YOLO('yolo11s-seg.pt')

results = model.train(
    data=os.path.join(EXTRACT_DIR, 'data.yaml'),
    imgsz=640,
    epochs=100,
    batch=16,
    name=ROUND_NAME,
    project=os.path.join(RUNS_ROOT, 'segment'),
    exist_ok=True,
    device=0,
)

RUN_DIR = os.path.join(RUNS_ROOT, 'segment', ROUND_NAME)
print("학습 결과 경로:", RUN_DIR)

# ============================================================
# 5. 백업 - 이번엔 확실하게 드라이브에 저장 (최종 산출물)
# ============================================================
BACKUP_DIR = '/content/drive/MyDrive/model_weights'
LOG_DIR = '/content/drive/MyDrive/training_logs'
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

best_pt_src = os.path.join(RUN_DIR, 'weights', 'best.pt')
best_pt_dst = os.path.join(BACKUP_DIR, 'mini_seg_v3_final_best.pt')

shutil.copy(best_pt_src, best_pt_dst)
shutil.copy(os.path.join(RUN_DIR, 'results.csv'), os.path.join(LOG_DIR, f'{ROUND_NAME}_results.csv'))
shutil.copy(os.path.join(RUN_DIR, 'results.png'), os.path.join(LOG_DIR, f'{ROUND_NAME}_results.png'))

# 백업 성공 여부 확실히 확인
if os.path.exists(best_pt_dst):
    size_mb = os.path.getsize(best_pt_dst) / 1024 / 1024
    print(f"✅ 백업 완료: {best_pt_dst} ({size_mb:.1f} MB)")
else:
    print("⚠️ 백업 실패! 드라이브 마운트 상태나 경로를 확인하세요.")

# ============================================================
# 6. test셋 최종 평가 (최종 성능 확인)
# ============================================================
best_model = YOLO(best_pt_dst)

test_metrics = best_model.val(
    data=os.path.join(EXTRACT_DIR, 'data.yaml'),
    split='test',
    device=0,
)

print("\n=== 최종 Test셋 평가 결과 (이게 v3 최종 성능) ===")
print("Test mAP50:", test_metrics.seg.map50)
print("Test mAP50-95:", test_metrics.seg.map)

print(f"\n최종 pt 파일 위치: {best_pt_dst}")
print("이 파일이 최종 산출물입니다. 다운로드하거나 배포에 바로 사용하시면 됩니다.")
