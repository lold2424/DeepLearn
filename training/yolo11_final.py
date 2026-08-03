# -*- coding: utf-8 -*-
"""
yolo11 - 최종 정리본
치킨 박스 세그멘테이션 (box_open, box_close, food, table)

실행 순서:
  0. 한글 폰트 설치 (제일 먼저, 세션마다 1회)
  1. 드라이브 마운트 & CVAT/Roboflow export 압축 해제
  2. data.yaml 확인
  3. 미니 모델 학습 (+ 학습 끝나자마자 드라이브 백업)
  4. 학습 결과 확인 (대시보드 + train/val 비교 + 과적합 진단)
  5. test셋 최종 평가 (진짜 일반화 성능 확인)
  6. 나머지 이미지 예측 (자동 라벨링)
  7. Roboflow에 pre-annotation 업로드
  8. (선택) 영상 데모
"""

# ============================================================
# 0. 한글 폰트 설치 (세션 새로 열 때마다 제일 먼저 1회 실행)
# ============================================================
!apt-get -qq install fonts-nanum > /dev/null
!fc-cache -fv > /dev/null

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 드라이브 마운트 & 압축 해제
# ============================================================
import zipfile
import os
import shutil

from google.colab import drive
drive.mount('/content/drive')

CVAT_ZIP = '/content/drive/MyDrive/Image/Image_03.yolov11.zip'  # 최신 export로 교체
EXTRACT_DIR = '/content/labeled_298'  # 장수 바뀌면 폴더명도 같이 바꾸기

if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)  # 이전 라운드 잔여물 섞이지 않게 정리
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(CVAT_ZIP, 'r') as z:
    z.extractall(EXTRACT_DIR)

print("압축 해제 완료:", EXTRACT_DIR)


# ============================================================
# 2. data.yaml 확인 (클래스 순서 반드시 확인!)
# ============================================================
import yaml

with open(f'{EXTRACT_DIR}/data.yaml') as f:
    yml = yaml.safe_load(f)
print(yml)
print("클래스 순서 확인:", yml['names'])
# -> 이전 라운드와 순서가 동일한지 반드시 비교 (다르면 성능 비교 자체가 무의미해짐)


# ============================================================
# 3. Ultralytics 설치 & 미니 모델 학습
# ============================================================
!pip install ultralytics -q

from ultralytics import YOLO

ROUND_NAME = 'mini_seg_v4'  # 라운드마다 이름 바꾸기 (v3, v4, v5 ...)

# 같은 이름 폴더가 이미 있으면 Ultralytics가 자동으로 -2, -3 을 붙여버려서
# 나중에 경로 찾기 헷갈리니, 미리 지우고 시작 (필요 없으면 이 3줄 생략 가능)
prev_run_dir = f'/content/runs/segment/{ROUND_NAME}'
if os.path.exists(prev_run_dir):
    shutil.rmtree(prev_run_dir)

model = YOLO('yolo11s-seg.pt')

results = model.train(
    data=f'{EXTRACT_DIR}/data.yaml',
    imgsz=640,
    epochs=100,
    batch=8,
    patience=20,     # val 성능이 20 epoch 동안 안 좋아지면 자동 조기 종료 (약한 과적합 대비)
    name=ROUND_NAME,
    exist_ok=True,
)

RUN_DIR = f'/content/runs/segment/{ROUND_NAME}'  # 실제 저장된 경로 (위에서 폴더 정리했으니 -2 안 붙음)
print("학습 결과 경로:", RUN_DIR)

# ---- 학습 끝나자마자 바로 드라이브 백업 (세션 끊겨도 안전) ----
BACKUP_DIR = '/content/drive/MyDrive/model_weights'
LOG_DIR = '/content/drive/MyDrive/training_logs'
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

shutil.copy(f'{RUN_DIR}/weights/best.pt', f'{BACKUP_DIR}/{ROUND_NAME}_best.pt')
shutil.copy(f'{RUN_DIR}/results.csv', f'{LOG_DIR}/{ROUND_NAME}_results.csv')
shutil.copy(f'{RUN_DIR}/results.png', f'{LOG_DIR}/{ROUND_NAME}_results.png')
print(f"{ROUND_NAME} 백업 완료 (best.pt, results.csv, results.png)")


# ============================================================
# 4. 학습 결과 확인 (대시보드 + train/val 비교 + 자동 과적합 진단)
# ============================================================
# %run "/content/drive/MyDrive/plot_training_dashboard.py"
# %run "/content/drive/MyDrive/plot_train_vs_val_comparison.py"

plot_training_dashboard(
    csv_path=f'{RUN_DIR}/results.csv',
    save_path=f'{LOG_DIR}/{ROUND_NAME}_dashboard.png',
)

plot_train_vs_val(
    csv_path=f'{RUN_DIR}/results.csv',
    save_path=f'{LOG_DIR}/{ROUND_NAME}_overfit_check.png',
)

# best epoch 확인 (최종 epoch과 너무 멀면 과적합 의심)
import pandas as pd
df = pd.read_csv(f'{RUN_DIR}/results.csv')
df.columns = df.columns.str.strip()
best_epoch = df['metrics/mAP50-95(B)'].idxmax()
print(f"mAP50-95 기준 최고 성능 epoch: {best_epoch} / 전체 {len(df)} epoch")
print(f"최고 mAP50-95: {df['metrics/mAP50-95(B)'].max():.4f}")
print(f"마지막 epoch mAP50-95: {df['metrics/mAP50-95(B)'].iloc[-1]:.4f}")

# 이전 라운드와 비교 (실제 존재하는 csv만 넣을 것! 없는 경로 넣으면 에러남)
plot_round_comparison(
    round_configs=[
        {'name': 'v3 (298장)', 'csv_path': '/content/drive/MyDrive/training_logs/mini_seg_v3_results.csv'},
        {'name': f'{ROUND_NAME}', 'csv_path': f'{RUN_DIR}/results.csv'},
    ],
    save_path=f'{LOG_DIR}/round_comparison.png',
)


# ============================================================
# 5. test셋 최종 평가 (진짜 일반화 성능 - 가장 신뢰도 높은 지표)
# ============================================================
best_model = YOLO(f'{RUN_DIR}/weights/best.pt')

test_metrics = best_model.val(
    data=f'{EXTRACT_DIR}/data.yaml',
    split='test',
)

print("=== Test셋 평가 결과 ===")
print("Test mAP50:", test_metrics.seg.map50)
print("Test mAP50-95:", test_metrics.seg.map)
print("Test Precision:", test_metrics.seg.mp)
print("Test Recall:", test_metrics.seg.mr)
print(f"\n(참고) Val 기준 mAP50-95 최고값: {df['metrics/mAP50-95(B)'].max():.4f}")
print("-> 위 두 값 격차가 크면 과적합, 비슷하면 정상적으로 일반화된 것")


# ============================================================
# 6. 나머지 라벨 안 된 이미지들 예측 (자동 라벨링용)
# ============================================================
import glob

UNLABELED_DIR = '/content/drive/MyDrive/Image/unlabeled'

# 폴더 전체가 아니라 이미지 파일만 리스트로 (영상 섞여서 안 멈추게)
image_files = glob.glob(f'{UNLABELED_DIR}/*.jpg') + glob.glob(f'{UNLABELED_DIR}/*.png')
print(f"예측할 이미지 수: {len(image_files)}")

predict_results = best_model.predict(
    source=image_files,
    save=True,
    save_txt=True,
    conf=0.4,
    exist_ok=True,
)

print("예측 완료! 결과 위치:", predict_results[0].save_dir)


# ============================================================
# 7. Roboflow에 pre-annotation 업로드
# ============================================================
!pip install roboflow -q

from roboflow import Roboflow
from getpass import getpass

names = yml['names']
with open('/content/labelmap.txt', 'w') as f:
    f.write('\n'.join(names))

api_key = getpass("Roboflow API Key 입력: ")
rf = Roboflow(api_key=api_key)
project = rf.workspace("s-workspace-boumy").project("find-box_open-food-box_close-table-s8p8s")

PRED_LABEL_DIR = f'{predict_results[0].save_dir}/labels'

# 먼저 5장만 테스트
for img_path in image_files[:5]:
    base = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = f'{PRED_LABEL_DIR}/{base}.txt'

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

# ---- 5장 테스트 결과 success=True 확인 후 아래 실행 (나머지 전체) ----
"""
for img_path in image_files[5:]:
    base = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = f'{PRED_LABEL_DIR}/{base}.txt'

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


# ============================================================
# 8. (선택) 영상 데모 - 라벨링/학습 다 끝난 뒤 나중에 진행
# ============================================================
"""
video_results = best_model.predict(
    source='/content/drive/MyDrive/Image/unlabeled_videos/test_data02.mp4',
    save=True,
    conf=0.25,   # 영상은 조명/각도가 달라 학습 데이터보다 낮게 잡는 게 안전
    exist_ok=True,
    name='video_test',
)
print("영상 예측 결과:", video_results[0].save_dir)

# avi -> mp4 변환 후 노트북에서 재생
import glob, subprocess
from IPython.display import HTML
from base64 import b64encode

avi_path = glob.glob(f'{video_results[0].save_dir}/*.avi')[0]
mp4_path = avi_path.replace('.avi', '.mp4')
subprocess.run(['ffmpeg', '-y', '-i', avi_path, '-vcodec', 'libx264', mp4_path])

mp4 = open(mp4_path, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f'<video width=500 controls><source src="{data_url}" type="video/mp4"></video>')
"""
