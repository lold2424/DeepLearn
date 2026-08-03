# -*- coding: utf-8 -*-
"""
yolo11_v05 - 수정본
채도 처리(desaturation) 실험이 실제로 백업/평가/배포까지 이어지도록 수정.

주요 수정 사항:
1. 중복 학습 제거 - 채도 처리된 데이터로만 학습, 원본으로 재학습하는 블록 삭제
2. desaturate_box_regions.py 업로드 코드 추가
3. yml 미정의 버그 수정 (data.yaml을 명시적으로 로드)
4. 죽은 코드(desaturate_box_region 중복 함수 정의) 제거
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

print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")

ZIP_PATH = '/content/drive/MyDrive/Image/Image_05.yolov11.zip'
EXTRACT_DIR = '/content/yolo11_v5'

if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)

print("압축 해제 완료:", EXTRACT_DIR)

# ============================================================
# 2-1. data.yaml 로드 (yml 변수 정의 - 8번 섹션에서 사용됨)
# ============================================================
with open(os.path.join(EXTRACT_DIR, 'data.yaml'), encoding='utf-8') as f:
    yml = yaml.safe_load(f)
print(yml)
print("클래스 순서 확인:", yml['names'])

# ============================================================
# 2-2. desaturate_box_regions.py 업로드
# ============================================================
# 세션에 파일이 없으면 아래 주석 풀어서 업로드 창 띄우기
# from google.colab import files
# uploaded = files.upload()  # desaturate_box_regions.py 선택

# 드라이브에 미리 올려두셨다면 복사만:
shutil.copy('/content/drive/MyDrive/desaturate_box_regions.py', '/content/desaturate_box_regions.py')

# ============================================================
# 2-3. 이미지 전처리 (채도 수정) - train에만 적용
# ============================================================
!python /content/desaturate_box_regions.py \
    --images_dir /content/yolo11_v5/train/images \
    --labels_dir /content/yolo11_v5/train/labels \
    --data_yaml /content/yolo11_v5/data.yaml \
    --output_images_dir /content/yolo11_v5/train_desat/images \
    --output_labels_dir /content/yolo11_v5/train_desat/labels

from IPython.display import Image, display

sample = os.listdir('/content/yolo11_v5/train_desat/images')[0]
display(Image(filename=f'/content/yolo11_v5/train_desat/images/{sample}', width=500))

# ============================================================
# 2-4. 채도 처리된 데이터를 가리키는 새 data.yaml 작성
# ============================================================
new_data_yaml = {
    'train': '/content/yolo11_v5/train_desat/images',  # 채도 처리된 폴더
    'val': '/content/yolo11_v5/valid/images',            # val은 원본 그대로 (실제 평가 위해)
    'test': '/content/yolo11_v5/test/images',             # test도 원본 그대로
    'nc': 4,
    'names': yml['names'],  # 원본 data.yaml과 동일한 클래스 순서 유지
}

with open('/content/yolo11_v5/data_desat.yaml', 'w') as f:
    yaml.dump(new_data_yaml, f, allow_unicode=True)

print(open('/content/yolo11_v5/data_desat.yaml').read())

# ============================================================
# 3. 학습 (채도 처리된 데이터로 - 이것 하나만 진행, 재학습 없음)
# ============================================================
ROUND_NAME = 'mini_seg_v5_desat'
RUNS_ROOT = '/content/runs'

prev_run_dir = os.path.join(RUNS_ROOT, 'segment', ROUND_NAME)
if os.path.exists(prev_run_dir):
    shutil.rmtree(prev_run_dir)

model = YOLO('yolo11s-seg.pt')

results = model.train(
    data='/content/yolo11_v5/data_desat.yaml',   # 채도 처리 버전으로 학습
    imgsz=640,
    epochs=100,
    batch=16,
    patience=20,
    name=ROUND_NAME,
    project=os.path.join(RUNS_ROOT, 'segment'),
    exist_ok=True,
    device=0,
)

RUN_DIR = os.path.join(RUNS_ROOT, 'segment', ROUND_NAME)
print("학습 결과 경로:", RUN_DIR)

# ============================================================
# 4. 백업
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
# 5. test셋 최종 평가 (원본 test 데이터로 - val/test는 desat 처리 안 했으므로 정확한 실전 평가)
# ============================================================
best_model = YOLO(os.path.join(RUN_DIR, 'weights', 'best.pt'))

test_metrics = best_model.val(
    data='/content/yolo11_v5/data_desat.yaml',
    split='test',
    device=0,
)

print("=== Test셋 평가 결과 (채도 처리 학습 모델) ===")
print("Test mAP50:", test_metrics.seg.map50)
print("Test mAP50-95:", test_metrics.seg.map)

# ============================================================
# 6. 나머지 이미지 예측
# ============================================================
UNLABELED_DIR = '/content/drive/MyDrive/Image/unlabeled'

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
# 7. Roboflow에 pre-annotation 업로드
# ============================================================
!pip install roboflow -q

from roboflow import Roboflow
from getpass import getpass

names = yml['names']
with open('/content/labelmap.txt', 'w', encoding='utf-8') as f:
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
        annotation_labelmap='/content/labelmap.txt',
        is_prediction=True,
        split='train',
    )
    print(base, "→", result)

# ---- 5장 테스트 success=True 확인 후 아래 실행 ----
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
