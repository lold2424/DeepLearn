# -*- coding: utf-8 -*-
"""
라벨링 라운드(51장 -> 100장 -> ... -> 330장)가 진행될 때마다
학습된 모델의 성능을 CSV 파일에 한 줄씩 누적 기록하는 스크립트.

포트폴리오에서 "데이터가 늘어남에 따라 성능이 어떻게 개선됐는지"를
표/그래프로 보여줄 때 그대로 사용할 수 있습니다.

사용법 (매 학습 라운드가 끝날 때마다 실행):
    LOG_CSV = '/content/drive/MyDrive/training_log.csv'  # 드라이브에 저장 -> 세션 끊겨도 안 없어짐
    log_round(
        log_path=LOG_CSV,
        round_name='round1_51images',
        num_images=51,
        model_path='/content/runs/segment/mini_seg_v1/weights/best.pt',
        data_yaml='/content/labeled_51/data.yaml',
    )
"""

import os
import csv
from datetime import datetime


def log_round(log_path, round_name, num_images, model_path, data_yaml):
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'round': round_name,
        'num_labeled_images': num_images,
        'mAP50_mask': round(float(metrics.seg.map50), 4),
        'mAP50-95_mask': round(float(metrics.seg.map), 4),
        'precision_mask': round(float(metrics.seg.mp), 4),
        'recall_mask': round(float(metrics.seg.mr), 4),
        'mAP50_box': round(float(metrics.box.map50), 4),
        'mAP50-95_box': round(float(metrics.box.map), 4),
        'model_path': model_path,
    }

    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"기록 완료: {round_name}")
    for k, v in row.items():
        print(f"  {k}: {v}")

    return row


# ============================================================
# 실제 사용 예시 (코랩 셀에 복사해서 쓰기)
# ============================================================
"""
LOG_CSV = '/content/drive/MyDrive/training_log.csv'

log_round(
    log_path=LOG_CSV,
    round_name='round1_51images',
    num_images=51,
    model_path='/content/runs/segment/mini_seg_v1/weights/best.pt',
    data_yaml='/content/labeled_51/data.yaml',
)

# 나중에 라벨을 더 채운 뒤 재학습하면, 새 라운드 이름으로 또 기록
# log_round(
#     log_path=LOG_CSV,
#     round_name='round2_150images',
#     num_images=150,
#     model_path='/content/runs/segment/mini_seg_v2/weights/best.pt',
#     data_yaml='/content/labeled_150/data.yaml',
# )

# 전체 기록 확인 (누적된 표 전체 보기)
import pandas as pd
print(pd.read_csv(LOG_CSV))
"""
