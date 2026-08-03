# -*- coding: utf-8 -*-
"""
YOLO 학습 결과(results.csv)의 주요 지표들을 한 장의 대시보드로 시각화.
각 그래프 제목에 지표 설명을 같이 넣어서 포트폴리오/보고서용으로 바로 쓸 수 있게 구성.

사용법 (코랩):
    plot_training_dashboard(
        csv_path='/content/runs/segment/mini_seg_v1/results.csv',
        save_path='/content/drive/MyDrive/training_dashboard.png',
    )
"""

import pandas as pd
import matplotlib.pyplot as plt


# 지표별 설명 (사용자가 정리한 내용 그대로 반영)
METRIC_INFO = {
    'train/cls_loss':        ('Class Loss (분류 손실)', '클래스를 얼마나 정확히 예측하는지에 대한 손실\n▼ 낮을수록 정확하게 분류'),
    'train/dfl_loss':        ('DFL Loss (위치 손실)', '객체의 위치를 얼마나 정확히 예측하는지에 대한 손실\n▼ 낮을수록 위치를 잘 예측'),
    'val/cls_loss':          ('Val Class Loss (검증 분류 손실)', '검증 데이터 기준 분류 손실\n▼ 낮을수록 좋음'),
    'val/box_loss':          ('Val Box Loss (검증 위치 손실)', '검증 데이터 기준 객체 위치 예측 손실\n▼ 낮을수록 좋음'),
    'metrics/precision(B)':  ('Precision (정밀도)', '양성이라 예측한 것 중 실제 양성 비율\n▲ 높을수록 정확하게 분류'),
    'metrics/recall(B)':     ('Recall (재현율)', '실제 양성 중 모델이 놓치지 않고 찾아낸 비율\n▲ 높을수록 실제 양성을 잘 잡아냄'),
    'metrics/mAP50(B)':      ('mAP50', 'IoU 0.5 기준 평균 정밀도(AP)\n▲ 높을수록 정확하게 검출'),
    'metrics/mAP50-95(B)':   ('mAP50-95', '여러 IoU 임계값(0.5~0.95)에 대한 평균 AP\n▲ 높을수록 종합적으로 정확'),
}


def plot_training_dashboard(csv_path, save_path=None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # 실제 존재하는 컬럼만 골라서 그림 (버전에 따라 일부 컬럼명이 다를 수 있음)
    available = [c for c in METRIC_INFO.keys() if c in df.columns]
    missing = [c for c in METRIC_INFO.keys() if c not in df.columns]
    if missing:
        print("⚠️ csv에 없는 컬럼(버전에 따라 이름이 다를 수 있음):", missing)
        print("   실제 컬럼 목록:", df.columns.tolist())

    n = len(available)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(available):
        title, desc = METRIC_INFO[col]
        ax = axes[i]
        ax.plot(df['epoch'], df[col], marker='o', markersize=3, color='#3b6fd6')
        ax.set_title(f"{title}\n{desc}", fontsize=10, loc='left')
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)

    # 안 쓰는 빈 칸 숨기기
    for j in range(len(available), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 완료: {save_path}")
    plt.show()


# ============================================================
# 실행 예시
# ============================================================
"""
plot_training_dashboard(
    csv_path='/content/runs/segment/mini_seg_v1/results.csv',
    save_path='/content/drive/MyDrive/training_dashboard.png',
)
"""
