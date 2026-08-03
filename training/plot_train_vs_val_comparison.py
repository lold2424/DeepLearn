# -*- coding: utf-8 -*-
"""
1) train loss vs val loss를 겹쳐 그려서 과적합(overfitting) 여부를 눈으로 확인하는 차트
   -> train은 계속 내려가는데 val이 어느 순간부터 정체/상승하면 과적합 = 규제 필요 신호

2) 여러 학습 라운드(v1, v2, v3...)의 최종 성능을 막대그래프로 나란히 비교
   -> "데이터를 늘릴수록 성능이 좋아졌다"를 포트폴리오에 보여주기 좋음

사용법은 파일 하단 실행 예시 참고.
"""

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1) train vs val loss 오버레이 (과적합 진단용)
# ============================================================
def plot_train_vs_val(csv_path, save_path=None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Ultralytics 버전에 따라 컬럼명이 조금씩 다를 수 있어서 후보를 순서대로 시도
    pairs = [
        ('train/box_loss', 'val/box_loss', 'Box Loss (위치)'),
        ('train/cls_loss', 'val/cls_loss', 'Class Loss (분류)'),
        ('train/dfl_loss', 'val/dfl_loss', 'DFL Loss'),
        ('train/seg_loss', 'val/seg_loss', 'Seg Loss (마스크)'),
    ]
    available = [(t, v, name) for t, v, name in pairs if t in df.columns and v in df.columns]

    if not available:
        print("⚠️ train/val 짝이 맞는 loss 컬럼을 못 찾았습니다. 실제 컬럼:", df.columns.tolist())
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (train_col, val_col, name) in zip(axes, available):
        ax.plot(df['epoch'], df[train_col], label='Train Loss', color='#3b6fd6')
        ax.plot(df['epoch'], df[val_col], label='Val Loss', color='#e0623b')
        ax.set_title(f"{name}\nTrain vs Val")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 완료: {save_path}")
    plt.show()

    # 간단 진단 코멘트 자동 출력
    print("\n=== 자동 진단 ===")
    for train_col, val_col, name in available:
        last_n = min(10, len(df))
        train_trend = df[train_col].iloc[-last_n:].diff().mean()
        val_trend = df[val_col].iloc[-last_n:].diff().mean()
        if train_trend < 0 and val_trend > 0:
            print(f"[{name}] ⚠️ 후반 epoch에서 train은 감소, val은 증가 -> 과적합 가능성 있음, 규제 검토 필요")
        elif abs(val_trend) < abs(train_trend) * 0.3:
            print(f"[{name}] val loss가 거의 정체 상태 -> 데이터가 더 필요하거나 학습이 수렴한 상태")
        else:
            print(f"[{name}] train/val 둘 다 비슷하게 감소 중 -> 정상적인 학습 진행")


# ============================================================
# 2) 여러 라운드의 최종 성능 막대그래프 비교
# ============================================================
def plot_round_comparison(round_configs, save_path=None):
    """
    round_configs: [
        {'name': 'v1 (51장)', 'csv_path': '.../results.csv'},
        {'name': 'v3 (298장)', 'csv_path': '.../results.csv'},
    ]
    각 csv의 '마지막 epoch' 값을 최종 성능으로 사용.
    """
    metrics_cols = {
        'Precision': 'metrics/precision(B)',
        'Recall': 'metrics/recall(B)',
        'mAP50': 'metrics/mAP50(B)',
        'mAP50-95': 'metrics/mAP50-95(B)',
    }

    records = []
    for cfg in round_configs:
        df = pd.read_csv(cfg['csv_path'])
        df.columns = df.columns.str.strip()
        last_row = df.iloc[-1]
        row = {'round': cfg['name']}
        for label, col in metrics_cols.items():
            row[label] = last_row[col] if col in df.columns else None
        records.append(row)

    result_df = pd.DataFrame(records).set_index('round')
    print(result_df)

    ax = result_df.plot(kind='bar', figsize=(10, 6), rot=0)
    ax.set_ylabel('Score')
    ax.set_title('라운드별 최종 성능 비교')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 완료: {save_path}")
    plt.show()

    return result_df


# ============================================================
# 실행 예시
# ============================================================
"""
# 1) 과적합 여부 확인 (v3 기준)
plot_train_vs_val(
    csv_path='/content/runs/segment/mini_seg_v3/results.csv',
    save_path='/content/drive/MyDrive/v3_overfit_check.png',
)

# 2) v1 vs v3 최종 성능 비교
plot_round_comparison(
    round_configs=[
        {'name': 'v1 (51장)', 'csv_path': '/content/runs/segment/mini_seg_v1/results.csv'},
        {'name': 'v3 (298장)', 'csv_path': '/content/runs/segment/mini_seg_v3/results.csv'},
    ],
    save_path='/content/drive/MyDrive/round_comparison.png',
)
"""
