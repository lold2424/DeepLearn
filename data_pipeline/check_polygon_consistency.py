# -*- coding: utf-8 -*-
"""
특정 클래스의 라벨들이 '점 개수'(폴리곤 정밀도) 기준으로 일관성이 있는지 확인.
같은 클래스인데 어떤 건 4~6개 점(대충 그린 사각형), 어떤 건 수십 개 점(정교한 폴리곤)으로
섞여 있으면 세그멘테이션 학습에 혼란을 줄 수 있음.

사용법:
    python check_polygon_consistency.py --labels_dir train/labels --data_yaml data.yaml --class_name table
"""

import os
import argparse
import yaml
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels_dir', action='append', required=True)
    ap.add_argument('--data_yaml', required=True)
    ap.add_argument('--class_name', required=True, help='확인할 클래스 이름 (예: table)')
    args = ap.parse_args()

    with open(args.data_yaml, encoding='utf-8') as f:
        yml = yaml.safe_load(f)
    names = yml['names']

    if args.class_name not in names:
        print(f"⚠️ '{args.class_name}' 클래스가 없습니다. 실제 클래스: {names}")
        return

    target_cls_id = names.index(args.class_name)
    print(f"'{args.class_name}' 클래스 id: {target_cls_id}\n")

    point_counts = []
    examples = []  # (파일명, 점개수) 몇 개 기록해서 나중에 확인용
    bbox_format_count = 0  # 5개 값(class x y w h)짜리 bbox 형식 라벨 개수 (폴리곤 아님, 별도 집계)

    for labels_dir in args.labels_dir:
        if not os.path.exists(labels_dir):
            print(f"⚠️ 경로 없음: {labels_dir}")
            continue
        for fname in os.listdir(labels_dir):
            if not fname.endswith('.txt'):
                continue
            with open(os.path.join(labels_dir, fname), encoding='utf-8') as f:
                for line in f:
                    vals = line.strip().split()
                    if not vals:
                        continue
                    cls_id = int(vals[0])
                    if cls_id != target_cls_id:
                        continue
                    num_coords = len(vals) - 1

                    if num_coords == 4:
                        # bbox 형식(class x_center y_center w h) -> 폴리곤 점 개수 계산에서 제외
                        bbox_format_count += 1
                        continue

                    num_points = num_coords // 2
                    point_counts.append(num_points)
                    examples.append((fname, num_points))

    if not point_counts and bbox_format_count == 0:
        print("해당 클래스의 인스턴스를 찾지 못했습니다.")
        return

    print(f"폴리곤 형식 인스턴스 수: {len(point_counts)}")
    print(f"bbox 형식(class x y w h) 인스턴스 수: {bbox_format_count}  <- 폴리곤과 섞여 있다면 이 자체가 형식 불일치 신호\n")

    if not point_counts:
        print("폴리곤 형식 인스턴스가 없어 점 개수 분포는 계산하지 않습니다.")
        return
    print(f"점 개수 최소: {min(point_counts)}, 최대: {max(point_counts)}, 평균: {sum(point_counts)/len(point_counts):.1f}\n")

    # 구간별로 분포 확인 (4~6=사각형 추정, 그 이상=정교한 폴리곤 추정)
    buckets = Counter()
    for p in point_counts:
        if p <= 6:
            buckets['4~6개 (대충 그린 사각형 추정)'] += 1
        elif p <= 15:
            buckets['7~15개 (중간)'] += 1
        else:
            buckets['16개 이상 (정교한 폴리곤 추정)'] += 1

    print("=== 점 개수 구간별 분포 ===")
    for bucket, count in buckets.items():
        ratio = count / len(point_counts) * 100
        print(f"  {bucket}: {count}개 ({ratio:.1f}%)")

    # 두 구간 다 상당수 있으면 혼재 가능성 경고
    non_zero_buckets = sum(1 for c in buckets.values() if c > 0)
    if non_zero_buckets >= 2 and min(buckets.values()) / len(point_counts) > 0.15:
        print("\n⚠️ 라벨링 방식이 섞여 있을 가능성이 높습니다 (여러 구간에 고르게 분포).")
        print("   같은 클래스를 이미지마다 다른 정밀도로 라벨링했다면, 재작업을 고려하세요.")
    else:
        print("\n라벨링 방식이 비교적 일관되어 보입니다.")

    # 예시 파일 몇 개씩 보여주기 (각 구간에서)
    print("\n=== 구간별 예시 파일 (최대 5개씩) ===")
    shown = {'적음': 0, '중간': 0, '많음': 0}
    for fname, p in sorted(examples, key=lambda x: x[1]):
        if p <= 6 and shown['적음'] < 5:
            print(f"  [점 {p}개] {fname}")
            shown['적음'] += 1
    for fname, p in examples:
        if p > 15 and shown['많음'] < 5:
            print(f"  [점 {p}개] {fname}")
            shown['많음'] += 1


if __name__ == '__main__':
    main()
