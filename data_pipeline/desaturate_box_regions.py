# -*- coding: utf-8 -*-
"""
box_open / box_close 클래스의 폴리곤 영역만 채도를 낮춰서
'박스 표면에 인쇄된 그림'을 food로 오탐하는 문제를 완화하는 전처리.

train 세트에만 적용 (val/test는 실제 환경 그대로 평가해야 하므로 원본 유지).

사용법:
    python desaturate_box_regions.py \
        --images_dir /path/to/train/images \
        --labels_dir /path/to/train/labels \
        --data_yaml /path/to/data.yaml \
        --output_images_dir /path/to/train_desat/images \
        --output_labels_dir /path/to/train_desat/labels

처리 후 model.train(data=...) 의 data.yaml에서 train 경로를
output_images_dir 쪽으로 바꿔서 학습하면 됩니다.
(라벨은 좌표가 그대로라 바뀌지 않으므로 단순 복사)
"""

import os
import shutil
import argparse

import cv2
import numpy as np
import yaml


def polygon_norm_to_pixel(coords, w, h):
    pts = []
    for i in range(0, len(coords), 2):
        x = coords[i] * w
        y = coords[i + 1] * h
        pts.append((x, y))
    return pts


def desaturate_regions(image, polygons, desaturate_ratio=0.9):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.where(mask == 255, hsv[..., 1] * (1 - desaturate_ratio), hsv[..., 1])
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--labels_dir', required=True)
    ap.add_argument('--data_yaml', required=True)
    ap.add_argument('--output_images_dir', required=True)
    ap.add_argument('--output_labels_dir', required=True)
    ap.add_argument('--target_classes', nargs='+', default=['box_open', 'box_close'],
                     help='채도를 낮출 대상 클래스 이름들 (기본: box_open box_close)')
    ap.add_argument('--desaturate_ratio', type=float, default=0.9,
                     help='0=원본 유지, 1=완전 무채색 (기본 0.9)')
    args = ap.parse_args()

    with open(args.data_yaml, encoding='utf-8') as f:
        yml = yaml.safe_load(f)
    names = yml['names']

    target_cls_ids = set()
    for cname in args.target_classes:
        if cname not in names:
            print(f"⚠️ '{cname}' 클래스가 data.yaml에 없습니다. 실제 클래스: {names}")
            continue
        target_cls_ids.add(names.index(cname))

    if not target_cls_ids:
        print("대상 클래스를 하나도 찾지 못해 종료합니다.")
        return

    print(f"채도를 낮출 대상 클래스 id: {target_cls_ids} ({args.target_classes})")

    os.makedirs(args.output_images_dir, exist_ok=True)
    os.makedirs(args.output_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    processed, skipped = 0, 0

    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        label_path = os.path.join(args.labels_dir, base + '.txt')
        img_path = os.path.join(args.images_dir, img_file)

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ 이미지 로드 실패: {img_file}")
            skipped += 1
            continue

        h, w = image.shape[:2]

        # 라벨이 없는 이미지는 원본 그대로 복사만
        if not os.path.exists(label_path):
            cv2.imwrite(os.path.join(args.output_images_dir, img_file), image)
            skipped += 1
            continue

        target_polygons = []
        with open(label_path, encoding='utf-8') as f:
            for line in f:
                vals = line.strip().split()
                if not vals:
                    continue
                cls_id = int(vals[0])
                if cls_id not in target_cls_ids:
                    continue
                coords = list(map(float, vals[1:]))
                if len(coords) < 6:  # bbox 형식(5개 값)은 폴리곤이 아니므로 건너뜀
                    continue
                target_polygons.append(polygon_norm_to_pixel(coords, w, h))

        if target_polygons:
            processed_image = desaturate_regions(image, target_polygons, args.desaturate_ratio)
            processed += 1
        else:
            processed_image = image  # 대상 클래스 없으면 원본 그대로

        cv2.imwrite(os.path.join(args.output_images_dir, img_file), processed_image)
        # 라벨은 좌표가 바뀌지 않으므로 그대로 복사
        shutil.copy(label_path, os.path.join(args.output_labels_dir, base + '.txt'))

    print(f"\n완료! 채도 처리된 이미지: {processed}장, 원본 그대로 복사: {skipped}장")
    print(f"저장 위치: {args.output_images_dir}")


if __name__ == '__main__':
    main()
