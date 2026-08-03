# -*- coding: utf-8 -*-
"""
폴리곤(세그멘테이션) 형식 라벨을 bbox(x_center y_center w h) 형식으로 일괄 변환하고,
box 클래스(0)로 변환된 결과가 input 클래스(1)와 많이 겹치면(=사실상 같은 대상일 가능성)
자동으로 의심 목록에 추가해주는 스크립트.

사용법 (Colab 예시):
    python convert_polygon_labels.py --labels_dir /content/train/labels
    python convert_polygon_labels.py --labels_dir /content/valid/labels

동작:
1. labels_dir 안의 모든 .txt 파일을 검사
2. 한 줄이 5개 값(class x y w h)이 아니면 폴리곤으로 간주 -> min/max bbox로 변환
3. 변환 후 같은 이미지 안의 다른 박스와 IoU가 IOU_THRESHOLD(기본 0.5) 이상이면
   "review_needed.csv"에 기록 (박스로 오인된 게 아니라 input과 같은 대상일 가능성 있음)
4. 변환된 결과는 원본을 덮어쓰지 않고 별도 폴더(labels_converted)에 저장 -> 안전하게 검토 후 교체 가능
"""

import os
import argparse
import csv


def polygon_to_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return xc, yc, w, h


def to_xyxy(b):
    xc, yc, w, h = b
    return xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2


def intersection_area(b1, b2):
    x1min, y1min, x1max, y1max = to_xyxy(b1)
    x2min, y2min, x2max, y2max = to_xyxy(b2)

    inter_xmin = max(x1min, x2min)
    inter_ymin = max(y1min, y2min)
    inter_xmax = min(x1max, x2max)
    inter_ymax = min(y1max, y2max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    return inter_w * inter_h


def bbox_area(b):
    x1min, y1min, x1max, y1max = to_xyxy(b)
    return max(0.0, x1max - x1min) * max(0.0, y1max - y1min)


def bbox_iou(b1, b2):
    inter_area = intersection_area(b1, b2)
    area1 = bbox_area(b1)
    area2 = bbox_area(b2)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def containment_ratio(box, other):
    """box 면적 중 other와 겹치는 비율. box가 other를 얼마나 '뒤덮고 있는지'를 봄.
    box가 크고 other가 그 안에 쏙 들어가는 경우에도 1에 가깝게 나옴 -> IoU의 약점 보완."""
    inter_area = intersection_area(box, other)
    area = bbox_area(box)
    if area <= 0:
        return 0.0
    return inter_area / area


def process_file(txt_path):
    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    parsed = []  # (cls_id, xc, yc, w, h, was_polygon)
    for line in lines:
        vals = list(map(float, line.split()))
        cls_id = int(vals[0])
        if len(vals) == 5:
            _, xc, yc, w, h = vals
            parsed.append((cls_id, xc, yc, w, h, False))
        else:
            xc, yc, w, h = polygon_to_bbox(vals[1:])
            parsed.append((cls_id, xc, yc, w, h, True))

    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels_dir', required=True, help='labels 폴더 경로 (예: /content/train/labels)')
    ap.add_argument('--containment_threshold', type=float, default=0.5,
                     help='box 면적 중 input과 겹치는 비율 기준값 (기본 0.5 = 50%% 이상 겹치면 의심)')
    ap.add_argument('--input_class_id', type=int, default=1, help='input 클래스의 id (기본 1)')
    ap.add_argument('--box_class_id', type=int, default=0, help='box 클래스의 id (기본 0)')
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(args.labels_dir.rstrip('/')), 'labels_converted')
    os.makedirs(out_dir, exist_ok=True)

    review_rows = []

    txt_files = [f for f in os.listdir(args.labels_dir) if f.endswith('.txt')]
    print(f"총 {len(txt_files)}개 라벨 파일 검사 시작...")

    polygon_file_count = 0

    for fname in txt_files:
        path = os.path.join(args.labels_dir, fname)
        parsed = process_file(path)

        had_polygon = any(p[5] for p in parsed)
        if had_polygon:
            polygon_file_count += 1

        # box(변환된 것 포함) vs input 간 IoU 체크
        boxes_of_target_cls = [p for p in parsed if p[0] == args.box_class_id]
        boxes_of_input_cls = [p for p in parsed if p[0] == args.input_class_id]

        for b in boxes_of_target_cls:
            if not b[5]:
                continue  # 원래부터 bbox였던 건 검사 대상 아님
            for i in boxes_of_input_cls:
                iou = bbox_iou(b[1:5], i[1:5])
                ratio = containment_ratio(b[1:5], i[1:5])  # box 면적 중 input과 겹치는 비율
                if ratio >= args.containment_threshold:
                    review_rows.append({
                        'file': fname,
                        'box_bbox': f"{b[1]:.4f},{b[2]:.4f},{b[3]:.4f},{b[4]:.4f}",
                        'input_bbox': f"{i[1]:.4f},{i[2]:.4f},{i[3]:.4f},{i[4]:.4f}",
                        'iou': f"{iou:.3f}",
                        'containment_ratio': f"{ratio:.3f}",
                    })

        # 변환 결과 저장 (원본 보존, 별도 폴더)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, 'w') as f:
            for cls_id, xc, yc, w, h, _ in parsed:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"폴리곤이 포함된 파일: {polygon_file_count}개")
    print(f"변환된 라벨은 여기 저장됨: {out_dir}")

    if review_rows:
        csv_path = os.path.join(os.path.dirname(args.labels_dir.rstrip('/')), 'review_needed.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'box_bbox', 'input_bbox', 'iou', 'containment_ratio'])
            writer.writeheader()
            writer.writerows(review_rows)
        print(f"\n⚠️ box 면적의 {args.containment_threshold*100:.0f}% 이상이 input과 겹치는 의심 항목 {len(review_rows)}건 발견")
        print(f"   목록: {csv_path}")
        print("   -> 이 목록에 있는 파일들은 box 라벨이 실제로는 input(내용물)과 같은 대상일 가능성이 높으니 직접 확인 필요")
    else:
        print("\nbox와 input이 크게 겹치는 항목은 발견되지 않았습니다.")


if __name__ == '__main__':
    main()
