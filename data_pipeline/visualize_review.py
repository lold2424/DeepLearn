# -*- coding: utf-8 -*-
"""
review_needed.csv에 나온 의심 항목들을 실제 이미지 위에 그려서
한 폴더에 모아 저장 -> 눈으로 빠르게 훑어보고 판단하기 위한 스크립트

사용법 (Colab):
    python visualize_review.py --split train
    python visualize_review.py --split valid

전제:
- /content/{split}/images 에 원본 이미지가 있어야 함
- /content/{split}/review_needed.csv 가 있어야 함 (convert_polygon_labels.py 실행 결과)

결과:
- /content/{split}/review_check/ 폴더에 각 의심 파일마다
  "박스이름_check.jpg" 로 저장됨 (초록=box, 빨강=input)
"""

import os
import csv
import argparse
from PIL import Image, ImageDraw


def draw_box(draw, bbox_str, w, h, color, label):
    xc, yc, bw, bh = map(float, bbox_str.split(','))
    xmin = (xc - bw / 2) * w
    xmax = (xc + bw / 2) * w
    ymin = (yc - bh / 2) * h
    ymax = (yc + bh / 2) * h
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
    draw.text((xmin, max(0, ymin - 15)), label, fill=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'valid'])
    args = ap.parse_args()

    base = f'/content/{args.split}'
    csv_path = os.path.join(base, 'review_needed.csv')
    img_dir = os.path.join(base, 'images')
    out_dir = os.path.join(base, 'review_check')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"csv 없음: {csv_path}")
        return

    with open(csv_path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    print(f"{len(rows)}건 시각화 시작...")

    for row in rows:
        txt_name = row['file']
        base_name = os.path.splitext(txt_name)[0]

        img_path = None
        for ext in ('.jpg', '.jpeg', '.png'):
            candidate = os.path.join(img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"이미지 못 찾음: {base_name}")
            continue

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        draw = ImageDraw.Draw(img)

        draw_box(draw, row['box_bbox'], w, h, (0, 255, 0), f"box(IoU={row['iou']})")
        draw_box(draw, row['input_bbox'], w, h, (255, 0, 0), "input")

        out_path = os.path.join(out_dir, f"{base_name}_check.jpg")
        img.save(out_path)

    print(f"완료! 결과는 여기 저장됨: {out_dir}")
    print("이 폴더를 좌측 파일 탐색기에서 열어 하나씩 눈으로 확인하세요.")


if __name__ == '__main__':
    main()
