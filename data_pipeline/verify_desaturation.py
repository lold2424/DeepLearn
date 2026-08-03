# -*- coding: utf-8 -*-
"""
채도 처리 전/후 이미지에서 box 영역의 평균 채도(Saturation) 값을 비교.
숫자가 확실히 줄어들었으면 처리가 제대로 된 것 (눈으로 티가 안 나도 정상일 수 있음).

사용법:
    python verify_desaturation.py \
        --original_image /content/yolo11_v5/train/images/파일명.jpg \
        --processed_image /content/yolo11_v5/train_desat/images/파일명.jpg \
        --label_path /content/yolo11_v5/train/labels/파일명.txt \
        --target_classes 0 1
"""

import argparse
import cv2
import numpy as np


def polygon_norm_to_pixel(coords, w, h):
    pts = []
    for i in range(0, len(coords), 2):
        pts.append((coords[i] * w, coords[i + 1] * h))
    return pts


def get_box_mask(label_path, target_cls_ids, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            cls_id = int(vals[0])
            if cls_id not in target_cls_ids:
                continue
            coords = list(map(float, vals[1:]))
            if len(coords) < 6:
                continue
            pts = np.array(polygon_norm_to_pixel(coords, w, h), dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original_image', required=True)
    ap.add_argument('--processed_image', required=True)
    ap.add_argument('--label_path', required=True)
    ap.add_argument('--target_classes', nargs='+', type=int, default=[0, 1],
                     help='box_open/box_close의 class id (기본 0 1)')
    args = ap.parse_args()

    orig = cv2.imread(args.original_image)
    proc = cv2.imread(args.processed_image)

    if orig is None or proc is None:
        print("이미지를 읽지 못했습니다. 경로를 확인하세요.")
        return

    h, w = orig.shape[:2]
    target_cls_ids = set(args.target_classes)
    mask = get_box_mask(args.label_path, target_cls_ids, w, h)

    if mask.sum() == 0:
        print("이 이미지에는 대상 클래스(box_open/box_close) 영역이 없습니다.")
        return

    orig_hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    proc_hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)

    orig_sat_in_box = orig_hsv[..., 1][mask == 255]
    proc_sat_in_box = proc_hsv[..., 1][mask == 255]

    print(f"박스 영역 픽셀 수: {mask.sum() // 255}")
    print(f"원본 평균 채도(S):     {orig_sat_in_box.mean():.2f} / 255")
    print(f"처리 후 평균 채도(S):  {proc_sat_in_box.mean():.2f} / 255")
    print(f"감소율: {(1 - proc_sat_in_box.mean() / (orig_sat_in_box.mean() + 1e-6)) * 100:.1f}%")

    if proc_sat_in_box.mean() < orig_sat_in_box.mean() * 0.5:
        print("\n✅ 채도가 확실히 줄어들었습니다. 코드는 정상 동작 중입니다.")
        print("   (원본 채도가 애초에 낮은 이미지라 눈으로 티가 안 날 뿐입니다.)")
    else:
        print("\n⚠️ 채도가 별로 줄지 않았습니다. 라벨 매칭이나 폴리곤 좌표를 다시 확인해보세요.")


if __name__ == '__main__':
    main()
