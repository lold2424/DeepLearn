# -*- coding: utf-8 -*-
"""
이미지 단위 클래스 등장 비율 계산.

인스턴스 개수(치킨 조각 몇 개)가 아니라
"이 클래스가 등장하는 이미지가 전체 중 몇 %인가"를 계산합니다.
한 이미지에 food가 2개 나와도 해당 이미지는 food 카운트 1회로만 처리합니다.

사용법:
    python count_images_per_class.py --labels_dir /path/to/labels --data_yaml /path/to/data.yaml

여러 split(train/valid/test)을 합쳐서 보고 싶으면 --labels_dir을 여러 번 줄 수 있습니다:
    python count_images_per_class.py \
        --labels_dir train/labels --labels_dir valid/labels --labels_dir test/labels \
        --data_yaml data.yaml
"""

import os
import argparse
import yaml
from collections import Counter


def count_images_per_class(labels_dirs, names):
    total_images = 0
    class_image_count = Counter()  # 클래스별로, 해당 클래스가 "등장한 이미지 수"

    for labels_dir in labels_dirs:
        if not os.path.exists(labels_dir):
            print(f"⚠️ 경로 없음, 건너뜀: {labels_dir}")
            continue

        for fname in os.listdir(labels_dir):
            if not fname.endswith('.txt'):
                continue
            total_images += 1

            classes_in_this_image = set()  # set이라 중복 자동 제거 -> 이미지당 클래스 1회만 카운트
            with open(os.path.join(labels_dir, fname), encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls_id = int(line.split()[0])
                    classes_in_this_image.add(cls_id)

            for cls_id in classes_in_this_image:
                class_image_count[cls_id] += 1

    return total_images, class_image_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels_dir', action='append', required=True,
                     help='labels 폴더 경로. 여러 번 지정하면 합쳐서 계산 (train/valid/test 등)')
    ap.add_argument('--data_yaml', required=True, help='클래스 이름 순서를 확인할 data.yaml 경로')
    args = ap.parse_args()

    with open(args.data_yaml, encoding='utf-8') as f:
        yml = yaml.safe_load(f)
    names = yml['names']
    print("클래스 순서:", names)

    total_images, class_image_count = count_images_per_class(args.labels_dir, names)

    print(f"\n전체 이미지 수: {total_images}장\n")
    print(f"{'클래스':<12}{'등장 이미지 수':>12}{'비율':>10}")
    print("-" * 36)
    for cls_id, cls_name in enumerate(names):
        count = class_image_count.get(cls_id, 0)
        ratio = count / total_images * 100 if total_images else 0
        print(f"{cls_name:<12}{count:>12}{ratio:>9.1f}%")


if __name__ == '__main__':
    main()
