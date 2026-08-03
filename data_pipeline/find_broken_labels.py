# -*- coding: utf-8 -*-
"""
폴리곤 점이 3개 미만(면적을 이룰 수 없는 깨진 라벨)인 인스턴스를 찾아
정확한 파일 경로 + 줄 번호 + 원본 라인 내용까지 출력.

사용법:
    python find_broken_labels.py --labels_dir train/labels --labels_dir valid/labels --labels_dir test/labels --data_yaml data.yaml
"""

import os
import argparse
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels_dir', action='append', required=True)
    ap.add_argument('--data_yaml', required=True)
    ap.add_argument('--min_points', type=int, default=3, help='이 값 미만인 점 개수를 깨진 라벨로 간주 (기본 3)')
    args = ap.parse_args()

    with open(args.data_yaml, encoding='utf-8') as f:
        yml = yaml.safe_load(f)
    names = yml['names']

    broken = []

    for labels_dir in args.labels_dir:
        if not os.path.exists(labels_dir):
            print(f"⚠️ 경로 없음: {labels_dir}")
            continue

        for fname in os.listdir(labels_dir):
            if not fname.endswith('.txt'):
                continue
            full_path = os.path.join(labels_dir, fname)

            with open(full_path, encoding='utf-8') as f:
                for line_no, line in enumerate(f, start=1):
                    vals = line.strip().split()
                    if not vals:
                        continue
                    cls_id = int(vals[0])
                    num_coords = len(vals) - 1

                    if num_coords == 4:
                        # 5개 값(class + 4개) = bbox 형식이라 폴리곤 점 개념이 아님, 정상으로 취급
                        continue

                    num_points = num_coords // 2
                    if num_points < args.min_points:
                        cls_name = names[cls_id] if cls_id < len(names) else f"id={cls_id}"
                        broken.append({
                            'path': full_path,
                            'line_no': line_no,
                            'class': cls_name,
                            'num_points': num_points,
                            'raw_line': line.strip(),
                        })

    if not broken:
        print("깨진 라벨을 찾지 못했습니다 (모두 정상).")
        return

    print(f"총 {len(broken)}건의 깨진 라벨 발견:\n")
    for b in broken:
        print(f"파일: {b['path']}")
        print(f"  줄 번호: {b['line_no']}")
        print(f"  클래스: {b['class']}")
        print(f"  점 개수: {b['num_points']}")
        print(f"  원본 내용: {b['raw_line'][:80]}{'...' if len(b['raw_line']) > 80 else ''}")
        print()

    print("=== 정리: 파일 경로만 모아보기 (엑셀/메모장에 붙여넣기용) ===")
    for b in broken:
        print(b['path'])


if __name__ == '__main__':
    main()
