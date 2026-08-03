# -*- coding: utf-8 -*-
"""
실전용 배치 예측 스크립트.

테스트용 predict()와의 차이점:
1. 이미지 하나가 깨져 있어도 전체가 멈추지 않도록 개별 예외 처리
2. 이미지/영상을 명시적으로 구분해서 처리 (섞여서 멈추는 문제 방지)
3. 예측 결과를 CSV 로그로 남겨서 나중에 검색/분석 가능
4. 대량 이미지 처리 시 메모리 문제 방지를 위해 stream=True 사용

사용법:
    python run_inference_batch.py \
        --model /content/drive/MyDrive/model_weights/mini_seg_v3_final_best.pt \
        --source_dir /content/drive/MyDrive/Image/unlabeled \
        --output_dir /content/drive/MyDrive/inference_runs/2026-08-02 \
        --conf 0.4
"""

import os
import csv
import glob
import argparse
from datetime import datetime

import torch
from ultralytics import YOLO


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv')


def get_device():
    """GPU가 있으면 0번 GPU, 없으면 CPU를 자동 선택."""
    if torch.cuda.is_available():
        print(f"GPU 감지됨: {torch.cuda.get_device_name(0)} → device=0 사용")
        return 0
    print("GPU 없음 → device='cpu'로 실행 (속도가 느릴 수 있습니다)")
    return 'cpu'


def collect_files(source_dir):
    images, videos = [], []
    for f in os.listdir(source_dir):
        ext = os.path.splitext(f)[1].lower()
        full = os.path.join(source_dir, f)
        if ext in IMAGE_EXTS:
            images.append(full)
        elif ext in VIDEO_EXTS:
            videos.append(full)
    return images, videos


def run_image_batch(model, image_paths, output_dir, conf, device):
    """이미지들을 안전하게 배치 예측하고, 결과를 로그 리스트로 반환."""
    log_rows = []

    if not image_paths:
        return log_rows

    # stream=True: 대량 이미지도 메모리 누적 없이 하나씩 처리
    results_gen = model.predict(
        source=image_paths,
        save=True,
        save_txt=True,
        conf=conf,
        project=output_dir,
        name='predictions',
        exist_ok=True,
        device=device,
        stream=True,
    )

    for result in results_gen:
        img_path = result.path
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            if result.boxes is None or len(result.boxes) == 0:
                log_rows.append({
                    'timestamp': timestamp,
                    'file': os.path.basename(img_path),
                    'detected_classes': '',
                    'confidences': '',
                    'num_detections': 0,
                    'status': 'no_detection',
                })
                continue

            names = result.names
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            detected = [names[c] for c in cls_ids]
            log_rows.append({
                'timestamp': timestamp,
                'file': os.path.basename(img_path),
                'detected_classes': ';'.join(detected),
                'confidences': ';'.join(f'{c:.3f}' for c in confs),
                'num_detections': len(detected),
                'status': 'ok',
            })
        except Exception as e:
            # 개별 이미지 처리 중 오류가 나도 전체는 계속 진행
            log_rows.append({
                'timestamp': timestamp,
                'file': os.path.basename(img_path),
                'detected_classes': '',
                'confidences': '',
                'num_detections': 0,
                'status': f'error: {e}',
            })

    return log_rows


def run_video_batch(model, video_paths, output_dir, conf, device):
    """영상은 이미지와 분리해서 별도 처리 (예측 결과 영상만 저장, 프레임별 로그는 생략)."""
    log_rows = []
    for vid_path in video_paths:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            model.predict(
                source=vid_path,
                save=True,
                conf=conf,
                project=output_dir,
                name='video_predictions',
                exist_ok=True,
                device=device,
            )
            log_rows.append({
                'timestamp': timestamp, 'file': os.path.basename(vid_path),
                'detected_classes': '(영상 - 상세 로그 생략)', 'confidences': '',
                'num_detections': '', 'status': 'ok',
            })
        except Exception as e:
            log_rows.append({
                'timestamp': timestamp, 'file': os.path.basename(vid_path),
                'detected_classes': '', 'confidences': '', 'num_detections': '',
                'status': f'error: {e}',
            })
    return log_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='best.pt 경로')
    ap.add_argument('--source_dir', required=True, help='예측할 이미지/영상이 있는 폴더')
    ap.add_argument('--output_dir', required=True, help='결과 저장 폴더 (드라이브 경로 권장)')
    ap.add_argument('--conf', type=float, default=0.4)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"모델 로드: {args.model}")
    model = YOLO(args.model)
    device = get_device()

    images, videos = collect_files(args.source_dir)
    print(f"이미지 {len(images)}개, 영상 {len(videos)}개 발견")

    all_logs = []
    all_logs.extend(run_image_batch(model, images, args.output_dir, args.conf, device))
    all_logs.extend(run_video_batch(model, videos, args.output_dir, args.conf, device))

    # 로그 CSV로 저장
    log_path = os.path.join(args.output_dir, f"inference_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    if all_logs:
        with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_logs[0].keys()))
            writer.writeheader()
            writer.writerows(all_logs)

    ok_count = sum(1 for r in all_logs if r['status'] == 'ok')
    err_count = sum(1 for r in all_logs if 'error' in str(r['status']))
    no_det_count = sum(1 for r in all_logs if r['status'] == 'no_detection')

    print(f"\n=== 완료 ===")
    print(f"정상 처리: {ok_count}개 / 미검출: {no_det_count}개 / 오류: {err_count}개")
    print(f"로그 저장 위치: {log_path}")
    print(f"결과 이미지 저장 위치: {os.path.join(args.output_dir, 'predictions')}")


if __name__ == '__main__':
    main()
