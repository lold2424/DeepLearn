# -*- coding: utf-8 -*-
"""
치킨 박스 세그멘테이션 모델 웹 데모 (Streamlit) - 이미지 + 영상 지원

로컬에서 미리 확인:
    pip install streamlit ultralytics pillow opencv-python-headless
    streamlit run app.py

배포:
    1. 이 파일 + requirements.txt + packages.txt + best.pt를 GitHub 레포에 push
    2. share.streamlit.io 접속 → GitHub 로그인 → 레포 선택 → Deploy
"""

import os
import tempfile
import threading

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(page_title="치킨 박스 상태 인식 데모", page_icon="🍗", layout="centered")

MODEL_PATH = "best.pt"

IMAGE_EXTS = ("jpg", "jpeg", "png")
VIDEO_EXTS = ("mp4", "mov", "avi")

# 여러 사용자가 동시에 무거운 작업(특히 영상)을 돌려서
# 서버 자원이 겹치지 않도록 하는 전역 잠금 (같은 프로세스 안 모든 세션이 공유)
_inference_lock = threading.Lock()


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# ============================================================
# 화면 구성
# ============================================================
st.title("🍗 치킨 배달 박스 상태 인식 AI")
st.caption("YOLO11-Segmentation 기반 — box_open / box_close / food / table 인식")

st.markdown(
    "배달 박스 사진 또는 짧은 영상을 업로드하면, AI가 **박스 개폐 상태와 내용물 유무**를 자동으로 인식합니다. "
    "환불 사기 여부를 CCTV로 일일이 확인하는 대신, 포장 완료 시점의 기록으로 즉시 검증할 수 있습니다."
)

conf_threshold = st.slider("Confidence 임계값", min_value=0.1, max_value=0.9, value=0.4, step=0.05)

uploaded_file = st.file_uploader(
    "이미지(jpg, png) 또는 짧은 영상(mp4, mov, avi)을 업로드하세요",
    type=list(IMAGE_EXTS) + list(VIDEO_EXTS),
)

if uploaded_file is not None:
    ext = uploaded_file.name.split(".")[-1].lower()

    # ------------------------------------------------------------
    # 이미지 처리
    # ------------------------------------------------------------
    if ext in IMAGE_EXTS:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 이미지")
            st.image(image, use_container_width=True)

        with st.spinner("AI가 분석 중입니다..."):
            if not _inference_lock.acquire(blocking=False):
                st.info("⏳ 다른 사용자가 분석 중입니다. 순서가 되면 자동으로 처리됩니다...")
                _inference_lock.acquire(blocking=True)
            try:
                results = model.predict(source=np.array(image), conf=conf_threshold, verbose=False)
                result = results[0]
                annotated_rgb = result.plot()[:, :, ::-1]  # BGR -> RGB
            finally:
                _inference_lock.release()

        with col2:
            st.subheader("AI 분석 결과")
            st.image(annotated_rgb, use_container_width=True)

        st.subheader("검출 상세")
        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            rows = [{"클래스": names[c], "신뢰도": f"{conf:.1%}"} for c, conf in zip(cls_ids, confs)]
            st.table(rows)

            detected_classes = set(names[c] for c in cls_ids)
            if "food" in detected_classes and ("box_open" in detected_classes or "box_close" in detected_classes):
                st.success("✅ 박스와 내용물이 함께 검출되었습니다 — 정상 포장으로 판단됩니다.")
            elif "table" in detected_classes and "food" not in detected_classes:
                st.warning("⚠️ 내용물 없이 빈 트레이만 검출되었습니다 — 확인이 필요할 수 있습니다.")
        else:
            st.info("검출된 객체가 없습니다. Confidence 임계값을 낮춰보세요.")

    # ------------------------------------------------------------
    # 영상 처리
    # ------------------------------------------------------------
    elif ext in VIDEO_EXTS:
        st.warning("⏳ 영상은 프레임 단위로 처리되어 시간이 걸릴 수 있습니다 (CPU 서버 기준, 짧은 영상 권장).")

        # 업로드된 영상을 임시 파일로 저장 (cv2는 파일 경로가 필요함)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_in:
            tmp_in.write(uploaded_file.read())
            in_path = tmp_in.name

        if not _inference_lock.acquire(blocking=False):
            st.info("⏳ 다른 사용자가 영상을 분석 중입니다. 순서가 되면 자동으로 시작됩니다...")
            _inference_lock.acquire(blocking=True)

        try:
            cap = cv2.VideoCapture(in_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            progress = st.progress(0, text="영상 분석 중...")
            frame_idx = 0
            detected_summary = set()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]
                annotated = result.plot()  # BGR, cv2와 동일 색공간이라 그대로 사용
                writer.write(annotated)

                if result.boxes is not None and len(result.boxes) > 0:
                    names = result.names
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    detected_summary.update(names[c] for c in cls_ids)

                frame_idx += 1
                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0), text=f"분석 중... ({frame_idx}/{total_frames} 프레임)")

            cap.release()
            writer.release()
            progress.empty()
        finally:
            _inference_lock.release()

        st.subheader("AI 분석 결과 영상")
        with open(out_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button("결과 영상 다운로드", data=video_bytes, file_name="result.mp4", mime="video/mp4")

        st.subheader("영상 전체에서 검출된 클래스")
        if detected_summary:
            st.write(", ".join(sorted(detected_summary)))
        else:
            st.info("검출된 객체가 없습니다. Confidence 임계값을 낮춰보세요.")

        # 임시 파일 정리
        os.remove(in_path)
        os.remove(out_path)

else:
    st.info("👆 위에서 이미지 또는 영상을 업로드하면 분석이 시작됩니다.")

st.markdown("---")
st.caption("YOLO11s-Seg · 학습 데이터 298장 · Mask mAP50-95 0.70")
