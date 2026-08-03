# -*- coding: utf-8 -*-
"""
치킨 박스 세그멘테이션 모델 웹 데모 (Streamlit)

로컬에서 미리 확인:
    pip install streamlit ultralytics pillow
    streamlit run app.py

배포:
    1. 이 파일 + requirements.txt + best.pt(모델 파일)를 GitHub 레포에 push
    2. share.streamlit.io 접속 → GitHub 로그인 → 레포 선택 → Deploy
    3. 몇 분 뒤 공개 URL(예: https://your-app.streamlit.app) 생성됨
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(page_title="치킨 박스 상태 인식 데모", page_icon="🍗", layout="centered")

MODEL_PATH = "mini_seg_v3_best.pt"


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
    "배달 박스 사진을 업로드하면, AI가 **박스 개폐 상태와 내용물 유무**를 자동으로 인식합니다. "
    "환불 사기 여부를 CCTV로 일일이 확인하는 대신, 포장 완료 시점의 사진 한 장으로 즉시 검증할 수 있습니다."
)

conf_threshold = st.slider("Confidence 임계값", min_value=0.1, max_value=0.9, value=0.4, step=0.05)

uploaded_file = st.file_uploader("이미지를 업로드하세요 (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)

    with st.spinner("AI가 분석 중입니다..."):
        results = model.predict(source=np.array(image), conf=conf_threshold, verbose=False)
        result = results[0]

        # 결과 이미지 (박스/폴리곤이 그려진 상태)
        annotated = result.plot()  # numpy array (BGR)
        annotated_rgb = annotated[:, :, ::-1]  # BGR -> RGB

    with col2:
        st.subheader("AI 분석 결과")
        st.image(annotated_rgb, use_container_width=True)

    # 검출 결과 표
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

else:
    st.info("👆 위에서 이미지를 업로드하면 분석이 시작됩니다.")

st.markdown("---")
st.caption("YOLO11s-Seg · 학습 데이터 298장 · Mask mAP50-95 0.70")
