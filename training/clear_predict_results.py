# -*- coding: utf-8 -*-
"""
predict() 실행 결과 폴더(/content/runs/segment/predict)를 삭제하는 코드.
재예측 전에 이전 결과가 남아있으면 섞이거나 exist_ok로 덮어써지는 게
헷갈릴 수 있어서, 깨끗하게 지우고 다시 시작할 때 사용.
"""

import shutil
import os

PREDICT_DIR = '/content/runs/segment/predict'

if os.path.exists(PREDICT_DIR):
    shutil.rmtree(PREDICT_DIR)
    print(f"삭제 완료: {PREDICT_DIR}")
else:
    print(f"이미 없음: {PREDICT_DIR}")

# 만약 predict, predict2, predict3 ... 여러 개가 쌓여있다면 한 번에 다 지우고 싶을 때:
# import glob
# for d in glob.glob('/content/runs/segment/predict*'):
#     shutil.rmtree(d)
#     print(f"삭제 완료: {d}")
