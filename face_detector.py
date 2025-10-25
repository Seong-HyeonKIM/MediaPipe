#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 MediaPipe 초기화 (Face Detection)
# =========================================
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# model_selection:
#   0 = 근거리(셀카/웹캠), 1 = 원거리(멀리 있는 얼굴)
face = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# =========================================
# 📸 카메라/영상 연결
# =========================================
cap = cv2.VideoCapture(0)             # 기본 카메라 사용 시
# cap = cv2.VideoCapture("video.mp4")  # 동영상 파일 사용 시

print("📷 얼굴 탐지를 시작합니다 — ESC를 누르면 종료됩니다.")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("⚠️ 프레임을 읽지 못했습니다. 카메라/파일 경로를 확인하세요.")
        break

    # 보기 편하도록 좌우 반전 (셀카 뷰)
    frame = cv2.flip(frame, 1)

    # BGR → RGB 변환 (MediaPipe는 RGB 입력)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 탐지 수행
    result = face.process(rgb)

    # =========================================
    # 🙂 탐지 결과 그리기
    # - bounding box + 6개 keypoints(눈, 코, 입 등)
    # =========================================
    if result.detections:
        for det in result.detections:
            mp_drawing.draw_detection(frame, det)

    # 화면 표시
    cv2.imshow("🙂 MediaPipe Face Detector", frame)

    # ESC 키(27)로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()