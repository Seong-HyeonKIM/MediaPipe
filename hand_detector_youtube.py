#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
유튜브 링크의 영상을 자동으로 다운로드해
MediaPipe Hands로 손을 탐지/시각화하는 예제.

사용법:
    python hand_detector_youtube.py "https://www.youtube.com/watch?v=영상ID"
"""

import os
import sys
import cv2
import mediapipe as mp
from yt_dlp import YoutubeDL

# ============================
# 🔽 유튜브 다운로드 함수
# ============================
def download_youtube_video(url: str, out_dir: str = "downloads", out_name: str = "video.mp4") -> str:
    """
    유튜브 영상을 MP4로 다운로드하고 파일 경로를 반환합니다.
    동일 이름이 있으면 덮어씁니다.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    ydl_opts = {
        # mp4 우선, 안 되면 가장 좋은 포맷
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best",
        "merge_output_format": "mp4",
        "outtmpl": out_path,
        "overwrites": True,
        "quiet": True,
        "noprogress": False,
    }

    print(f"⬇️  유튜브 영상 다운로드 중...\nURL: {url}")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(out_path):
        raise FileNotFoundError("동영상 다운로드에 실패했습니다.")
    print(f"✅ 다운로드 완료: {out_path}")
    return out_path


# ============================
# ✋ MediaPipe Hands 초기화
# ============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def run_hand_detector(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠️ 영상을 열 수 없습니다. 경로를 확인하세요:", video_path)
        return

    # 성능 최적화 옵션 (필요 시 조정)
    hands = mp_hands.Hands(
        static_image_mode=False,        # 동영상 스트림
        max_num_hands=2,               # 최대 손 개수
        min_detection_confidence=0.5,  # 탐지 신뢰도
        min_tracking_confidence=0.5    # 추적 신뢰도
    )

    print("▶️  재생/분석 시작 — 창에서 Q 또는 ESC로 종료")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 손 탐지
        result = hands.process(rgb)

        # 결과 그리기
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

        cv2.imshow("🖐 MediaPipe Hand Detector (YouTube)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


def main():
    if len(sys.argv) < 2:
        print("사용법: python hand_detector_youtube.py \"https://www.youtube.com/watch?v=...\"")
        sys.exit(1)

    url = sys.argv[1].strip()
    try:
        video_path = download_youtube_video(url)
        run_hand_detector(video_path)
    except Exception as e:
        print("에러:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
