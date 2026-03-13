import argparse
import os
import sys
from typing import Tuple

import cv2
import numpy as np


def put_info(img: np.ndarray, text: str, org: Tuple[int, int] = (10, 25)) -> np.ndarray:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_to_match(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def process_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 표기
    labeled_orig = frame.copy()
    labeled_gray = to_bgr(gray.copy())
    labeled_edges = to_bgr(edges.copy())

    put_info(labeled_orig, "Original")
    put_info(labeled_gray, "Grayscale")
    put_info(labeled_edges, "Canny Edges")

    # 동일 크기/채널로 정렬 후 가로 스택
    h, w = labeled_orig.shape[:2]
    labeled_gray = resize_to_match(labeled_gray, (h, w))
    labeled_edges = resize_to_match(labeled_edges, (h, w))
    stacked = np.hstack([labeled_orig, labeled_gray, labeled_edges])
    return stacked


def run_webcam(device: int = 0) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[ERROR] Webcam not accessible (device {device}).")
        return

    cv2.namedWindow("OpenCV Demo", cv2.WINDOW_NORMAL)
    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame. Exiting.")
            break
        view = process_frame(frame)
        cv2.imshow("OpenCV Demo", view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(path: str) -> None:
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        return
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Failed to read image: {path}")
        return
    view = process_frame(img)
    cv2.namedWindow("OpenCV Demo", cv2.WINDOW_NORMAL)
    cv2.imshow("OpenCV Demo", view)
    print("[INFO] Press any key (window focused) to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(path: str) -> None:
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {path}")
        return
    cv2.namedWindow("OpenCV Demo", cv2.WINDOW_NORMAL)
    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = process_frame(frame)
        cv2.imshow("OpenCV Demo", view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenCV quick starter")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--webcam", action="store_true", help="Use webcam (default)")
    g.add_argument("--image", type=str, help="Path to an image file")
    g.add_argument("--video", type=str, help="Path to a video file")
    p.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0)")
    return p.parse_args()


def main() -> None:
    try:
        print(f"OpenCV version: {cv2.__version__}")
    except Exception:
        print("[WARN] Unable to read cv2.__version__")

    args = parse_args()
    if args.image:
        run_image(args.image)
    elif args.video:
        run_video(args.video)
    else:
        # default to webcam if nothing specified
        run_webcam(args.device)


if __name__ == "__main__":
    sys.exit(main())

