import argparse
import os
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np


Point = Tuple[int, int]


def project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def default_roi_path() -> str:
    return os.path.join(project_root(), "tracking", "openfield", "roi.json")


def default_zones_path() -> str:
    return os.path.join(project_root(), "tracking", "openfield", "zones.json")


def load_roi(path: str) -> Optional[List[Point]]:
    if not os.path.isfile(path):
        return None
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = data.get("points")
        if not pts:
            return None
        return [(int(p[0]), int(p[1])) for p in pts]
    except Exception:
        return None


def load_zones(path: str) -> Optional[List[dict]]:
    if not os.path.isfile(path):
        return None
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        zones = data.get("zones")
        if not zones:
            return None
        out = []
        for z in zones:
            name = str(z.get("name", "zone"))
            pts = z.get("points", [])
            if pts and len(pts) >= 3:
                out.append({"name": name, "points": [(int(p[0]), int(p[1])) for p in pts]})
        return out or None
    except Exception:
        return None


def draw_overlay(frame: np.ndarray, roi: Optional[List[Point]], zones: Optional[List[dict]]) -> np.ndarray:
    vis = frame.copy()
    overlay = np.zeros_like(vis)
    if roi:
        cv2.polylines(overlay, [np.array(roi, np.int32)], True, (0, 255, 255), 2)
    if zones:
        for z in zones:
            poly = np.array(z["points"], np.int32)
            # fill light and outline for visibility
            cv2.fillPoly(overlay, [poly], (0, 165, 255))
            cv2.polylines(overlay, [poly], True, (0, 0, 0), 3)
            cv2.polylines(overlay, [poly], True, (255, 255, 255), 1)
            # label near centroid
            M = cv2.moments(poly)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, str(z["name"]), (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2, cv2.LINE_AA)
    vis = cv2.addWeighted(vis, 1.0, overlay, 0.35, 0)
    return vis


def timestamp_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _open_camera(device: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(device)
    if cap.isOpened():
        return cap
    # Windows-specific fallback
    try:
        _ = cv2.CAP_DSHOW  # type: ignore[attr-defined]
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)  # type: ignore[attr-defined]
        if cap.isOpened():
            return cap
    except Exception:
        pass
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="OpenField arena overlay + optional recording")
    p.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0)")
    p.add_argument("--fps", type=float, default=30.0, help="Target FPS for recording (default: 30)")
    p.add_argument("--roi", type=str, default=default_roi_path(), help="ROI JSON path")
    p.add_argument("--zones", type=str, default=default_zones_path(), help="Zones JSON path")
    p.add_argument("--out", type=str, default=None, help="Output video path (default: tracking/openfield/rec_YYYYmmdd-HHMMSS.mp4)")
    p.add_argument("--record", action="store_true", help="Start recording immediately")
    p.add_argument("--with-overlay", action="store_true", help="Record with overlay drawn on frames")
    p.add_argument("--width", type=int, default=0, help="Request camera width (e.g., 1280)")
    p.add_argument("--height", type=int, default=0, help="Request camera height (e.g., 720)")
    p.add_argument(
        "--roi-ref-size",
        type=str,
        default="",
        help="Reference size WxH used when ROI/Zones were created (e.g., 1920x1080) to rescale to current camera size",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=["any", "dshow", "msmf"],
        default="any",
        help="Capture backend on Windows (any/dshow/msmf)",
    )
    args = p.parse_args()

    roi = load_roi(args.roi)
    zones = load_zones(args.zones)

    print(f"[INFO] Opening device {args.device} …")
    cap = _open_camera(args.device)
    # If user requested a specific backend, reopen with it
    if args.backend != "any":
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        be = getattr(cv2, 'CAP_DSHOW', 0) if args.backend == 'dshow' else getattr(cv2, 'CAP_MSMF', 0)
        cap = cv2.VideoCapture(args.device, be) if be != 0 else cv2.VideoCapture(args.device)
    if cap is None or not cap.isOpened():
        print(f"[ERROR] Webcam not accessible (device {args.device}).")
        return

    # Figure out size and apply requested resolution if provided
    if args.width and args.height:
        # Try enabling MJPG for higher resolutions on Windows webcams
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in <= 1:
        fps_in = args.fps

    print(f"[INFO] Negotiated camera size: {w}x{h} @ ~{fps_in:.1f} FPS")
    # If a specific size was requested but not honored and no backend fixed yet,
    # try alternative Windows backends automatically.
    if args.width and args.height and (w != args.width or h != args.height) and args.backend == "any":
        for name, be in (("dshow", getattr(cv2, 'CAP_DSHOW', 0)), ("msmf", getattr(cv2, 'CAP_MSMF', 0))):
            print(f"[INFO] Reopening with backend {name} to request {args.width}x{args.height} …")
            try:
                cap.release()
            except Exception:
                pass
            cap = cv2.VideoCapture(args.device, be) if be != 0 else cv2.VideoCapture(args.device)
            if not cap.isOpened():
                continue
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            print(f"[INFO] Negotiated camera size: {w}x{h}")
            if w == args.width and h == args.height:
                break
        else:
            print(f"[WARN] Could not set requested size {args.width}x{args.height}; using {w}x{h}.")

    # Optional: rescale ROI/Zones from a reference size to current camera size
    def _parse_wh(s: str) -> Optional[Tuple[int, int]]:
        try:
            if not s:
                return None
            s = s.lower().strip().replace(" ", "")
            if "x" not in s:
                return None
            a, b = s.split("x", 1)
            return int(a), int(b)
        except Exception:
            return None

    ref_arg = getattr(args, 'roi_ref_size', '')
    ref = _parse_wh(ref_arg)

    def _scale_points(pts: List[Point], src: Tuple[int, int], dst: Tuple[int, int]) -> List[Point]:
        sx = dst[0] / max(1, src[0])
        sy = dst[1] / max(1, src[1])
        return [(int(round(x * sx)), int(round(y * sy))) for (x, y) in pts]

    if ref and (roi or zones):
        src_wh = (ref[0], ref[1])
        dst_wh = (w, h)
        if roi:
            roi = _scale_points(roi, src_wh, dst_wh)
        if zones:
            scaled = []
            for z in zones:
                scaled.append({
                    "name": z["name"],
                    "points": _scale_points(z["points"], src_wh, dst_wh)
                })
            zones = scaled
        print(f"[INFO] Rescaled ROI/Zones from {src_wh[0]}x{src_wh[1]} to {dst_wh[0]}x{dst_wh[1]}")

    # Output path
    out_path = args.out
    if not out_path:
        out_dir = os.path.join(project_root(), "tracking", "openfield")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"rec_{timestamp_str()}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
    writer = None
    recording = False

    if args.record:
        writer = cv2.VideoWriter(out_path, fourcc, float(args.fps), (w, h))
        if not writer.isOpened():
            print(f"[ERROR] Failed to open writer: {out_path}")
            writer = None
        else:
            recording = True
            print(f"[REC] Recording to {out_path}")

    window = "OpenField Recorder (q:quit r:rec/stop o:overlay s:snapshot f:fullscreen)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    show_overlay = True
    snapshot_idx = 0

    print("[INFO] Loaded:")
    print(f"  ROI:   {'OK' if roi else 'None'} -> {args.roi}")
    print(f"  Zones: {'OK' if zones else 'None'} -> {args.zones}")
    print("[INFO] Controls: q quit | r rec/stop | o overlay toggle | s snapshot")

    # Warm-up read to avoid blank window
    fail_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            fail_count += 1
            if fail_count > 100:
                print("[ERROR] Failed to read from camera (timeout).")
                break
            cv2.waitKey(10)
            continue

        raw = frame
        display = draw_overlay(raw, roi, zones) if show_overlay else raw

        cv2.putText(display, f"overlay:{'ON' if show_overlay else 'OFF'}  rec:{'ON' if recording else 'OFF'}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)
        cv2.imshow(window, display)

        # write raw or with overlay according to flag
        if recording and writer is not None:
            # record raw or overlay according to flag
            writer.write(display if args.with_overlay else raw)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            show_overlay = not show_overlay
        elif key == ord('f'):
            # toggle fullscreen
            try:
                now = cv2.getWindowProperty(window, cv2.WND_PROP_FULLSCREEN)
                target = cv2.WINDOW_NORMAL if int(now) == 1 else cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, target)
            except Exception:
                pass
        elif key == ord('r'):
            # toggle recording
            if not recording:
                # start
                if writer is None:
                    writer = cv2.VideoWriter(out_path, fourcc, float(args.fps), (w, h))
                    if not writer.isOpened():
                        print(f"[ERROR] Failed to open writer: {out_path}")
                        writer = None
                        recording = False
                    else:
                        recording = True
                        print(f"[REC] Recording to {out_path}")
                else:
                    recording = True
                    print(f"[REC] Recording to {out_path}")
            else:
                # stop
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("[REC] Stopped")
        elif key == ord('s'):
            snap_dir = os.path.join(project_root(), "tracking", "openfield")
            os.makedirs(snap_dir, exist_ok=True)
            fn = os.path.join(snap_dir, f"snap_{timestamp_str()}_{snapshot_idx:02d}.png")
            cv2.imwrite(fn, display)
            print(f"[SAVE] snapshot -> {fn}")
            snapshot_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
