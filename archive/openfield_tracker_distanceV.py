import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

# Optional: Unicode text rendering via Pillow (for Korean)
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


Point = Tuple[int, int]


def project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def default_video_path() -> str:
    return os.path.join(project_root(), "tracking", "openfield", "m1_30min.mp4")


def default_roi_path() -> str:
    return os.path.join(project_root(), "tracking", "openfield", "roi.json")


def find_default_font() -> Optional[str]:
    candidates = []
    root = project_root()
    try:
        # common locations: project root and a fonts subfolder
        for base in [root, os.path.join(root, "fonts")]:
            if os.path.isdir(base):
                for fn in os.listdir(base):
                    low = fn.lower()
                    if (low.endswith(".ttf") or low.endswith(".otf")) and (
                        "nanum" in low or "gothic" in low or "noto" in low or "malgun" in low or "gulim" in low
                    ):
                        candidates.append(os.path.join(base, fn))
    except Exception:
        pass
    return candidates[0] if candidates else None


def setup_matplotlib_font(font_path: Optional[str]) -> bool:
    """Configure Matplotlib to use a Korean-capable font if available.
    Returns True if a font was set successfully.
    """
    try:
        if not font_path or not os.path.isfile(font_path):
            return False
        import matplotlib
        from matplotlib import font_manager
        # Register and use the provided font
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
        matplotlib.rcParams['font.family'] = family
        matplotlib.rcParams['axes.unicode_minus'] = False
        return True
    except Exception:
        return False


# ---- Safe save helpers (images/plots) ----
def _timestamp_suffix() -> str:
    try:
        from datetime import datetime
        return datetime.now().strftime("_%Y%m%d-%H%M%S")
    except Exception:
        return ""


def safe_imwrite(path: str, img: np.ndarray) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, img)
    if ok:
        print(f"[SAVE] image -> {path}")
        return path
    # fallback with timestamped filename (file busy or locked)
    base, ext = os.path.splitext(path)
    alt = f"{base}{_timestamp_suffix()}{ext}"
    ok2 = cv2.imwrite(alt, img)
    if ok2:
        print(f"[SAVE] image (alt) -> {alt}")
        return alt
    print(f"[WARN] Failed to save image: {path}")
    return path


def safe_pltsave(fig, path: str, font_path: Optional[str] = None) -> str:
    # Force non-interactive backend to avoid Qt conflicts
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        if font_path:
            setup_matplotlib_font(font_path)
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        print(f"[WARN] Matplotlib not available for saving plot: {e}")
        return path
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print(f"[SAVE] plot -> {path}")
        return path
    except Exception:
        base, ext = os.path.splitext(path)
        alt = f"{base}{_timestamp_suffix()}{ext}"
        try:
            fig.savefig(alt, bbox_inches="tight")
            print(f"[SAVE] plot (alt) -> {alt}")
            return alt
        except Exception as e2:
            print(f"[WARN] Failed to save plot: {path} ({e2})")
            return path


# ---- Zone plots (stats + distance/speed) ----
def save_zone_plots(
    export_csv: Optional[str],
    stats: Dict[str, Dict[str, object]],
    cm_per_px: Optional[float],
    font_path: Optional[str],
    meta: Optional[Dict[str, str]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        setup_matplotlib_font(font_path)
        import matplotlib.pyplot as plt  # type: ignore

        # Determine labels: prefer center/margin when present
        labels = list(stats.keys())
        if "center" in stats or "margin" in stats:
            lab = []
            if "center" in stats:
                lab.append("center")
            if "margin" in stats:
                lab.append("margin")
            # include others if any
            for k in labels:
                if k not in lab:
                    lab.append(k)
            labels = lab

        totals = [float(stats[k]["time_ms"]) if isinstance(stats[k]["time_ms"], float) else float(stats[k]["time_ms"]) / 1000.0 for k in labels]
        visits = [int(stats[k]["visits"]) for k in labels]
        dw = [stats[k]["dwells"] for k in labels]  # type: ignore
        means = [float(sum(x) / len(x)) if x else 0.0 for x in dw]
        maxes = [float(max(x)) if x else 0.0 for x in dw]

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        (ax1, ax2), (ax3, ax4) = axes
        ax1.bar(labels, totals, color="#4e79a7"); ax1.set_title("Total time (s) / 총 체류시간"); ax1.set_ylabel("s")
        ax2.bar(labels, visits, color="#59a14f"); ax2.set_title("Visits / 방문 수")
        ax3.bar(labels, means, color="#f28e2b"); ax3.set_title("Mean dwell (s) / 평균 머문 시간"); ax3.set_ylabel("s")
        ax4.bar(labels, maxes, color="#e15759"); ax4.set_title("Max dwell (s) / 최대 머문 시간"); ax4.set_ylabel("s")
        title_meta = ""
        if meta:
            title_meta = f"{meta.get('name','')} / {meta.get('student_id','')} / {meta.get('mouse_id','')}"
        fig.suptitle(f"Zone Analysis / 구역 분석 — {title_meta}")
        for ax in (ax1, ax2, ax3, ax4):
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        img_path = os.path.splitext(export_csv)[0] + "_zone_stats.png" if export_csv else os.path.join(project_root(), "tracking", "openfield", "zone_stats.png")
        safe_pltsave(fig, img_path, font_path)
        plt.close(fig)

        if cm_per_px and cm_per_px > 0:
            dists = [float(stats.get(k, {}).get("dist_cm", 0.0)) for k in labels]
            speeds = [(dists[i] / totals[i]) if totals[i] > 0 else 0.0 for i in range(len(labels))]
            fig2, (axd, axs) = plt.subplots(1, 2, figsize=(10, 4))
            axd.bar(labels, dists, color="#76b7b2"); axd.set_title("Distance (cm) / 거리"); axd.set_ylabel("cm")
            axs.bar(labels, speeds, color="#edc948"); axs.set_title("Mean speed (cm/s) / 평균 속도"); axs.set_ylabel("cm/s")
            for ax in (axd, axs):
                for tick in ax.get_xticklabels():
                    tick.set_rotation(15)
            fig2.suptitle(f"Zone Distance/Speed / 구역별 거리·속도 — {title_meta}")
            img2 = os.path.splitext(export_csv)[0] + "_zone_distance_speed.png" if export_csv else os.path.join(project_root(), "tracking", "openfield", "zone_distance_speed.png")
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            safe_pltsave(fig2, img2, font_path)
            plt.close(fig2)
    except Exception as e:
        print(f"[WARN] Zone plots not saved: {e}")


def default_calibration_path() -> str:
    return os.path.join(project_root(), "tracking", "openfield", "calibration.json")


def load_calibration(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        val = data.get("cm_per_px")
        return float(val) if val else None
    except Exception:
        return None


def save_calibration(path: str, cm_per_px: float, frame: Optional[np.ndarray] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, object] = {"cm_per_px": float(cm_per_px)}
    try:
        if frame is not None:
            h, w = frame.shape[:2]
            payload.update({"width": int(w), "height": int(h)})
        from datetime import datetime
        payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def default_zones_path() -> str:
    return os.path.join(project_root(), "tracking", "openfield", "zones.json")


def load_zones(path: str) -> Optional[List[Dict]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        zones = data.get("zones")
        if not zones:
            return None
        out = []
        for z in zones:
            name = z.get("name", "zone")
            pts = z.get("points", [])
            if pts and len(pts) >= 3:
                out.append({"name": str(name), "points": [(int(p[0]), int(p[1])) for p in pts]})
        return out or None
    except Exception:
        return None


def save_zones(path: str, zones: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"zones": [{"name": z["name"], "points": z["points"]} for z in zones]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_roi(path: str) -> Optional[List[Point]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = data.get("points")
        if not pts:
            return None
        return [(int(p[0]), int(p[1])) for p in pts]
    except Exception:
        return None


def save_roi(path: str, points: List[Point]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"points": points}, f, ensure_ascii=False, indent=2)


@dataclass
class ROIEditorState:
    window: str
    image: np.ndarray
    points: List[Point]
    closed: bool = False


def _draw_text(vis: np.ndarray, text: str, org: Tuple[int, int], font_path: Optional[str] = None,
               font_size: int = 20, color: Tuple[int, int, int] = (50, 220, 50)) -> None:
    if font_path and _HAS_PIL and os.path.isfile(font_path):
        try:
            pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            font = ImageFont.truetype(font_path, font_size)
            draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
            vis[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass

    # Force robust zone plots saving regardless of previous errors
    try:
        save_zone_plots(export_csv, stats, cm_per_px, font_path, meta)
    except Exception as e:
        print(f"[WARN] save_zone_plots failed: {e}")

    # Ensure zone plots always saved via robust saver (Agg + logging)
    try:
        save_zone_plots(export_csv, stats, cm_per_px, font_path, meta)
    except Exception as e:
        print(f"[WARN] save_zone_plots failed: {e}")
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_roi(image: np.ndarray, points: List[Point], closed: bool, font_path: Optional[str] = None) -> np.ndarray:
    vis = image.copy()
    color_line = (0, 255, 255)
    color_pt = (0, 165, 255)
    thickness = 2
    for i, p in enumerate(points):
        cv2.circle(vis, p, 4, color_pt, -1, lineType=cv2.LINE_AA)
        if i > 0:
            cv2.line(vis, points[i - 1], p, color_line, thickness, lineType=cv2.LINE_AA)
    if closed and len(points) >= 3:
        cv2.line(vis, points[-1], points[0], color_line, thickness, lineType=cv2.LINE_AA)
    hint = "좌클릭: 점추가 | 우클릭: 되돌리기 | C: 닫기 | R: 초기화 | S: 저장 | Enter: 확정"
    _draw_text(vis, hint, (10, 25), font_path)
    return vis


def roi_mouse_cb(event, x, y, flags, state: ROIEditorState):  # type: ignore[override]
    if event == cv2.EVENT_LBUTTONDOWN:
        state.points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if state.points:
            state.points.pop()


def interactive_roi(frame: np.ndarray, preset: Optional[List[Point]] = None, save_path: Optional[str] = None,
                    font_path: Optional[str] = None) -> Optional[List[Point]]:
    state = ROIEditorState(window="ROI Editor", image=frame.copy(), points=list(preset or []), closed=False)
    cv2.namedWindow(state.window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(state.window, roi_mouse_cb, state)

    while True:
        vis = draw_roi(state.image, state.points, state.closed, font_path)
        cv2.imshow(state.window, vis)
        key = cv2.waitKeyEx(20)
        if key != -1:
            key = key & 0xFFFFFFFF
        key_ff = key & 0xFF
        if key_ff == 255 or key == -1:
            continue

        if key_ff in (ord('c'), ord('C')):
            state.closed = len(state.points) >= 3
        elif key_ff in (ord('r'), ord('R')):
            state.points.clear()
            state.closed = False
        elif key_ff in (ord('s'), ord('S')):
            if len(state.points) >= 3 and save_path:
                save_roi(save_path, state.points)
                print(f"[INFO] ROI saved to {save_path}")
        elif key_ff in (13, 10):  # Enter
            if len(state.points) >= 3:
                cv2.destroyWindow(state.window)
                return state.points
        elif key_ff in (27, ord('q'), ord('Q')):  # Esc/q
            cv2.destroyWindow(state.window)
            return None


def mask_from_roi(shape: Tuple[int, int], points: List[Point]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def find_mouse_centroid(bin_img: np.ndarray, min_area: int, last_pt: Optional[Point] = None, max_jump: Optional[float] = None) -> Optional[Tuple[Point, np.ndarray]]:
    """Rollback: choose the largest contour by area above min_area.
    Ignores last_pt/max_jump to match the earlier, simpler behavior.
    """
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # filter by area and pick largest
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        return None
    c = max(candidates, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), c


def _draw_timeline(vis: np.ndarray, frame_idx: int, total_frames: int, font_path: Optional[str] = None) -> None:
    if total_frames <= 0:
        return
    h, w = vis.shape[:2]
    bar_h = max(6, h // 80)
    y1 = h - bar_h - 10
    y2 = h - 10
    cv2.rectangle(vis, (10, y1), (w - 10, y2), (60, 60, 60), -1)
    prog = min(max(frame_idx / max(1, total_frames - 1), 0.0), 1.0)
    x2p = 10 + int((w - 20) * prog)
    cv2.rectangle(vis, (10, y1), (x2p, y2), (0, 200, 0), -1)
    _draw_text(vis, f"frame {frame_idx}/{total_frames-1}", (10, y1 - 10), font_path)


def process_stream(
    cap: cv2.VideoCapture,
    roi_pts: List[Point],
    export_csv: Optional[str],
    min_area: int,
    show: bool = True,
    start_frame: int = 0,
    fps_override: Optional[float] = None,
    skip: int = 1,
    trail_sec: int = 5,
    font_path: Optional[str] = None,
    method: str = "bg",
    bright_thresh: int = 0,
    preferred_window_size: Optional[Tuple[int, int]] = None,
    max_jump: int = 80,
    meta: Optional[Dict[str, str]] = None,
    zones: Optional[List[Dict]] = None,
    cm_per_px: Optional[float] = None,
) -> None:
    mask = None
    path_pts: List[Point] = []
    path_ts: List[float] = []
    valid_segments: List[bool] = []
    # zone definitions and stats
    zone_names: List[str] = []
    zone_polys: List[np.ndarray] = []
    if zones:
        for z in zones:
            zone_names.append(str(z.get("name", "zone")))
            zone_polys.append(np.array(z.get("points"), dtype=np.int32))
    stats: Dict[str, Dict[str, object]] = {}
    def ensure_zone(name: str):
        if name not in stats:
            stats[name] = {"time_ms": 0, "visits": 0, "dwells": []}
    current_zone: Optional[str] = None
    current_start_ms: Optional[int] = None
    frame_idx = int(start_frame)
    writer = None
    if export_csv:
        os.makedirs(os.path.dirname(export_csv) or ".", exist_ok=True)
        # utf-8-sig adds BOM so Excel shows Korean correctly
        writer = open(export_csv, "w", encoding="utf-8-sig")
        if meta:
            for k, v in meta.items():
                writer.write(f"# {k}: {v}\n")
        if cm_per_px and cm_per_px > 0:
            writer.write(f"# cm_per_px: {cm_per_px}\n")
        writer.write("frame,timestamp_s,x,y,area\n")

    # Use background subtractor to adapt changing illumination
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    window = "OpenField Tracker"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(fps_override or (cap.get(cv2.CAP_PROP_FPS) or 30.0))
    recent_ms = max(1, int(trail_sec * 1000))
    if show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        if preferred_window_size:
            try:
                is_full = cv2.getWindowProperty(window, cv2.WND_PROP_FULLSCREEN)
            except Exception:
                is_full = -1
            if is_full not in (cv2.WINDOW_FULLSCREEN, 1):
                w, h = preferred_window_size
                cv2.resizeWindow(window, w * 2 + 10, h)

    abort = False
    first_bg: Optional[np.ndarray] = None
    heatmap: Optional[np.ndarray] = None
    lost_count = 0
    total_dist_px: float = 0.0
    total_dist_cm: float = 0.0
    # filters to reduce startup spikes on first segment(s)
    warmup_sec = 0.5
    max_speed_cm_s = 300.0
    total_dist_px: float = 0.0
    total_dist_cm: float = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        ts_s = frame_idx / fps

        if mask is None:
            mask = mask_from_roi(frame.shape[:2], roi_pts)

        # preproc + ROI mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if method == "bright":
            # Bright subject on darker background
            if bright_thresh > 0:
                _, fg = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
            else:
                _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            fg = bg.apply(gray)
            fg = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        fg = cv2.bitwise_and(fg, mask)
        # Clean up
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        # Rollback detection: largest contour above min_area (no nearest/jump logic)
        found = find_mouse_centroid(
            fg,
            min_area=min_area,
        )
        if first_bg is None:
            first_bg = frame.copy()
            heatmap = np.zeros(frame.shape[:2], dtype=np.uint32)
        vis_recent = frame.copy()
        vis_full = frame.copy()
        overlay_recent = np.zeros_like(vis_recent)
        overlay_full = np.zeros_like(vis_full)
        zones_layer_recent = np.zeros_like(vis_recent)
        zones_layer_full = np.zeros_like(vis_full)
        # draw ROI outline
        cv2.polylines(overlay_recent, [np.array(roi_pts, np.int32)], True, (0, 255, 255), 2)
        cv2.polylines(overlay_full, [np.array(roi_pts, np.int32)], True, (0, 255, 255), 2)
        # draw zones below paths
        if zone_polys:
            for name, poly in zip(zone_names, zone_polys):
                color = (0, 165, 255) if name.lower() == "center" else (255, 200, 0)
                cv2.fillPoly(zones_layer_recent, [poly], color)
                cv2.fillPoly(zones_layer_full, [poly], color)
                cv2.polylines(zones_layer_recent, [poly], True, (0, 0, 0), 3)
                cv2.polylines(zones_layer_recent, [poly], True, (255, 255, 255), 1)
                cv2.polylines(zones_layer_full, [poly], True, (0, 0, 0), 3)
                cv2.polylines(zones_layer_full, [poly], True, (255, 255, 255), 1)
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    zx = int(M["m10"] / M["m00"])
                    zy = int(M["m01"] / M["m00"])
                    _draw_text(zones_layer_recent, name, (zx - 20, zy), font_path)
                    _draw_text(zones_layer_full, name, (zx - 20, zy), font_path)

        if found:
            (cx, cy), contour = found
            path_pts.append((cx, cy))
            path_ts.append(ts_s)
            # segment distance (px, cm) with validity
            seg_px = 0.0
            seg_cm = 0.0
            seg_valid = False
            if len(path_pts) >= 2:
                dx = path_pts[-1][0] - path_pts[-2][0]
                dy = path_pts[-1][1] - path_pts[-2][1]
                seg_px = float((dx*dx + dy*dy) ** 0.5)
                dt = path_ts[-1] - path_ts[-2]
                if dt > 0 and (path_ts[-2] - path_ts[0] >= warmup_sec):
                    seg_valid = True
                if cm_per_px and cm_per_px > 0 and dt > 0:
                    seg_cm = seg_px * cm_per_px
                    speed = seg_cm / dt
                    if speed > max_speed_cm_s:
                        seg_valid = False
                if seg_valid:
                    total_dist_px += seg_px
                    if cm_per_px and cm_per_px > 0:
                        total_dist_cm += seg_cm
                valid_segments.append(seg_valid)
    # rollback: no lost-counter reset logic
            if heatmap is not None and 0 <= cy < heatmap.shape[0] and 0 <= cx < heatmap.shape[1]:
                heatmap[cy, cx] += 1
            cv2.circle(overlay_recent, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(overlay_full, (cx, cy), 5, (0, 0, 255), -1)
            cv2.drawContours(overlay_recent, [contour], -1, (255, 0, 0), 2)
            cv2.drawContours(overlay_full, [contour], -1, (255, 0, 0), 2)
            if writer is not None:
                writer.write(f"{frame_idx},{ts_s:.3f},{cx},{cy},{int(cv2.contourArea(contour))}\n")
            # zone classification
            zone_label: Optional[str] = None
            if zone_polys:
                for name, poly in zip(zone_names, zone_polys):
                    if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                        zone_label = name
                        break
                if zone_label is None and cv2.pointPolygonTest(np.array(roi_pts, np.int32), (float(cx), float(cy)), False) >= 0:
                    zone_label = "margin"
            elif cv2.pointPolygonTest(np.array(roi_pts, np.int32), (float(cx), float(cy)), False) >= 0:
                zone_label = "roi"
            if zone_label != current_zone:
                if current_zone is not None and current_start_ms is not None:
                    ensure_zone(current_zone)
                    # convert ms-based start to seconds if needed
                    start_s = current_start_ms / 1000.0 if isinstance(current_start_ms, int) else float(current_start_ms)
                    dwell = ts_s - start_s
                    # store in seconds
                    prev_total = float(stats[current_zone]["time_ms"]) if isinstance(stats[current_zone]["time_ms"], float) else float(stats[current_zone]["time_ms"]) / 1000.0
                    stats[current_zone]["time_ms"] = prev_total + max(0.0, dwell)
                    cast_list = stats[current_zone]["dwells"]  # type: ignore
                    cast_list.append(max(0.0, dwell))
                current_zone = zone_label
                current_start_ms = ts_s  # from now on keep seconds
                if current_zone is not None:
                    ensure_zone(current_zone)
                    stats[current_zone]["visits"] = int(stats[current_zone]["visits"]) + 1
            # attribute distance to the new/current zone
            if zone_label is not None and seg_cm > 0 and ('seg_valid' in locals() and seg_valid):
                ensure_zone(zone_label)
                stats[zone_label]["dist_cm"] = float(stats[zone_label].get("dist_cm", 0.0)) + seg_cm

        # rollback: do not auto-reset background subtractor based on lost frames

        # path trail (recent/full)
        if len(path_pts) > 1:
            # Full path
            for i in range(1, len(path_pts)):
                if i-1 < len(valid_segments) and valid_segments[i-1]:
                    cv2.line(overlay_full, path_pts[i - 1], path_pts[i], (0, 200, 0), 2)
            # Recent path (within recent_ms)
            cutoff = ts_s - float(trail_sec)
            # find first index >= cutoff; if none, draw nothing (avoid full trail)
            start_i = len(path_ts)
            for i, t in enumerate(path_ts):
                if t >= cutoff:
                    start_i = i
                    break
            for i in range(max(1, start_i), len(path_pts)):
                if i-1 < len(valid_segments) and valid_segments[i-1]:
                    cv2.line(overlay_recent, path_pts[i - 1], path_pts[i], (0, 200, 0), 2)

        # blend zones first (under paths), then paths overlays
        vis_recent = cv2.addWeighted(vis_recent, 1.0, zones_layer_recent, 0.35, 0)
        vis_full = cv2.addWeighted(vis_full, 1.0, zones_layer_full, 0.35, 0)
        vis_recent = cv2.addWeighted(vis_recent, 1.0, overlay_recent, 0.8, 0)
        vis_full = cv2.addWeighted(vis_full, 1.0, overlay_full, 0.6, 0)
        if show:
            _draw_text(vis_recent, "q:종료  p:일시정지  </>:속도", (10, 25), font_path)
            _draw_text(vis_recent, f"frame:{frame_idx}  skip:{skip}  t:{ts_s:.2f}s", (10, 50), font_path)
            _draw_timeline(vis_recent, frame_idx, total_frames, font_path)
            _draw_text(vis_full, "전체 궤적", (10, 25), font_path)
            _draw_timeline(vis_full, frame_idx, total_frames, font_path)
            # compose side-by-side
            h1, w1 = vis_recent.shape[:2]
            h2, w2 = vis_full.shape[:2]
            h_all = max(h1, h2)
            canvas = np.zeros((h_all, w1 + w2 + 10, 3), dtype=np.uint8)
            canvas[:h1, :w1] = vis_recent
            canvas[:h2, w1 + 10:w1 + 10 + w2] = vis_full
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(1)
            if key != -1:
                key = key & 0xFFFFFFFF
            key_ff = key & 0xFF
            if key_ff == ord('q'):
                abort = True
                break
            if key_ff == ord('p'):
                while True:
                    key2 = cv2.waitKeyEx(50)
                    if key2 != -1:
                        key2 = key2 & 0xFFFFFFFF
                    key2_ff = key2 & 0xFF
                    if key2_ff in (ord('p'), ord('q')):
                        if key2_ff == ord('q'):
                            abort = True
                        break
                if abort:
                    break
            if key_ff in (ord('>'), ord('.')):
                skip = min(skip * 2, 64)
            if key_ff in (ord('<'), ord(',')):
                skip = max(1, skip // 2)

        # Fast-forward processing by skipping frames (if requested)
        if skip > 1:
            for _ in range(skip - 1):
                grabbed = cap.grab()
                if not grabbed:
                    break
                frame_idx += 1

    # finalize last dwell
    if current_zone is not None and current_start_ms is not None:
        ensure_zone(current_zone)
        dwell = (path_ts[-1] if path_ts else 0) - (current_start_ms / 1000.0 if isinstance(current_start_ms, int) else current_start_ms)
        # convert all times internally to seconds
        stats[current_zone]["time_ms"] = float(stats[current_zone]["time_ms"]) + max(0.0, dwell)
        cast_list = stats[current_zone]["dwells"]  # type: ignore
        cast_list.append(max(0.0, dwell))

    # export zone summary
    if zones is not None or stats:
        try:
            import csv
            out_path = None
            if export_csv:
                base, _ = os.path.splitext(export_csv)
                out_path = base + "_zones.csv"
            else:
                out_path = os.path.join(project_root(), "tracking", "openfield", "zones_summary.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["zone", "total_time_s", "visits", "mean_dwell_s", "max_dwell_s", "distance_cm", "mean_speed_cm_s"])
                for name, data in stats.items():
                    dwells = data["dwells"]  # type: ignore
                    mean_dwell = float(sum(dwells) / len(dwells)) if dwells else 0.0
                    max_dwell = float(max(dwells)) if dwells else 0.0
                    total_s = float(data["time_ms"]) if isinstance(data["time_ms"], float) else float(data["time_ms"]) / 1000.0
                    dist_cm = float(data.get("dist_cm", 0.0))
                    mean_speed = (dist_cm / total_s) if total_s > 0 else 0.0
                    w.writerow([name, f"{total_s:.3f}", int(data["visits"]), f"{mean_dwell:.3f}", f"{max_dwell:.3f}", f"{dist_cm:.3f}", f"{mean_speed:.3f}"])
        except Exception:
            pass

    # export overall summary
    try:
        if export_csv and (cm_per_px and cm_per_px > 0):
            base, _ = os.path.splitext(export_csv)
            summary = base + "_summary.csv"
            import csv
            total_time_s = path_ts[-1] - path_ts[0] if len(path_ts) >= 2 else 0.0
            mean_vel = (total_dist_cm / total_time_s) if total_time_s > 0 else 0.0
            with open(summary, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["cm_per_px", "distance_cm", "mean_velocity_cm_s", "total_time_s"])
                w.writerow([f"{cm_per_px:.6f}", f"{total_dist_cm:.3f}", f"{mean_vel:.3f}", f"{total_time_s:.3f}"])
    except Exception:
        pass

        try:
            # Ensure Korean-capable font for titles/labels (e.g., NanumGothic)
            import matplotlib
            matplotlib.use("Agg", force=True)
            setup_matplotlib_font(font_path)
            import matplotlib.pyplot as plt  # type: ignore
            # Ensure center vs margin comparison when zones are defined
            if zone_polys:
                ensure_zone("center")
                ensure_zone("margin")
                labels = ["center", "margin"]
            else:
                # fallback when only ROI exists
                labels = list(stats.keys())
            totals = [float(stats.get(k, {"time_ms": 0.0})["time_ms"]) if isinstance(stats.get(k, {"time_ms": 0.0})["time_ms"], float) else float(stats.get(k, {"time_ms": 0})["time_ms"]) / 1000.0 for k in labels]
            visits = [int(stats.get(k, {"visits": 0})["visits"]) for k in labels]
            means = [float(sum(stats.get(k, {"dwells": []})["dwells"]) / len(stats.get(k, {"dwells": []})["dwells"])) if stats.get(k, {"dwells": []})["dwells"] else 0.0 for k in labels]
            maxes = [float(max(stats.get(k, {"dwells": []})["dwells"])) if stats.get(k, {"dwells": []})["dwells"] else 0.0 for k in labels]
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))
            (ax1, ax2), (ax3, ax4) = axes
            ax1.bar(labels, totals, color="#4e79a7"); ax1.set_title("Total time (s)"); ax1.set_ylabel("s")
            ax2.bar(labels, visits, color="#59a14f"); ax2.set_title("Visits")
            ax3.bar(labels, means, color="#f28e2b"); ax3.set_title("Mean dwell (s)"); ax3.set_ylabel("s")
            ax4.bar(labels, maxes, color="#e15759"); ax4.set_title("Max dwell (s)"); ax4.set_ylabel("s")
            title_meta = ""
            if meta:
                title_meta = f"{meta.get('name','')} / {meta.get('student_id','')} / {meta.get('mouse_id','')}"
            fig.suptitle(f"Zone Analysis / 구역 분석 — {title_meta}")
            for ax in (ax1, ax2, ax3, ax4):
                for tick in ax.get_xticklabels():
                    tick.set_rotation(15)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            img_path = os.path.splitext(export_csv)[0] + "_zone_stats.png" if export_csv else os.path.join(project_root(), "tracking", "openfield", "zone_stats.png")
            safe_pltsave(fig, img_path, font_path)
            plt.close(fig)

            # Additional zone distance/speed bars if calibration exists
            if cm_per_px and cm_per_px > 0:
                dists = [float(stats.get(k, {}).get("dist_cm", 0.0)) for k in labels]
                speeds = []
                for i, k in enumerate(labels):
                    t = totals[i]
                    d = dists[i]
                    speeds.append((d / t) if t > 0 else 0.0)
                fig2, (axd, axs) = plt.subplots(1, 2, figsize=(10, 4))
                axd.bar(labels, dists, color="#76b7b2"); axd.set_title("Distance (cm) / 거리"); axd.set_ylabel("cm")
                axs.bar(labels, speeds, color="#edc948"); axs.set_title("Mean speed (cm/s) / 평균 속도"); axs.set_ylabel("cm/s")
                for ax in (axd, axs):
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(15)
                fig2.suptitle(f"Zone Distance/Speed / 구역별 거리·속도 — {title_meta}")
                img2 = os.path.splitext(export_csv)[0] + "_zone_distance_speed.png" if export_csv else os.path.join(project_root(), "tracking", "openfield", "zone_distance_speed.png")
                plt.tight_layout(rect=[0, 0, 1, 0.90])
                safe_pltsave(fig2, img2, font_path)
                plt.close(fig2)
        except Exception:
            pass

    # Save static plots: full track overlay and heatmap
    try:
        if first_bg is not None and len(path_pts) > 1:
            bg1 = first_bg.copy()
            over_z = np.zeros_like(bg1)
            # zones fill
            if zone_polys:
                for name, poly in zip(zone_names, zone_polys):
                    color = (0, 165, 255) if name.lower() == "center" else (255, 200, 0)
                    cv2.fillPoly(over_z, [poly], color)
                    cv2.polylines(over_z, [poly], True, (0, 0, 0), 3)
                    cv2.polylines(over_z, [poly], True, (255, 255, 255), 1)
            cv2.polylines(over_z, [np.array(roi_pts, np.int32)], True, (0, 255, 255), 2)
            bg1 = cv2.addWeighted(bg1, 1.0, over_z, 0.35, 0)
            # full path
            path_layer = np.zeros_like(bg1)
            for i in range(1, len(path_pts)):
                cv2.line(path_layer, path_pts[i-1], path_pts[i], (0, 200, 0), 2)
            out_img1 = cv2.addWeighted(bg1, 1.0, path_layer, 0.9, 0)
            base = os.path.splitext(export_csv)[0] if export_csv else os.path.join(project_root(), "tracking", "openfield", "m1_30min")
            title_meta = ""
            if meta:
                title_meta = f"{meta.get('name','')} / {meta.get('student_id','')} / {meta.get('mouse_id','')}"
            _draw_text(out_img1, f"Track Overlay / 경로 오버레이 — {title_meta}", (10, 25), font_path)
            safe_imwrite(base + "_track_plot.png", out_img1)
            # heatmap with percentile scaling and warm colormap
            if heatmap is not None:
                hm = heatmap.astype(np.float32)
                # Smooth with Gaussian to create meaningful occupancy regions
                h, w = hm.shape[:2]
                k = max(5, (min(h, w) // 30) | 1)  # odd kernel size ~3.3% of min dimension
                hm = cv2.GaussianBlur(hm, (k, k), 0)
                if hm.max() > 0:
                    # Robust normalize to 95th percentile
                    import numpy as _np
                    p95 = float(_np.percentile(hm, 95)) if hm.size > 0 else float(hm.max())
                    scale = p95 if p95 > 0 else float(hm.max())
                    hm = hm / max(1.0, scale)
                    hm = np.clip(hm, 0.0, 1.0)
                hm_img = (hm * 255).astype(np.uint8)
                # warm colormap (TURBO or HOT)
                cmap_id = cv2.COLORMAP_TURBO if hasattr(cv2, 'COLORMAP_TURBO') else cv2.COLORMAP_HOT
                hm_color = cv2.applyColorMap(hm_img, cmap_id)
                # mask outside ROI
                mask_roi = np.zeros(hm_img.shape, dtype=np.uint8)
                cv2.fillPoly(mask_roi, [np.array(roi_pts, np.int32)], 255)
                hm_color = cv2.bitwise_and(hm_color, cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR))
                title_meta = ""
                if meta:
                    title_meta = f"{meta.get('name','')} / {meta.get('student_id','')} / {meta.get('mouse_id','')}"
                out_img2 = cv2.addWeighted(bg1, 0.6, hm_color, 0.8, 0)
                # add a title strip with metadata
                _draw_text(out_img2, f"Heatmap / 점유 열지도 — {title_meta}", (10, 25), font_path)
                safe_imwrite(base + "_heatmap.png", out_img2)

            # Distance & Speed time-series (if calibration exists)
            if cm_per_px and cm_per_px > 0 and len(path_ts) > 1:
                setup_matplotlib_font(font_path)
                import matplotlib.pyplot as plt  # type: ignore
                times = path_ts[1:]
                seg_cm = []
                inst_speed = []
                for i in range(1, len(path_pts)):
                    dt = path_ts[i] - path_ts[i-1]
                    dx = path_pts[i][0] - path_pts[i-1][0]
                    dy = path_pts[i][1] - path_pts[i-1][1]
                    dcm = ((dx*dx + dy*dy) ** 0.5) * cm_per_px
                    seg_cm.append(dcm)
                    inst_speed.append((dcm / dt) if dt > 0 else 0.0)
                cum_cm = []
                s = 0.0
                for v in seg_cm:
                    s += v
                    cum_cm.append(s)
                fig_ts, (axA, axB) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                axA.plot(times, cum_cm, color="#4e79a7"); axA.set_ylabel("cm"); axA.set_title("Cumulative Distance / 누적 거리")
                axB.plot(times, inst_speed, color="#e15759"); axB.set_ylabel("cm/s"); axB.set_xlabel("time (s)"); axB.set_title("Instantaneous Speed / 순간 속도")
                title_meta = ""
                if meta:
                    title_meta = f"{meta.get('name','')} / {meta.get('student_id','')} / {meta.get('mouse_id','')}"
                fig_ts.suptitle(f"Distance & Speed / 거리·속도 — {title_meta}")
                plt.tight_layout(rect=[0, 0, 1, 0.94])
                plt.savefig(base + "_distance_speed.png")
                plt.close(fig_ts)
    except Exception:
        pass

    if writer:
        writer.close()
    if show:
        cv2.destroyAllWindows()


def read_frame_at(cap: cv2.VideoCapture, idx: int) -> Tuple[bool, Optional[np.ndarray]]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, frame = cap.read()
    return ok, frame if ok else None


def preplay_select_start(cap: cv2.VideoCapture, roi_pts: List[Point], font_path: Optional[str] = None, method: str = "bg", bright_thresh: int = 0) -> Optional[Tuple[int, Tuple[int, int]]]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    play = False
    step = 1  # frames per tick when playing
    window = "Pre-Play: Side-by-Side (S/Enter)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # timeline via trackbar for robust scrubbing
    def on_trackbar(val):
        nonlocal idx
        idx = int(val)
    if total > 0:
        cv2.createTrackbar("frame", window, idx, max(0, total - 1), on_trackbar)

    def overlay_info(frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        # draw ROI polygon
        if roi_pts:
            cv2.polylines(vis, [np.array(roi_pts, np.int32)], True, (0, 255, 255), 2)
        t_ms = int((idx / fps) * 1000)
        msg1 = "Space 재생/일시정지  ,/.: 프레임 이동  ←/→: ±5초  ↑/↓: ±1초"
        msg2 = "</>: 속도(step x2/x0.5)  Home/End: 점프  S/Enter: 시작  Q:취소"
        msg3 = f"frame {idx}/{total}  t={t_ms}ms  step={step}"
        _draw_text(vis, msg1, (10, 25), font_path)
        _draw_text(vis, msg2, (10, 50), font_path)
        _draw_text(vis, msg3, (10, 75), font_path)
        return vis

    while True:
        ok, frame = read_frame_at(cap, idx)
        if not ok or frame is None:
            break
        # Build side-by-side preview (left: original, right: method preview)
        left = frame.copy()
        if roi_pts:
            cv2.polylines(left, [np.array(roi_pts, np.int32)], True, (0, 255, 255), 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if method == "bright":
            if bright_thresh > 0:
                _, fg = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
            else:
                _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            fg = cv2.Canny(gray, 50, 150)
        right = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
        if roi_pts:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(roi_pts, np.int32)], 255)
            right = cv2.bitwise_and(right, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        h1, w1 = left.shape[:2]
        h2, w2 = right.shape[:2]
        h = max(h1, h2)
        canvas = np.zeros((h, w1 + w2 + 10, 3), dtype=np.uint8)
        canvas[:h1, :w1] = left
        canvas[:h2, w1 + 10:w1 + 10 + w2] = right
        t_ms = int((idx / fps) * 1000)
        _draw_text(canvas, "원본", (10, 25), font_path)
        _draw_text(canvas, "프리뷰", (w1 + 20, 25), font_path)
        msg1 = "Space 재생/정지  ,/.: 프레임  A/D: ±1초  J/L: ±5초  </>: 속도"
        msg2 = "S/Enter: 시작  Home/End: 점프  Q:취소  트랙바 드래그: 타임라인"
        msg3 = f"frame {idx}/{total}  t={t_ms}ms  step={step}"
        _draw_text(canvas, msg1, (10, 50), font_path)
        _draw_text(canvas, msg2, (10, 75), font_path)
        _draw_text(canvas, msg3, (10, 100), font_path)
        cv2.imshow(window, canvas)
        if total > 0:
            cv2.setTrackbarPos("frame", window, max(0, min(idx, total - 1)))
        key = cv2.waitKeyEx(20)
        if key != -1:
            key = key & 0xFFFFFFFF
        # Map extended arrows on Windows
        LEFT = {81, 2424832}
        RIGHT = {83, 2555904}
        UP = {82, 2490368}
        DOWN = {84, 2621440}

        if key == -1:  # no key
            if play:
                idx = min(idx + step, max(0, total - 1)) if total > 0 else idx + step
            continue

        if key & 0xFF == ord(' '):  # space
            play = not play
        elif key & 0xFF in (ord('>'), ord('.')):
            step = min(step * 2, 128)
        elif key & 0xFF in (ord('<'), ord(',')):
            step = max(1, step // 2)
        elif key in LEFT or key & 0xFF in (ord('j'), ord('J')):  # left arrow or J
            idx = max(0, idx - int(5 * fps))
        elif key in RIGHT or key & 0xFF in (ord('l'), ord('L')):  # right arrow or L
            idx = min(max(0, total - 1), idx + int(5 * fps)) if total > 0 else idx + int(5 * fps)
        elif key in UP or key & 0xFF in (ord('d'), ord('D')):  # up arrow or D (+1s)
            idx = min(max(0, total - 1), idx + int(1 * fps)) if total > 0 else idx + int(1 * fps)
        elif key in DOWN or key & 0xFF in (ord('a'), ord('A')):  # down arrow or A (-1s)
            idx = max(0, idx - int(1 * fps))
        elif key & 0xFF in (ord('S'), ord('s')) or key & 0xFF in (13, 10):  # enter
            preferred_size = None
            try:
                _, _, w, h = cv2.getWindowImageRect(window)
                preferred_size = (w, h)
            except Exception:
                preferred_size = None
            cv2.destroyWindow(window)
            return idx, preferred_size or (frame.shape[1], frame.shape[0])
        elif key == 36 or key == 2359296:  # Home
            idx = 0
        elif key == 35:  # End (may not be portable)
            if total > 0:
                idx = max(0, total - 1)
        elif key & 0xFF in (27, ord('q'), ord('Q')):
            cv2.destroyWindow(window)
            return None

    cv2.destroyWindow(window)
    return None


def interactive_zone(frame: np.ndarray, name: str, preset: Optional[List[Point]] = None,
                     font_path: Optional[str] = None) -> Optional[List[Point]]:
    title = f"Zone Editor: {name}"
    state = ROIEditorState(window=title, image=frame.copy(), points=list(preset or []), closed=False)
    cv2.namedWindow(state.window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(state.window, roi_mouse_cb, state)
    while True:
        vis = draw_roi(state.image, state.points, state.closed, font_path)
        _draw_text(vis, f"Editing zone: {name}", (10, 50), font_path)
        cv2.imshow(state.window, vis)
        key = cv2.waitKeyEx(20)
        if key != -1:
            key = key & 0xFFFFFFFF
        key_ff = key & 0xFF
        if key_ff == 255 or key == -1:
            continue
        if key_ff in (ord('c'), ord('C')):
            state.closed = len(state.points) >= 3
        elif key_ff in (ord('r'), ord('R')):
            state.points.clear()
            state.closed = False
        elif key_ff in (13, 10):
            if len(state.points) >= 3:
                cv2.destroyWindow(state.window)
                return state.points
        elif key_ff in (27, ord('q'), ord('Q')):
            cv2.destroyWindow(state.window)
            return None


def preplay_select_start_sxs(
    cap: cv2.VideoCapture,
    roi_pts: List[Point],
    font_path: Optional[str] = None,
    method: str = "bg",
    bright_thresh: int = 0,
    zones: Optional[List[Dict]] = None,
) -> Optional[Tuple[int, Tuple[int, int]]]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    play = False
    step = 1
    window = "Pre-Play: Side-by-Side (S/Enter)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # default to fullscreen for convenience
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass

    def on_trackbar(val):
        nonlocal idx
        idx = int(val)
    if total > 0:
        cv2.createTrackbar("frame", window, idx, max(0, total - 1), on_trackbar)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        left = frame.copy()
        if roi_pts:
            cv2.polylines(left, [np.array(roi_pts, np.int32)], True, (0, 255, 255), 2)
        # draw zones (center etc.) if provided
        if zones:
            zone_polys = [np.array(z.get("points"), dtype=np.int32) for z in zones if z.get("points")]
            zone_names = [str(z.get("name", "zone")) for z in zones if z.get("points")]
            overlay_left = np.zeros_like(left)
            for name, poly in zip(zone_names, zone_polys):
                color = (0, 165, 255) if name.lower() == "center" else (255, 200, 0)
                cv2.fillPoly(overlay_left, [poly], color)
                cv2.polylines(overlay_left, [poly], True, (0, 0, 0), 3)
                cv2.polylines(overlay_left, [poly], True, (255, 255, 255), 1)
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    _draw_text(overlay_left, name, (cx - 20, cy), font_path)
            left = cv2.addWeighted(left, 1.0, overlay_left, 0.35, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if method == "bright":
            if bright_thresh > 0:
                _, fg = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
            else:
                _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            fg = cv2.Canny(gray, 50, 150)
        right = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
        if roi_pts:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(roi_pts, np.int32)], 255)
            right = cv2.bitwise_and(right, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        # optionally also hint zones on the right preview for consistency
        if zones:
            overlay_right = np.zeros_like(right)
            for name, poly in zip(zone_names, zone_polys):
                color = (0, 80, 200) if name.lower() == "center" else (100, 160, 255)
                cv2.polylines(overlay_right, [poly], True, color, 2)
            # draw ROI edge too
            cv2.polylines(overlay_right, [np.array(roi_pts, np.int32)], True, (0, 255, 255), 2)
            right = cv2.addWeighted(right, 1.0, overlay_right, 0.8, 0)

        h1, w1 = left.shape[:2]
        h2, w2 = right.shape[:2]
        H = max(h1, h2)
        canvas = np.zeros((H, w1 + w2 + 10, 3), dtype=np.uint8)
        canvas[:h1, :w1] = left
        canvas[:h2, w1 + 10:w1 + 10 + w2] = right
        t_ms = int((idx / fps) * 1000)
        _draw_text(canvas, "Original", (10, 25), font_path)
        _draw_text(canvas, "Preview", (w1 + 20, 25), font_path)
        _draw_text(canvas, f"frame {idx}/{max(0, total - 1)}  t={t_ms}ms  step={step}", (10, 50), font_path)
        _draw_text(canvas, "Space play/pause  ,/. frame  A/D ±1s  J/L ±5s  </> speed  Home/End jump  S/Enter start  Q quit", (10, 75), font_path)

        _draw_text(canvas, "K: calibrate scale (draw 2 points, then input cm)", (10, 125), font_path)
        cv2.imshow(window, canvas)
        if total > 0:
            cv2.setTrackbarPos("frame", window, max(0, min(idx, total - 1)))
        key = cv2.waitKeyEx(20)
        if key != -1:
            key = key & 0xFFFFFFFF

        LEFT = {81, 2424832}
        RIGHT = {83, 2555904}
        UP = {82, 2490368}
        DOWN = {84, 2621440}

        if key == -1:
            if play:
                idx = min(idx + step, max(0, total - 1)) if total > 0 else idx + step
            continue
        kff = key & 0xFF
        if kff == ord(' '):
            play = not play
        elif kff in (ord('>'), ord('.')):
            step = min(step * 2, 128)
        elif kff in (ord('<'), ord(',')):
            step = max(1, step // 2)
        elif key in LEFT or kff in (ord('j'), ord('J')):
            idx = max(0, idx - int(5 * fps))
        elif key in RIGHT or kff in (ord('l'), ord('L')):
            idx = min(max(0, total - 1), idx + int(5 * fps)) if total > 0 else idx + int(5 * fps)
        elif key in UP or kff in (ord('d'), ord('D')):
            idx = min(max(0, total - 1), idx + int(1 * fps)) if total > 0 else idx + int(1 * fps)
        elif key in DOWN or kff in (ord('a'), ord('A')):
            idx = max(0, idx - int(1 * fps))
        elif kff in (ord('k'), ord('K')):
            # interactive calibration on current frame
            cm = interactive_calibration(frame, roi_pts, zones, font_path)
            if cm is not None:
                try:
                    save_calibration(default_calibration_path(), cm, frame)
                    print(f"[INFO] Saved calibration: {cm:.4f} cm/px")
                except Exception:
                    pass
        elif kff in (ord('s'), ord('S')) or kff in (13, 10):
            preferred_size = None
            try:
                _, _, w, h = cv2.getWindowImageRect(window)
                preferred_size = (w, h)
            except Exception:
                preferred_size = None
            cv2.destroyWindow(window)
            return idx, preferred_size or (frame.shape[1], frame.shape[0])
        elif key == 36 or key == 2359296:
            idx = 0
        elif key == 35:
            if total > 0:
                idx = max(0, total - 1)
        elif kff in (27, ord('q'), ord('Q')):
            cv2.destroyWindow(window)
            return None

    cv2.destroyWindow(window)
    return None


def interactive_calibration(frame: np.ndarray, roi_pts: List[Point], zones: Optional[List[Dict]], font_path: Optional[str]) -> Optional[float]:
    pts: List[Point] = []
    tmp = frame.copy()
    title = "Calibration: click 2 points, type cm, Enter to save"
    win = title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    text_buf: List[str] = []
    def cb(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
    cv2.setMouseCallback(win, cb)
    while True:
        vis = tmp.copy()
        if roi_pts:
            cv2.polylines(vis, [np.array(roi_pts, np.int32)], True, (0, 255, 255), 2)
        if zones:
            for z in zones:
                poly = np.array(z.get("points", []), np.int32)
                if poly.size > 0:
                    cv2.polylines(vis, [poly], True, (0, 165, 255), 2)
        mid: Optional[Tuple[int, int]] = None
        if len(pts) >= 1:
            cv2.circle(vis, pts[0], 5, (0, 0, 255), -1)
        if len(pts) == 2:
            cv2.circle(vis, pts[1], 5, (0, 0, 255), -1)
            cv2.line(vis, pts[0], pts[1], (0, 0, 255), 2)
            mid = ((pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2)
            dpx = float(((pts[0][0]-pts[1][0])**2 + (pts[0][1]-pts[1][1])**2)**0.5)
            _draw_text(vis, f"pixels: {dpx:.2f}", (10, 25), font_path)
        guide = "Type length(cm), Enter save | Backspace delete | R reset | Esc cancel"
        _draw_text(vis, guide, (10, 50), font_path)
        if mid is not None:
            _draw_text(vis, "".join(text_buf) or "cm?", (mid[0]-20, mid[1]-10), font_path)
        cv2.imshow(win, vis)
        key = cv2.waitKeyEx(30)
        if key != -1:
            key = key & 0xFFFFFFFF
        k = key & 0xFF
        if k == 27:
            cv2.destroyWindow(win)
            return None
        if k in (ord('r'), ord('R')):
            pts.clear(); text_buf.clear()
            continue
        if k in (13, 10) and len(pts) == 2 and text_buf:
            try:
                cm = float("".join(text_buf))
                dpx = float(((pts[0][0]-pts[1][0])**2 + (pts[0][1]-pts[1][1])**2)**0.5)
                if cm > 0 and dpx > 0:
                    cv2.destroyWindow(win)
                    return cm / dpx
            except Exception:
                pass
        # typing buffer
        if (ord('0') <= k <= ord('9')) or k == ord('.'):
            text_buf.append(chr(k))
        elif k in (8, 127):  # backspace/delete
            if text_buf:
                text_buf.pop()
    return None


def main():
    parser = argparse.ArgumentParser(description="OpenField mouse path tracker with ROI and pre-play start selection")
    parser.add_argument("--video", type=str, help="Path to a video file (or resolved from --mouse-id)")
    parser.add_argument("--mouse-id", type=str, default=None, help="Mouse ID to resolve video (searches tracking/openfield/*{ID}*.mp4)")
    parser.add_argument("--roi", type=str, default=default_roi_path(), help="ROI JSON path (saved/loaded)")
    parser.add_argument("--zones", type=str, default=default_zones_path(), help="Zones JSON path (saved/loaded)")
    parser.add_argument("--edit-roi", action="store_true", help="Force redraw of ROI")
    parser.add_argument("--define-zones", action="store_true", help="Open zone editor to add/replace zones (e.g., center)")
    parser.add_argument("--min-area", type=int, default=300, help="Minimum contour area to accept as mouse")
    parser.add_argument("--export-csv", type=str, default=None, help="CSV output path for tracked points")
    parser.add_argument("--start-frame", type=int, default=None, help="Start tracking from this frame index")
    parser.add_argument("--start-ms", type=int, default=None, help="Start tracking from this timestamp (ms)")
    parser.add_argument("--no-preplay", action="store_true", help="Skip interactive pre-play start selection")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (fast-forward tracking)")
    parser.add_argument("--trail-sec", type=int, default=5, help="Recent trail duration in seconds (for Recent window)")
    parser.add_argument("--font", type=str, default=None, help="Path to a TTF/OTF font file for Korean text")
    parser.add_argument("--method", type=str, choices=["bg", "bright"], default="bg", help="Detection method (default: bg)")
    parser.add_argument("--bright-thresh", type=int, default=0, help="Threshold (0=Otsu) when method=bright")
    parser.add_argument("--max-jump", type=int, default=80, help="Max acceptable jump (pixels) between frames; larger is ignored")
    parser.add_argument("--name", type=str, default=None, help="Experimenter name")
    parser.add_argument("--student-id", type=str, default=None, help="Student ID")
    args = parser.parse_args()

    # Prompt metadata upfront for simple, arg-free runs (Korean prompts)
    exp_name = args.name or input("실험자 이름: ").strip() or "unknown"
    student_id = args.student_id or input("학번: ").strip() or "unknown"
    mouse_id = args.mouse_id or input("마우스 ID: ").strip()
    metadata = {"name": exp_name, "student_id": student_id, "mouse_id": mouse_id or "unknown"}

    # Determine font automatically if not provided
    font_path = args.font or find_default_font()

    # Resolve recorded video (no webcam)
    vid_path = args.video
    if not vid_path:
        root = os.path.join(project_root(), "tracking", "openfield")
        if mouse_id:
            candidates = []
            try:
                for fn in os.listdir(root):
                    if fn.lower().endswith(".mp4") and mouse_id.lower() in fn.lower():
                        candidates.append(os.path.join(root, fn))
            except FileNotFoundError:
                pass
            if len(candidates) == 1:
                vid_path = candidates[0]
            elif len(candidates) > 1:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                vid_path = candidates[0]
        if not vid_path:
            vid_path = default_video_path()
    if not os.path.isfile(vid_path):
        print(f"[ERROR] Video not found: {vid_path}")
        return
    cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {vid_path}")
        return

    # Read a frame to define ROI
    ok, first = cap.read()
    if not ok:
        print("[ERROR] Could not read from source.")
        cap.release()
        return

    roi_pts = load_roi(args.roi)
    if not roi_pts or args.edit_roi:
        print("[INFO] No ROI found. Draw polygon ROI on the first frame.")
        sel = interactive_roi(first, preset=None, save_path=args.roi, font_path=font_path)
        if not sel:
            print("[INFO] ROI selection cancelled. Exiting.")
            cap.release()
            return
        roi_pts = sel
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Load or define zones (e.g., center vs. margin)
    zones = load_zones(args.zones)
    if args.define_zones or zones is None:
        ans = input("Define a center zone inside ROI? [y/N]: ").strip().lower()
        if ans == 'y':
            zpts = interactive_zone(first, name="center", preset=None, font_path=font_path)
            if zpts and len(zpts) >= 3:
                zones = [{"name": "center", "points": zpts}]
                try:
                    save_zones(args.zones, zones)
                    print(f"[INFO] Zones saved to {args.zones}")
                except Exception:
                    pass
        elif zones is None:
            zones = []

    start_frame = 0
    preferred_size: Optional[Tuple[int, int]] = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # Load calibration if present (pre‑play K may update; we will reload again after pre‑play)
    cm_per_px = load_calibration(default_calibration_path())
    if cm_per_px:
        try:
            print(f"[INFO] Loaded calibration: {cm_per_px:.6f} cm/px")
        except Exception:
            pass
    if args.start_frame is not None:
        start_frame = max(0, min(args.start_frame, max(0, total - 1))) if total > 0 else max(0, args.start_frame)
    elif args.start_ms is not None:
        est = int((args.start_ms / 1000.0) * fps)
        start_frame = max(0, min(est, max(0, total - 1))) if total > 0 else max(0, est)
    elif not args.no_preplay:
        sel = preplay_select_start_sxs(cap, roi_pts, font_path=font_path, method=args.method, bright_thresh=args.bright_thresh, zones=zones)
        if sel is None:
            print("[INFO] Start selection cancelled. Exiting.")
            cap.release()
            return
        start_frame, preferred_size = sel
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Reload calibration in case user pressed K and saved during pre‑play
    latest_cm = load_calibration(default_calibration_path())
    if latest_cm and latest_cm != cm_per_px:
        cm_per_px = latest_cm
        try:
            print(f"[INFO] Using calibration: {cm_per_px:.6f} cm/px")
        except Exception:
            pass

    export_csv = args.export_csv
    if export_csv is None:
        # default CSV next to the actual video path
        base = os.path.splitext(os.path.basename(vid_path))[0]
        export_csv = os.path.join(os.path.dirname(vid_path), f"{base}_track.csv")

    process_stream(
        cap,
        roi_pts=roi_pts,
        export_csv=export_csv,
        min_area=args.min_area,
        show=True,
        start_frame=start_frame,
        fps_override=None,
        skip=max(1, int(args.skip)),
        trail_sec=max(1, int(args.trail_sec)),
        font_path=font_path,
        method=args.method,
        bright_thresh=max(0, int(args.bright_thresh)),
        preferred_window_size=preferred_size,
        max_jump=int(args.max_jump),
        meta={**metadata, "video": os.path.basename(vid_path)},
        zones=zones,
        cm_per_px=cm_per_px,
    )


if __name__ == "__main__":
    main()
