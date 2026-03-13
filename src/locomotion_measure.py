import os
import csv
import json
import math
import argparse
from typing import List, Tuple, Optional, Dict


def project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def default_out_dir() -> str:
    return os.path.join(project_root(), "tracking", "openfield")


def default_calibration_path() -> str:
    return os.path.join(default_out_dir(), "calibration.json")


def find_default_font() -> Optional[str]:
    root = project_root()
    try:
        for base in [root, os.path.join(root, "fonts")] :
            if not os.path.isdir(base):
                continue
            for fn in os.listdir(base):
                low = fn.lower()
                if (low.endswith(".ttf") or low.endswith(".otf")) and (
                    "nanum" in low or "gothic" in low or "noto" in low or "malgun" in low or "gulim" in low
                ):
                    return os.path.join(base, fn)
    except Exception:
        pass
    return None


def setup_matplotlib_font(font_path: Optional[str]) -> None:
    try:
        import matplotlib
        from matplotlib import font_manager
        matplotlib.use("Agg", force=True)
        if font_path and os.path.isfile(font_path):
            font_manager.fontManager.addfont(font_path)
            family = font_manager.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams['font.family'] = family
            matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


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


def list_track_files(folder: str) -> List[str]:
    out: List[str] = []
    try:
        for fn in os.listdir(folder):
            if fn.lower().endswith("_track.csv"):
                out.append(os.path.join(folder, fn))
    except FileNotFoundError:
        pass
    out.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return out


def parse_track_csv(path: str) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, str]]:
    timestamps: List[float] = []
    points: List[Tuple[float, float]] = []
    meta: Dict[str, str] = {}
    # read meta lines first (starting with '# ')
    with open(path, "r", encoding="utf-8-sig") as f:
        # peek meta
        pos = f.tell()
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("#"):
                # rewind to start of header
                f.seek(pos)
                break
            # parse: '# key: value'
            try:
                s = line.lstrip("#").strip()
                if ":" in s:
                    k, v = s.split(":", 1)
                    meta[k.strip()] = v.strip()
            except Exception:
                pass
            pos = f.tell()

    # now read table
    with open(path, "r", encoding="utf-8-sig", newline="") as f2:
        # skip commented lines
        rows = [ln for ln in f2.read().splitlines() if not ln.startswith("#")]
    if not rows:
        return timestamps, points, meta
    reader = csv.DictReader(rows)
    for row in reader:  # type: ignore[arg-type]
        try:
            t = float(row.get("timestamp_s", "0") or 0)
            x = float(row.get("x", "nan"))
            y = float(row.get("y", "nan"))
            if math.isfinite(x) and math.isfinite(y):
                timestamps.append(t)
                points.append((x, y))
        except Exception:
            continue
    return timestamps, points, meta


def compute_distance_cm(points: List[Tuple[float, float]], cm_per_px: float) -> float:
    if not points or cm_per_px <= 0:
        return 0.0
    total_px = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        total_px += math.hypot(x1 - x0, y1 - y0)
    return total_px * cm_per_px


def compute_mean_speed_cm_s(dist_cm: float, timestamps: List[float]) -> float:
    if not timestamps:
        return 0.0
    t0, t1 = timestamps[0], timestamps[-1]
    dt = max(0.0, t1 - t0)
    if dt <= 0:
        return 0.0
    return dist_cm / dt


def save_summary_csv(out_path: str, dist_cm: float, mean_speed: float, meta: Dict[str, str]) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        # meta as commented header
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total_distance_cm", f"{dist_cm:.3f}"])
        w.writerow(["mean_speed_cm_per_s", f"{mean_speed:.3f}"])
    print(f"[SAVE] csv -> {out_path}")
    return out_path


def save_bar_plot(img_path: str, dist_cm: float, mean_speed: float, font_path: Optional[str], meta: Dict[str, str]) -> str:
    try:
        setup_matplotlib_font(font_path)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        (ax1, ax2) = axes
        ax1.bar(["Distance"], [dist_cm], color="#4e79a7")
        ax1.set_ylabel("cm")
        ax1.set_title("Total Distance (cm) / 총 이동거리 (cm)")
        ax2.bar(["Velocity"], [mean_speed], color="#edc948")
        ax2.set_ylabel("cm/s")
        ax2.set_title("Mean Velocity (cm/s) / 평균 속도 (cm/s)")
        title_meta = ""
        if meta:
            title_meta = f"{meta.get('name','')} / {meta.get('student_id','')} / {meta.get('mouse_id','')}"
        fig.suptitle(f"Locomotion Summary / 보행 요약 — {title_meta}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
        fig.savefig(img_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVE] plot -> {img_path}")
        return img_path
    except Exception as e:
        print(f"[WARN] Failed to save plot: {e}")
        return img_path


def main() -> None:
    p = argparse.ArgumentParser(description="Compute locomotion metrics (distance, velocity) from an existing *_track.csv")
    p.add_argument("--dir", type=str, default=default_out_dir(), help="Directory to search for *_track.csv files")
    p.add_argument("--calib", type=str, default=default_calibration_path(), help="Path to calibration.json (cm_per_px)")
    p.add_argument("--font", type=str, default=None, help="TTF/OTF font path for Korean labels")
    args = p.parse_args()

    folder = args.dir
    files = list_track_files(folder)
    if not files:
        print(f"[ERROR] No *_track.csv found under: {folder}")
        return

    print("[SELECT] Choose a track file:")
    for i, pth in enumerate(files):
        print(f"  [{i}] {os.path.basename(pth)}")
    sel = input("Index (or full path): ").strip()
    if sel.isdigit():
        idx = int(sel)
        if not (0 <= idx < len(files)):
            print("[ERROR] Invalid index")
            return
        track_path = files[idx]
    else:
        track_path = sel if sel else files[0]
    if not os.path.isfile(track_path):
        print(f"[ERROR] File not found: {track_path}")
        return

    cm_per_px = load_calibration(args.calib)
    if not cm_per_px or cm_per_px <= 0:
        try:
            raw = input("cm_per_px 값이 없습니다. 직접 입력 (예: 0.068): ").strip()
            cm_per_px = float(raw)
        except Exception:
            print("[ERROR] Calibration missing; aborting.")
            return

    ts, pts, meta = parse_track_csv(track_path)
    if not pts:
        print("[ERROR] No points in CSV.")
        return

    dist_cm = compute_distance_cm(pts, cm_per_px)
    mean_speed = compute_mean_speed_cm_s(dist_cm, ts)

    base, _ = os.path.splitext(track_path)
    csv_out = base + "_locomotion.csv"
    img_out = base + "_locomotion.png"
    font_path = args.font or find_default_font()

    save_summary_csv(csv_out, dist_cm, mean_speed, meta)
    save_bar_plot(img_out, dist_cm, mean_speed, font_path, meta)


if __name__ == "__main__":
    main()

