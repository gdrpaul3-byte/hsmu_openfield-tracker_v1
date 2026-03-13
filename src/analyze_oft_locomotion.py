#!/usr/bin/env python3
"""Analyze locomotion metrics (distance, speed) with sex comparisons and plots."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required. Install it with `pip install matplotlib`.") from exc


_FONT_REGISTERED = False
_FONT_NAME: str | None = None


def setup_font() -> None:
    """Ensure Nanum Gothic fonts are registered and applied."""
    global _FONT_REGISTERED, _FONT_NAME
    try:
        from matplotlib import font_manager, rcParams  # type: ignore
    except ImportError:
        return

    if not _FONT_REGISTERED:
        base_dir = Path(__file__).resolve().parents[2]
        for candidate in [
            "NanumGothic.ttf",
            "NanumGothicBold.ttf",
            "NanumGothicExtraBold.ttf",
            "NanumGothicLight.ttf",
        ]:
            font_path = base_dir / candidate
            if font_path.exists():
                font_manager.fontManager.addfont(str(font_path))
                _FONT_NAME = font_manager.FontProperties(fname=str(font_path)).get_name()
                _FONT_REGISTERED = True
                break

    if _FONT_NAME:
        rcParams["font.family"] = _FONT_NAME
        rcParams["font.sans-serif"] = [_FONT_NAME]
        rcParams["axes.unicode_minus"] = False


def significance_label(p_value: float, alpha: float = 0.05) -> str:
    """Return bilingual significance string."""
    if not math.isfinite(p_value):
        return "p ? (판별 불가 / unclear)"
    comparator = "<" if p_value < alpha else "≥"
    status = (
        "유의함 / significant" if p_value < alpha else "유의하지 않음 / not significant"
    )
    return f"p {comparator} {alpha:.2f} ({status})"


def compute_sem(values: Sequence[float]) -> float:
    """Standard error for a sequence."""
    n = len(values)
    if n <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(n)


def independent_t_test(group1: Sequence[float], group2: Sequence[float]) -> Tuple[float, float, float]:
    """Welch's t-test between two groups. Returns t, df, p."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least two samples per group.")
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    var1, var2 = statistics.variance(group1), statistics.variance(group2)
    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        t_stat = float("inf") if mean1 != mean2 else 0.0
        p_value = 0.0 if mean1 != mean2 else 1.0
        df = (n1 + n2) - 2
    else:
        t_stat = (mean1 - mean2) / se
        num = (var1 / n1 + var2 / n2) ** 2
        den = (var1 * var1) / ((n1 * n1) * (n1 - 1)) + (var2 * var2) / ((n2 * n2) * (n2 - 1))
        df = num / den if den else (n1 + n2 - 2)
        # two-tailed p-value via Beta function
        p_value = two_tailed_t_pvalue(t_stat, df)
    return t_stat, df, p_value


def two_tailed_t_pvalue(t_stat: float, df: float) -> float:
    """Compute two-tailed t-test p using incomplete beta."""
    if df <= 0 or not math.isfinite(t_stat):
        return float("nan")
    x = df / (df + t_stat * t_stat)
    ib = regularized_incomplete_beta(df / 2.0, 0.5, x)
    if t_stat > 0:
        cdf = 1.0 - 0.5 * ib
    else:
        cdf = 0.5 * ib
    return 2.0 * min(cdf, 1.0 - cdf)


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) + ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta."""
    MAXIT = 200
    EPS = 3e-7
    FPMIN = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delh = d * c
        h *= delh
        if abs(delh - 1.0) < EPS:
            break
    return h


@dataclass
class LocomotionRow:
    mouse_id: str
    sex: str
    total_distance_cm: float
    mean_speed_cm_per_s: float


def load_locomotion(csv_path: Path) -> List[LocomotionRow]:
    """Parse oft_locomotion_metrics.csv into LocomotionRow objects."""
    rows: List[LocomotionRow] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for entry in reader:
            mouse_id = entry["mouse_id"].strip()
            sex = "Male" if mouse_id.upper().startswith("M") else "Female"
            rows.append(
                LocomotionRow(
                    mouse_id=mouse_id,
                    sex=sex,
                    total_distance_cm=float(entry["total_distance_cm"]),
                    mean_speed_cm_per_s=float(entry["mean_speed_cm_per_s"]),
                )
            )
    return rows


METRICS = {
    "total_distance_cm": {
        "label": "Total distance (cm) / 총 이동거리(cm)",
        "filename": "locomotion_distance",
    },
    "mean_speed_cm_per_s": {
        "label": "Mean speed (cm/s) / 평균 속도(cm/s)",
        "filename": "locomotion_speed",
    },
}


def plot_metric(rows: List[LocomotionRow], metric: str, output_path: Path) -> Tuple[float, float, float]:
    """Bar plot with jittered points comparing male vs female for metric."""
    plt.style.use("seaborn-v0_8-whitegrid")
    setup_font()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    male_values = [getattr(row, metric) for row in rows if row.sex == "Male"]
    female_values = [getattr(row, metric) for row in rows if row.sex == "Female"]

    means = [statistics.mean(male_values), statistics.mean(female_values)]
    sems = [compute_sem(male_values), compute_sem(female_values)]
    colors = ["#4C72B0", "#C44E52"]
    labels = ["Male / 수컷", "Female / 암컷"]
    positions = [0, 1]

    ax.bar(
        positions,
        means,
        yerr=sems,
        capsize=8,
        color=colors,
        edgecolor="black",
        alpha=0.85,
    )

    rng = random.Random(99)
    for idx, values in enumerate([male_values, female_values]):
        for value in values:
            ax.scatter(
                positions[idx] + rng.uniform(-0.08, 0.08),
                value,
                color="black",
                s=28,
                alpha=0.75,
            )

    ax.set_xticks(positions, labels)
    ax.set_ylabel(METRICS[metric]["label"])
    ax.set_title(f"Sex comparison of {METRICS[metric]['label']}")

    t_stat, df, p_value = independent_t_test(male_values, female_values)
    ax.text(
        0.02,
        0.98,
        f"Welch t-test (Male - Female): t({df:.2f}) = {t_stat:.3f}, p = {p_value:.4f}\n"
        f"{significance_label(p_value)}",
        ha="left",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return t_stat, df, p_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot locomotion metrics (total distance, mean speed) with sex comparisons."
    )
    parser.add_argument(
        "--locomotion-csv",
        type=Path,
        default=Path(__file__).parent / "oft_locomotion_metrics.csv",
        help="Path to oft_locomotion_metrics.csv produced by aggregate_oft_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "plots",
        help="Directory to store generated figures.",
    )
    args = parser.parse_args()

    csv_path = args.locomotion_csv.expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"Locomotion CSV not found: {csv_path}")

    rows = load_locomotion(csv_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, meta in METRICS.items():
        output_path = output_dir / f"{meta['filename']}.png"
        t_stat, df, p_value = plot_metric(rows, metric_key, output_path)
        male_mean = statistics.mean([getattr(r, metric_key) for r in rows if r.sex == "Male"])
        female_mean = statistics.mean([getattr(r, metric_key) for r in rows if r.sex == "Female"])
        print(
            f"[{meta['label']}] Welch t-test (Male - Female): "
            f"t({df:.2f}) = {t_stat:.3f}, p = {p_value:.4f} ({significance_label(p_value)})"
        )
        print(f"    Male mean: {male_mean:.2f}, Female mean: {female_mean:.2f}")
        print(f"    Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
