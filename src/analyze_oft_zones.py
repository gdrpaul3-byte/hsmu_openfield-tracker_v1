#!/usr/bin/env python3
"""Visualize center vs margin occupancy and run paired t-test."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

_FONT_REGISTERED = False
_FONT_NAME: str | None = None


def setup_font() -> None:
    """Register Nanum Gothic fonts once for Matplotlib and reapply rcParams."""
    global _FONT_REGISTERED, _FONT_NAME
    try:
        from matplotlib import font_manager, rcParams  # type: ignore
    except ImportError:
        return

    if not _FONT_REGISTERED:
        base_dir = Path(__file__).resolve().parents[2]
        font_candidates = [
            "NanumGothic.ttf",
            "NanumGothicBold.ttf",
            "NanumGothicExtraBold.ttf",
            "NanumGothicLight.ttf",
        ]
        for font_name in font_candidates:
            font_path = base_dir / font_name
            if font_path.exists():
                font_manager.fontManager.addfont(str(font_path))
                font_prop = font_manager.FontProperties(fname=str(font_path))
                _FONT_NAME = font_prop.get_name()
                _FONT_REGISTERED = True
                break

    if _FONT_NAME:
        rcParams["font.family"] = _FONT_NAME
        rcParams["font.sans-serif"] = [_FONT_NAME]
        rcParams["axes.unicode_minus"] = False

METRIC_CONFIG: Dict[str, Dict[str, str]] = {
    "total_time_s": {
        "label": "Time spent (s) / 체류 시간(초)",
        "title": "Open-field center vs margin occupancy / 오픈필드 중앙 vs 주변 체류",
        "filename": "total_time",
        "description": "Total time",
        "diff_suffix": " s",
    },
    "visits": {
        "label": "Number of visits / 진입 횟수(회)",
        "title": "Center vs margin entries / 중앙 vs 주변 진입 횟수",
        "filename": "visits",
        "description": "Visit count",
        "diff_suffix": " visits",
    },
    "mean_dwell_s": {
        "label": "Mean dwell (s) / 평균 체류시간(초)",
        "title": "Center vs margin mean dwell / 중앙 vs 주변 평균 체류",
        "filename": "mean_dwell",
        "description": "Mean dwell time",
        "diff_suffix": " s",
    },
    "max_dwell_s": {
        "label": "Max dwell (s) / 최대 체류시간(초)",
        "title": "Center vs margin max dwell / 중앙 vs 주변 최대 체류",
        "filename": "max_dwell",
        "description": "Max dwell time",
        "diff_suffix": " s",
    },
}

ZONE_KEYS = ("center", "margin")
ZONE_DISPLAY = {
    "center": "Center / 중앙",
    "margin": "Margin / 주변",
}

@dataclass
class ZonePair:
    mouse_id: str
    video: str
    center: Dict[str, float]
    margin: Dict[str, float]


def load_zone_pairs(zone_csv: Path) -> List[ZonePair]:
    """Read the per-zone CSV and return mice that have both center and margin."""
    per_mouse: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    video_by_mouse: Dict[str, str] = {}

    with zone_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mouse_id = row["mouse_id"].strip().upper()
            video_by_mouse[mouse_id] = row.get("video", "").strip()
            zone = row["zone"].strip().lower()
            metrics = {}
            for metric_key in METRIC_CONFIG:
                try:
                    metrics[metric_key] = float(row[metric_key])
                except (KeyError, TypeError, ValueError):
                    metrics[metric_key] = float("nan")
            per_mouse[mouse_id][zone] = metrics

    pairs: List[ZonePair] = []
    for mouse_id, zones in sorted(per_mouse.items()):
        if "center" in zones and "margin" in zones:
            pairs.append(
                ZonePair(
                    mouse_id=mouse_id,
                    video=video_by_mouse.get(mouse_id, ""),
                    center=zones["center"],
                    margin=zones["margin"],
                )
            )
    return pairs


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction approximation for incomplete beta."""
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


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Evaluate the regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) + ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def two_tailed_t_pvalue(t_stat: float, df: float) -> float:
    """Return two-tailed p-value for t statistic."""
    if df <= 0 or not math.isfinite(t_stat):
        return float("nan")
    x = df / (df + t_stat * t_stat)
    ib = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    if t_stat > 0:
        cdf = 1.0 - 0.5 * ib
    else:
        cdf = 0.5 * ib
    return 2.0 * min(cdf, 1.0 - cdf)


def paired_t_test(center: List[float], margin: List[float]) -> Dict[str, float]:
    """Manual paired t-test that does not rely on SciPy."""
    diffs = [m - c for c, m in zip(center, margin)]
    n = len(diffs)
    if n < 2:
        raise ValueError("Need at least two mice for a paired t-test.")

    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)
    df = n - 1
    if std_diff == 0:
        t_stat = float("inf") if mean_diff else 0.0
        p_value = 0.0 if mean_diff else 1.0
    else:
        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se
        p_value = two_tailed_t_pvalue(t_stat, df)
    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_value,
    }


def f_test_p_value(f_stat: float, df1: int, df2: int) -> float:
    """Two-tailed p-value for an F statistic using the incomplete beta function."""
    if df1 <= 0 or df2 <= 0 or not math.isfinite(f_stat) or f_stat < 0:
        return float("nan")
    x = (df1 * f_stat) / (df1 * f_stat + df2)
    cdf = _regularized_incomplete_beta(df1 / 2.0, df2 / 2.0, x)
    return max(0.0, min(1.0, 1.0 - cdf))


def independent_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """Welch's t-test for two independent groups."""
    n1 = len(group1)
    n2 = len(group2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least two samples in each group for t-test.")

    mean1 = statistics.mean(group1)
    mean2 = statistics.mean(group2)
    var1 = statistics.variance(group1)
    var2 = statistics.variance(group2)

    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        t_stat = float("inf") if mean1 != mean2 else 0.0
        p_value = 0.0 if mean1 != mean2 else 1.0
        df = n1 + n2 - 2
    else:
        t_stat = (mean1 - mean2) / se
        numerator = (var1 / n1 + var2 / n2) ** 2
        denominator = (var1 * var1) / (n1 * n1 * (n1 - 1)) + (var2 * var2) / (
            n2 * n2 * (n2 - 1)
        )
        df = numerator / denominator if denominator else n1 + n2 - 2
        p_value = two_tailed_t_pvalue(t_stat, df)

    return {
        "t_stat": t_stat,
        "df": df,
        "p_value": p_value,
        "mean_diff": mean1 - mean2,
        "group_means": (mean1, mean2),
    }


def compute_sem(values: List[float]) -> float:
    """Return standard error of the mean for a list of values."""
    n = len(values)
    if n <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(n)


def significance_label(p_value: float, alpha: float = 0.05) -> str:
    """Return a bilingual statement such as 'p < 0.05 (유의함 / significant)'."""
    if not math.isfinite(p_value):
        return "p ? (판별 불가 / unclear)"
    comparator = "<" if p_value < alpha else "≥"
    status = (
        "유의함 / significant" if p_value < alpha else "유의하지 않음 / not significant"
    )
    return f"p {comparator} {alpha:.2f} ({status})"


def plot_center_margin(
    pairs: List[ZonePair], metric_key: str, output_path: Path, metric_cfg: Dict[str, str]
) -> Dict[str, float]:
    """Create the bar plot with individual paired points."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    centers = [pair.center[metric_key] for pair in pairs]
    margins = [pair.margin[metric_key] for pair in pairs]
    stats = paired_t_test(centers, margins)

    zone_labels = [ZONE_DISPLAY["center"], ZONE_DISPLAY["margin"]]
    means = [statistics.mean(centers), statistics.mean(margins)]
    sems = [compute_sem(centers), compute_sem(margins)]
    colors = ["#4C72B0", "#DD8452"]
    x_positions = [0, 1]

    plt.style.use("seaborn-v0_8-whitegrid")
    setup_font()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(
        x_positions,
        means,
        yerr=sems,
        capsize=8,
        color=colors,
        width=0.6,
        edgecolor="black",
        alpha=0.85,
    )

    jitter_scale = 0.04
    rng = random.Random(42)
    for pair in pairs:
        jitter = [rng.uniform(-jitter_scale, jitter_scale) for _ in range(2)]
        xs = [x + j for x, j in zip(x_positions, jitter)]
        ys = [pair.center[metric_key], pair.margin[metric_key]]
        ax.plot(xs, ys, color="#8C8C8C", alpha=0.5, linewidth=0.8, zorder=2)
        ax.scatter(xs, ys, color="#222222", s=25, zorder=3)

    ax.set_xticks(x_positions, zone_labels)
    ax.set_ylabel(metric_cfg["label"])
    ax.set_title(metric_cfg["title"])
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    stats_text = (
        "Paired t-test / 대응표본 t-검정 (margin - center)\n"
        f"t({stats['df']}) = {stats['t_stat']:.3f}, "
        f"p = {stats['p_value']:.4e} ({significance_label(stats['p_value'])})\n"
        f"Mean diff = {stats['mean_diff']:.2f} ± {stats['std_diff']:.2f}{metric_cfg.get('diff_suffix', '')}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        ha="left",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return stats


def run_sex_zone_anova(pairs: List[ZonePair], metric_key: str) -> Dict[str, Dict[str, float]]:
    """Run Sex (between) × Zone (within) mixed ANOVA for the chosen metric."""
    subjects, sexes = build_sex_subjects(pairs, metric_key)
    if len(sexes) < 2:
        raise ValueError("Need both male (M*) and female (F*) mice for the ANOVA.")

    values = [sub[zone] for sub in subjects for zone in ZONE_KEYS]
    grand_mean = statistics.mean(values)
    zone_count = len(ZONE_KEYS)
    n_subjects = len(subjects)

    n_by_sex = {sex: sum(1 for sub in subjects if sub["sex"] == sex) for sex in sexes}
    mean_sex = {
        sex: statistics.mean(
            [sub[zone] for sub in subjects if sub["sex"] == sex for zone in ZONE_KEYS]
        )
        for sex in sexes
    }
    subject_means = {
        sub["mouse_id"]: statistics.mean([sub[zone] for zone in ZONE_KEYS]) for sub in subjects
    }
    mean_zone = {zone: statistics.mean([sub[zone] for sub in subjects]) for zone in ZONE_KEYS}
    mean_sex_zone = {
        (sex, zone): statistics.mean(
            [sub[zone] for sub in subjects if sub["sex"] == sex]
        )
        for sex in sexes
        for zone in ZONE_KEYS
    }

    SSA = sum(n_by_sex[sex] * zone_count * (mean_sex[sex] - grand_mean) ** 2 for sex in sexes)
    SS_subjects = sum(
        zone_count * (subject_means[sub["mouse_id"]] - mean_sex[sub["sex"]]) ** 2
        for sub in subjects
    )
    SS_within = sum(
        (sub[zone] - subject_means[sub["mouse_id"]]) ** 2
        for sub in subjects
        for zone in ZONE_KEYS
    )
    SSB = sum(n_subjects * (mean_zone[zone] - grand_mean) ** 2 for zone in ZONE_KEYS)
    SSAB = 0.0
    for sex in sexes:
        for zone in ZONE_KEYS:
            term = mean_sex_zone[(sex, zone)] - mean_sex[sex] - mean_zone[zone] + grand_mean
            SSAB += n_by_sex[sex] * term * term
    SS_error = max(0.0, SS_within - SSB - SSAB)

    dfA = len(sexes) - 1
    df_subjects = n_subjects - len(sexes)
    dfB = len(ZONE_KEYS) - 1
    dfAB = dfA * dfB
    df_error = df_subjects * dfB

    MSA = SSA / dfA if dfA > 0 else float("nan")
    MS_subjects = SS_subjects / df_subjects if df_subjects > 0 else float("nan")
    F_sex = MSA / MS_subjects if MS_subjects and MS_subjects != 0 else float("nan")

    MSB = SSB / dfB if dfB > 0 else float("nan")
    MSAB = SSAB / dfAB if dfAB > 0 else float("nan")
    MS_error = SS_error / df_error if df_error > 0 else float("nan")
    F_zone = MSB / MS_error if MS_error and MS_error != 0 else float("nan")
    F_interaction = MSAB / MS_error if MS_error and MS_error != 0 else float("nan")

    return {
        "sex": {
            "F": F_sex,
            "df1": dfA,
            "df2": df_subjects,
            "p": f_test_p_value(F_sex, dfA, df_subjects),
        },
        "zone": {
            "F": F_zone,
            "df1": dfB,
            "df2": df_error,
            "p": f_test_p_value(F_zone, dfB, df_error),
        },
        "interaction": {
            "F": F_interaction,
            "df1": dfAB,
            "df2": df_error,
            "p": f_test_p_value(F_interaction, dfAB, df_error),
        },
        "group_means": {
            sex: {zone: mean_sex_zone[(sex, zone)] for zone in ZONE_KEYS} for sex in sexes
        },
        "subjects": subjects,
    }


def build_sex_subjects(
    pairs: List[ZonePair], metric_key: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return list of subject records with sex info for the given metric."""
    subjects: List[Dict[str, Any]] = []
    for pair in pairs:
        mouse = pair.mouse_id.upper()
        if mouse.startswith("M"):
            sex = "Male"
        elif mouse.startswith("F"):
            sex = "Female"
        else:
            continue
        subjects.append(
            {
                "mouse_id": pair.mouse_id,
                "sex": sex,
                "center": pair.center[metric_key],
                "margin": pair.margin[metric_key],
            }
        )

    sexes = sorted({sub["sex"] for sub in subjects})
    return subjects, sexes


def posthoc_zone_ttests(subjects: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Run Male vs Female comparisons separately for each zone."""
    results: Dict[str, Dict[str, float]] = {}
    for zone in ZONE_KEYS:
        male_values = [sub[zone] for sub in subjects if sub["sex"] == "Male"]
        female_values = [sub[zone] for sub in subjects if sub["sex"] == "Female"]
        if len(male_values) < 2 or len(female_values) < 2:
            continue
        stats = independent_t_test(male_values, female_values)
        stats["male_mean"], stats["female_mean"] = stats["group_means"]
        del stats["group_means"]
        results[zone] = stats
    return results


def plot_sex_comparison(
    subjects: List[Dict[str, Any]],
    metric_cfg: Dict[str, str],
    output_path: Path,
    anova: Dict[str, Any] | None = None,
    posthoc: Dict[str, Dict[str, float]] | None = None,
) -> None:
    """Visualize male vs female differences for each zone."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    plt.style.use("seaborn-v0_8-whitegrid")
    setup_font()

    zone_order = list(ZONE_KEYS)
    zone_labels = [ZONE_DISPLAY[z] for z in zone_order]
    x_positions = [0, 1]
    width = 0.32

    sex_order = ["Male", "Female"]
    sex_colors = {"Male": "#4C72B0", "Female": "#C44E52"}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    rng = random.Random(123)
    for idx, sex in enumerate(sex_order):
        values_per_zone = [
            [sub[zone] for sub in subjects if sub["sex"] == sex] for zone in zone_order
        ]
        means = [statistics.mean(vals) if vals else 0.0 for vals in values_per_zone]
        sems = [compute_sem(vals) for vals in values_per_zone]
        offsets = [x + (idx - 0.5) * width for x in x_positions]
        ax.bar(
            offsets,
            means,
            width=width,
            yerr=sems,
            capsize=6,
            label=f"{sex}",
            color=sex_colors[sex],
            alpha=0.85,
        )

        for zone_idx, zone_vals in enumerate(values_per_zone):
            for val in zone_vals:
                jitter = rng.uniform(-width * 0.2, width * 0.2)
                ax.scatter(
                    offsets[zone_idx] + jitter,
                    val,
                    color=sex_colors[sex],
                    edgecolor="black",
                    linewidth=0.3,
                    s=30,
                    alpha=0.8,
                    zorder=3,
                )

    ax.set_xticks(x_positions, zone_labels)
    ax.set_ylabel(metric_cfg["label"])
    ax.set_title(f"{metric_cfg['title']} - Sex comparison / 성별 비교")
    ax.legend(frameon=False)

    text_lines = []
    if anova:
        sex_stats = anova["sex"]
        zone_stats = anova["zone"]
        int_stats = anova["interaction"]
        text_lines.extend(
            [
                "ANOVA 결과:",
                f"Sex effect: F({sex_stats['df1']}, {sex_stats['df2']}) = {sex_stats['F']:.3f}, "
                f"p = {sex_stats['p']:.4f} ({significance_label(sex_stats['p'])})",
                f"Zone effect: F({zone_stats['df1']}, {zone_stats['df2']}) = {zone_stats['F']:.3f}, "
                f"p = {zone_stats['p']:.4f} ({significance_label(zone_stats['p'])})",
                f"Sex×Zone: F({int_stats['df1']}, {int_stats['df2']}) = {int_stats['F']:.3f}, "
                f"p = {int_stats['p']:.4f} ({significance_label(int_stats['p'])})",
            ]
        )
    if posthoc:
        text_lines.append("Post hoc (Male vs Female):")
        for zone in zone_order:
            if zone in posthoc:
                stats = posthoc[zone]
                text_lines.append(
                    f"{ZONE_DISPLAY[zone]}: t({stats['df']:.2f}) = {stats['t_stat']:.3f}, "
                    f"p = {stats['p_value']:.4f} ({significance_label(stats['p_value'])})"
                )

    if text_lines:
        ax.text(
            1.02,
            0.98,
            "\n".join(text_lines),
            ha="left",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot center vs margin metrics and run paired t-test."
    )
    parser.add_argument(
        "--zone-csv",
        type=Path,
        default=Path(__file__).parent / "oft_zone_metrics.csv",
        help="Path to oft_zone_metrics.csv (from aggregate_oft_metrics.py).",
    )
    parser.add_argument(
        "--metric",
        choices=list(METRIC_CONFIG.keys()),
        default="total_time_s",
        help="Which metric to visualize when not using --all-metrics.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Generate figures for every available metric.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Custom output path when plotting a single metric.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to place auto-named figures.",
    )
    parser.add_argument(
        "--sex-anova",
        action="store_true",
        help="Also run Sex (Male vs Female) × Zone mixed ANOVA for the selected metrics.",
    )
    parser.add_argument(
        "--sex-figures",
        action="store_true",
        help="Generate additional male vs female comparison figures.",
    )
    args = parser.parse_args()

    zone_csv = args.zone_csv.expanduser().resolve()
    if not zone_csv.exists():
        raise SystemExit(f"Zone CSV not found: {zone_csv}")

    pairs = load_zone_pairs(zone_csv)
    if not pairs:
        raise SystemExit("No mice with both center and margin entries were found.")

    metrics = list(METRIC_CONFIG.keys()) if args.all_metrics else [args.metric]
    output_dir = args.output_dir.expanduser().resolve()
    single_output = args.output.expanduser().resolve() if args.output else None

    for metric_key in metrics:
        cfg = METRIC_CONFIG[metric_key]
        if not args.all_metrics and single_output:
            output_path = single_output
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"oft_center_margin_{cfg['filename']}.png"
            output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = plot_center_margin(pairs, metric_key, output_path, cfg)
        suffix = cfg.get("diff_suffix", "")
        print(
            f"[{cfg['description']}] Paired t-test / 대응표본 t-검정 (margin - center): "
            f"t({stats['df']}) = {stats['t_stat']:.3f}, "
            f"p = {stats['p_value']:.4f} ({significance_label(stats['p_value'])}), "
            f"mean diff = {stats['mean_diff']:.2f} ± {stats['std_diff']:.2f}{suffix}"
        )
        print(f"Figure saved to: {output_path}")

        need_sex_analysis = args.sex_anova or args.sex_figures
        if need_sex_analysis:
            try:
                anova = run_sex_zone_anova(pairs, metric_key)
            except ValueError as exc:
                print(f"Sex analysis skipped: {exc}")
                continue

            posthoc = posthoc_zone_ttests(anova["subjects"])

            if args.sex_anova:
                sex_stats = anova["sex"]
                zone_stats = anova["zone"]
                int_stats = anova["interaction"]
                print(
                    "  Sex main effect: "
                    f"F({sex_stats['df1']}, {sex_stats['df2']}) = {sex_stats['F']:.3f}, "
                    f"p = {sex_stats['p']:.4f} ({significance_label(sex_stats['p'])})"
                )
                print(
                    "  Zone main effect: "
                    f"F({zone_stats['df1']}, {zone_stats['df2']}) = {zone_stats['F']:.3f}, "
                    f"p = {zone_stats['p']:.4f} ({significance_label(zone_stats['p'])})"
                )
                print(
                    "  Sex × Zone interaction: "
                    f"F({int_stats['df1']}, {int_stats['df2']}) = {int_stats['F']:.3f}, "
                    f"p = {int_stats['p']:.4f} ({significance_label(int_stats['p'])})"
                )
                for sex, means in anova["group_means"].items():
                    print(
                        f"    {sex} means -> Center: {means['center']:.2f}, Margin: {means['margin']:.2f}"
                    )
                for zone, stats in posthoc.items():
                    print(
                        f"    Post hoc {ZONE_DISPLAY.get(zone, zone)} (Male vs Female): "
                        f"t({stats['df']:.2f}) = {stats['t_stat']:.3f}, "
                        f"p = {stats['p_value']:.4f} ({significance_label(stats['p_value'])}), "
                        f"mean diff (Male-Female) = {stats['mean_diff']:.2f} "
                        f"(Male {stats['male_mean']:.2f}, Female {stats['female_mean']:.2f})"
                    )

            if args.sex_figures or args.sex_anova:
                sex_fig_dir = output_dir
                sex_fig_dir.mkdir(parents=True, exist_ok=True)
                sex_fig = sex_fig_dir / f"oft_sex_comparison_{cfg['filename']}.png"
                plot_sex_comparison(anova["subjects"], cfg, sex_fig, anova, posthoc)
                print(f"Sex comparison figure saved to: {sex_fig}")

if __name__ == "__main__":
    main()
