#!/usr/bin/env python3
"""Aggregate open-field locomotion and zone metrics into separate CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_zone_file(path: Path) -> Iterable[Dict[str, str]]:
    """Yield zone statistics rows with numeric values."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {}
            for key, value in row.items():
                if not key:
                    continue
                clean_key = key.strip().lstrip("\ufeff").lower()
                normalized[clean_key] = value
            yield {
                "zone": normalized["zone"],
                "total_time_s": float(normalized["total_time_s"]),
                "visits": int(normalized["visits"]),
                "mean_dwell_s": float(normalized["mean_dwell_s"]),
                "max_dwell_s": float(normalized["max_dwell_s"]),
            }


def parse_locomotion_file(path: Path) -> Dict[str, str]:
    """Read locomotion metrics, ignoring experimenter metadata."""
    metadata: Dict[str, str] = {}
    metrics: Dict[str, float] = {}

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                clean_key = key.strip().lstrip("\ufeff").lower()
                metadata[clean_key] = value.strip()
                continue
            if line.lower().startswith("metric"):
                continue
            metric, _, value = line.partition(",")
            value = value.strip()
            if not value:
                continue
            try:
                clean_metric = metric.strip().lstrip("\ufeff").lower()
                metrics[clean_metric] = float(value)
            except ValueError:
                continue

    return {
        "mouse_id": metadata.get("mouse_id", ""),
        "video": metadata.get("video", ""),
        "total_distance_cm": metrics.get("total_distance_cm"),
        "mean_speed_cm_per_s": metrics.get("mean_speed_cm_per_s"),
    }


def collect_tables(data_dir: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split aggregated data into zone table and locomotion table."""
    zone_rows: List[Dict[str, str]] = []
    locomotion_rows: List[Dict[str, str]] = []
    zone_files = sorted(data_dir.glob("*_oft_track_zones.csv"))
    exported_mice: set[str] = set()

    for zone_file in zone_files:
        base_stem = zone_file.stem
        if not base_stem.endswith("_zones"):
            continue
        base_name = base_stem[: -len("_zones")]
        locomotion_file = zone_file.with_name(f"{base_name}_locomotion.csv")
        locomotion = parse_locomotion_file(locomotion_file) if locomotion_file.exists() else {}

        fallback_mouse_id = base_name.split("_", 1)[0]
        mouse_id = (locomotion.get("mouse_id") or fallback_mouse_id).strip().upper()
        video = (locomotion.get("video") or base_name.replace("_track", "") + ".mp4").strip()

        for zone_row in parse_zone_file(zone_file):
            zone_rows.append(
                {
                    "mouse_id": mouse_id,
                    "video": video,
                    **zone_row,
                }
            )

        if mouse_id not in exported_mice:
            locomotion_rows.append(
                {
                    "mouse_id": mouse_id,
                    "video": video,
                    "total_distance_cm": locomotion.get("total_distance_cm"),
                    "mean_speed_cm_per_s": locomotion.get("mean_speed_cm_per_s"),
                }
            )
            exported_mice.add(mouse_id)

    return zone_rows, locomotion_rows


def write_zone_output(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Persist per-zone metrics to CSV."""
    fieldnames = [
        "mouse_id",
        "video",
        "zone",
        "total_time_s",
        "visits",
        "mean_dwell_s",
        "max_dwell_s",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_locomotion_output(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Persist per-mouse locomotion metrics to CSV."""
    fieldnames = [
        "mouse_id",
        "video",
        "total_distance_cm",
        "mean_speed_cm_per_s",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-mouse open-field zone stats and locomotion metrics."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory that contains *_oft_track_{zones,locomotion}.csv files.",
    )
    parser.add_argument(
        "--zone-output",
        type=Path,
        default=Path(__file__).parent / "oft_zone_metrics.csv",
        help="Path to the per-zone aggregated CSV file to create.",
    )
    parser.add_argument(
        "--locomotion-output",
        type=Path,
        default=Path(__file__).parent / "oft_locomotion_metrics.csv",
        help="Path to the per-mouse locomotion CSV file to create.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    zone_rows, locomotion_rows = collect_tables(data_dir)
    if not zone_rows:
        raise SystemExit(f"No zone files found in {data_dir}")

    write_zone_output(zone_rows, args.zone_output.expanduser().resolve())
    write_locomotion_output(
        locomotion_rows, args.locomotion_output.expanduser().resolve()
    )
    print(
        f"Wrote {len(zone_rows)} zone rows -> {args.zone_output} "
        f"and {len(locomotion_rows)} locomotion rows -> {args.locomotion_output}"
    )


if __name__ == "__main__":
    main()
