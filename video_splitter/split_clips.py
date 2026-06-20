#!/usr/bin/env python3
"""
split_clips.py

Split compilation videos into individual clip files using a CSV from process_all.py.
Uses ffmpeg stream-copy (no re-encoding) so it's fast.

Usage:
    python split_clips.py clips.csv /path/to/videos /path/to/output_clips
    python split_clips.py clips.csv /path/to/videos /path/to/output_clips --dry-run
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def hms_to_sec(ts: str) -> float:
    p = ts.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])


def safe_filename(name: str) -> str:
    """Strip characters that cause problems in filenames across OS."""
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", name)
    return cleaned.strip(". ") or "clip"


def split_clip(video_path: Path, title: str, start: str, end: str, out_dir: Path, dry_run: bool) -> None:
    duration = hms_to_sec(end) - hms_to_sec(start)
    output = out_dir / f"{safe_filename(title)}.mp4"

    cmd = [
        "ffmpeg",
        "-ss", start,
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",                  # stream copy — fast, no quality loss
        "-avoid_negative_ts", "make_zero",
        "-y",
        str(output),
    ]

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.decode()[-300:]}")
    else:
        size_mb = output.stat().st_size / 1_048_576
        print(f"  OK  {output.name}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Split videos using a clips CSV")
    parser.add_argument("csv_file",    help="CSV produced by process_all.py")
    parser.add_argument("videos_root", help="Root folder containing source videos")
    parser.add_argument("output_dir",  help="Folder to write the split clips")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print ffmpeg commands without running them")
    args = parser.parse_args()

    csv_path    = Path(args.csv_file)
    videos_root = Path(args.videos_root)
    output_dir  = Path(args.output_dir)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Clips in CSV : {len(rows)}")
    print(f"Output dir   : {output_dir}")
    if args.dry_run:
        print("Mode         : DRY RUN (no files written)\n")

    errors = 0
    for row in rows:
        matches = list(videos_root.rglob(row["video"]))
        if not matches:
            print(f"\n[skip] {row['video']} not found under {videos_root}")
            errors += 1
            continue

        print(f"\n{row['video']} → {row['clip_name']}")
        try:
            split_clip(matches[0], row["clip_name"], row["start"], row["end"], output_dir, args.dry_run)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    print(f"\n{'='*60}")
    print(f"Done. {len(rows) - errors}/{len(rows)} clips processed.")
    if errors:
        print(f"{errors} error(s) — check output above.")


if __name__ == "__main__":
    main()
