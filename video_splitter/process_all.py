#!/usr/bin/env python3
"""
process_all.py

Detect skit title cards in compilation videos using a local Gemma 4 model via Ollama.
Scans a root folder recursively, queries Gemma for each sampled frame, and writes a CSV.

Usage:
    python process_all.py /path/to/RootFolder
    python process_all.py /path/to/RootFolder --interval 5 --model gemma4 --output clips.csv
    python process_all.py /path/to/RootFolder --resume   # skip already-processed videos
"""

import argparse
import base64
import csv
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ─── Defaults ───────────────────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}

DEFAULT_MODEL    = "gemma4"                  # run `ollama list` to check your exact name
DEFAULT_HOST     = "http://localhost:11434"
DEFAULT_INTERVAL = 5                         # seconds between sampled frames
DEFAULT_WORKERS  = 2                         # parallel Gemma API calls
DEFAULT_OUTPUT   = "clips.csv"

GAP_THRESHOLD    = 20.0  # max gap (s) between two title-card frames to be same card
MIN_CLIP_SECS    = 5.0   # discard clips shorter than this (likely false positives)

# ─── Prompt ────────────────────────────────────────────────────────────────────────────────

PROMPT = """\
Look at this video frame carefully.

Is this a TITLE CARD or TITLE SCREEN that introduces a new skit, segment, or short film?

Title cards typically show:
- A title or name of the upcoming segment, prominently displayed
- A decorative background, frame, or stylized design
- Little or no live action — it is a static or lightly animated screen

Respond ONLY with JSON, no other text:

{"is_title_card": true or false, "title": "The Exact Title Text or null"}

Rules:
- Title card, text VISIBLE   → {"is_title_card": true,  "title": "Title Here"}
- Title card, text NOT YET visible (still animating in) → {"is_title_card": true,  "title": null}
- NOT a title card (live scene, black screen, credits)  → {"is_title_card": false, "title": null}
"""

# ─── Timestamp helpers ──────────────────────────────────────────────────────────────────────

def hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def hms_to_sec(ts: str) -> float:
    p = ts.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

# ─── FFmpeg helpers ───────────────────────────────────────────────────────────────────────────

def get_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_frames(video_path: Path, interval: int, output_dir: Path) -> list[tuple[float, Path]]:
    """
    Single ffmpeg call: extract 1 frame every `interval` seconds.
    Returns sorted list of (timestamp_seconds, jpeg_path).
    """
    pattern = str(output_dir / "frame_%06d.jpg")
    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",   # high quality so text is legible
            "-y", pattern,
        ],
        capture_output=True, check=True,
    )
    frames = sorted(output_dir.glob("frame_*.jpg"))
    # Frame N (1-indexed) → timestamp (N-1) * interval
    return [(idx * interval, f) for idx, f in enumerate(frames)]

# ─── Gemma API ───────────────────────────────────────────────────────────────────────────────

def _parse_response(text: str) -> dict:
    """Extract JSON from Gemma output, tolerating markdown fences and stray text."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"is_title_card": False, "title": None}


def query_gemma(frame_path: Path, model: str, host: str, retries: int = 3) -> dict:
    with open(frame_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0},
    }

    for attempt in range(retries):
        try:
            resp = requests.post(f"{host}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            return _parse_response(resp.json()["message"]["content"])
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

# ─── Analysis pipeline ────────────────────────────────────────────────────────────────────────────

def analyze_frames(
    frames: list[tuple[float, Path]],
    model: str,
    host: str,
    workers: int,
    lock: threading.Lock,
) -> list[dict]:
    """Run Gemma on every frame (parallel). Returns detections sorted by timestamp."""
    results = []
    total = len(frames)
    done = [0]

    def process(item):
        timestamp, frame_path = item
        try:
            det = query_gemma(frame_path, model, host)
        except Exception as e:
            det = {"is_title_card": False, "title": None}
            with lock:
                print(f"    [warn] {hms(timestamp)} → {e}")

        with lock:
            done[0] += 1
            if det.get("is_title_card"):
                label = f"TITLE CARD → '{det['title']}'" if det["title"] else "TITLE CARD (text not visible yet)"
                print(f"    [{done[0]:4d}/{total}] {hms(timestamp)} {label}")
            elif done[0] % 50 == 0:
                print(f"    [{done[0]:4d}/{total}] {hms(timestamp)} ...")

        return {
            "timestamp": timestamp,
            "is_title_card": bool(det.get("is_title_card")),
            "title": det.get("title"),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in as_completed({pool.submit(process, f): f for f in frames}):
            results.append(result.result())

    return sorted(results, key=lambda x: x["timestamp"])


def group_title_cards(detections: list[dict]) -> list[dict]:
    """Merge consecutive title-card frames into single card events {start, end, title}."""
    events = []
    i = 0

    while i < len(detections):
        if not detections[i]["is_title_card"]:
            i += 1
            continue

        start = detections[i]["timestamp"]
        titles: list[str] = []
        j = i

        while j < len(detections):
            det = detections[j]
            gap = det["timestamp"] - detections[j - 1]["timestamp"] if j > i else 0
            if gap > GAP_THRESHOLD:
                break
            if det["is_title_card"]:
                if det["title"]:
                    titles.append(det["title"])
                j += 1
            else:
                break

        end = detections[j - 1]["timestamp"]
        best_title = Counter(titles).most_common(1)[0][0] if titles else "Unknown"
        events.append({"start": start, "end": end, "title": best_title})
        i = j

    return events


def title_cards_to_clips(events: list[dict], duration: float) -> list[dict]:
    """Convert title card events → clip boundaries. Clip N starts after card N ends."""
    clips = []
    for idx, ev in enumerate(events):
        start = ev["end"]
        end = events[idx + 1]["start"] if idx + 1 < len(events) else duration
        if (end - start) < MIN_CLIP_SECS:
            continue
        clips.append({"title": ev["title"], "start": start, "end": end})
    return clips

# ─── Per-video processing ─────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    model: str,
    host: str,
    interval: int,
    workers: int,
    writer: csv.writer,
    file_handle,
    lock: threading.Lock,
) -> int:
    duration = get_duration(video_path)
    print(f"  Duration : {hms(duration)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Frames   : extracting 1/{interval}s ... ", end="", flush=True)
        frames = extract_frames(video_path, interval, Path(tmpdir))
        print(f"{len(frames)} frames")

        print(f"  Gemma    : analyzing {len(frames)} frames with {model} ({workers} worker(s))...")
        detections = analyze_frames(frames, model, host, workers, lock)

    events = group_title_cards(detections)
    clips  = title_cards_to_clips(events, duration)

    print(f"  Result   : {len(clips)} clip(s) found")
    for clip in clips:
        print(f"    {clip['title']:<45s}  {hms(clip['start'])} → {hms(clip['end'])}")
        writer.writerow([video_path.name, clip["title"], hms(clip["start"]), hms(clip["end"])])

    file_handle.flush()
    return len(clips)

# ─── Resume support ────────────────────────────────────────────────────────────────────────────

def already_processed(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["video"] for row in csv.DictReader(f)}

# ─── Entry point ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect skit title cards in compilation videos using Gemma 4 (Ollama)"
    )
    parser.add_argument("root_folder", help="Root folder to scan recursively for videos")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help=f"Ollama model name — run `ollama list` to confirm (default: {DEFAULT_MODEL})")
    parser.add_argument("--host",     default=DEFAULT_HOST,
                        help=f"Ollama HTTP host (default: {DEFAULT_HOST})")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help=f"Seconds between sampled frames (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--workers",  type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel Gemma API workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--output",   default=DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip videos already present in the output CSV")
    args = parser.parse_args()

    root = Path(args.root_folder).resolve()
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        print("No video files found.")
        sys.exit(0)

    csv_path = Path(args.output)
    done_set = already_processed(csv_path) if args.resume else set()

    print(f"Videos found : {len(videos)}")
    print(f"Output CSV   : {csv_path}")
    print(f"Model        : {args.model} @ {args.host}")
    print(f"Frame rate   : 1 frame every {args.interval}s")
    print(f"Workers      : {args.workers}")
    if done_set:
        print(f"Resuming     : skipping {len(done_set)} already-processed video(s)")

    lock = threading.Lock()
    total_clips = 0
    mode = "a" if args.resume and csv_path.exists() else "w"

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["video", "clip_name", "start", "end"])

        for idx, video in enumerate(videos, 1):
            if video.name in done_set:
                print(f"\n[{idx}/{len(videos)}] SKIP (already done): {video.name}")
                continue

            print(f"\n[{idx}/{len(videos)}] {video.relative_to(root)}")
            try:
                total_clips += process_video(
                    video, args.model, args.host,
                    args.interval, args.workers,
                    writer, f, lock,
                )
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Done! {total_clips} total clips → {csv_path}")


if __name__ == "__main__":
    main()
