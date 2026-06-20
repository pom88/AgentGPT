#!/usr/bin/env python3
"""
process_all.py  (scene-detection edition)

Phase 1 — ffmpeg scene detection  : find cut/fade timestamps (fast, zero LLM calls)
Phase 2 — targeted frame extraction: extract 3 frames per cut (t+0.5s, t+1.5s, t+2.5s)
Phase 3 — Gemma classification     : ask Gemma only about those candidates
Phase 4 — CSV                       : group positives → clip boundaries

Usage:
    python process_all.py /path/to/RootFolder
    python process_all.py /path/to/RootFolder --limit 1 --threshold 0.3 --output test.csv
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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── Defaults ────────────────────────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS    = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"}
DEFAULT_MODEL       = "gemma4"
DEFAULT_HOST        = "http://localhost:11434"
DEFAULT_OUTPUT      = "clips.csv"
DEFAULT_THRESHOLD   = 0.3    # scene-change sensitivity (lower = more cuts detected)
DEFAULT_WORKERS     = 2
FRAMES_PER_CUT      = 3      # sample at cut+0.5s, cut+1.5s, cut+2.5s
MIN_CUT_GAP         = 3.0    # merge cuts within this window (same fade = multiple frames)
TITLE_CARD_WINDOW   = 20.0   # max gap (s) to merge into same title card event
MIN_CLIP_SECS       = 5.0

# ── Prompt ──────────────────────────────────────────────────────────────────────────────────────

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
- Title card, text VISIBLE              → {"is_title_card": true,  "title": "Title Here"}
- Title card, text NOT YET visible      → {"is_title_card": true,  "title": null}
- NOT a title card (live scene, black)  → {"is_title_card": false, "title": null}
"""

# ── Helpers ──────────────────────────────────────────────────────────────────────────────────────

def hms(s: float) -> str:
    h, rem = divmod(max(s, 0), 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"is_title_card": False, "title": None}

# ── FFmpeg ──────────────────────────────────────────────────────────────────────────────────────

def get_duration(video_path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def detect_cuts(video_path: Path, threshold: float) -> list[float]:
    """
    Phase 1: single ffmpeg pass, no frames saved, no LLM.
    Returns sorted list of scene-change timestamps (seconds).
    """
    r = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"select=gt(scene\\,{threshold}),showinfo",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    timestamps = []
    for line in r.stderr.splitlines():
        m = re.search(r"pts_time:(\d+\.?\d*)", line)
        if m:
            timestamps.append(float(m.group(1)))
    return sorted(set(timestamps))


def merge_cuts(cuts: list[float], min_gap: float) -> list[float]:
    """Collapse cuts that are part of the same fade/dissolve into one."""
    if not cuts:
        return []
    merged = [cuts[0]]
    for t in cuts[1:]:
        if t - merged[-1] >= min_gap:
            merged.append(t)
    return merged


def extract_frame(video_path: Path, timestamp: float) -> bytes:
    """Extract one frame at timestamp, return JPEG bytes. No persistent temp dir."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-ss", str(max(timestamp, 0)),
                "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", "-y", str(tmp_path),
            ],
            capture_output=True, check=True, timeout=30,
        )
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

# ── Gemma ───────────────────────────────────────────────────────────────────────────────────────

def query_gemma(img_bytes: bytes, model: str, host: str, retries: int = 3) -> dict:
    img_b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0},
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{host}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            return _parse_json(r.json()["message"]["content"])
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

# ── Analysis ──────────────────────────────────────────────────────────────────────────────────────

def analyze_cuts(
    video_path: Path,
    cuts: list[float],
    frames_per_cut: int,
    model: str,
    host: str,
    workers: int,
    lock: threading.Lock,
) -> list[dict]:
    """
    Phase 2+3: for each cut, sample frames_per_cut frames and classify with Gemma.
    Returns detections sorted by (cut_timestamp, sample_offset).
    """
    offsets = [0.5, 1.5, 2.5][:frames_per_cut]
    samples = [(cut_t, cut_t + off) for cut_t in cuts for off in offsets]
    total = len(samples)
    done = [0]
    results = []

    def process(item):
        cut_t, sample_t = item
        try:
            img = extract_frame(video_path, sample_t)
            det = query_gemma(img, model, host)
        except Exception as e:
            det = {"is_title_card": False, "title": None}
            with lock:
                print(f"    [warn] {hms(sample_t)} → {e}")

        with lock:
            done[0] += 1
            if det.get("is_title_card"):
                title_str = f"'{det['title']}'" if det["title"] else "(text not visible yet)"
                print(f"    [{done[0]:3d}/{total}] cut@{hms(cut_t)} +{sample_t-cut_t:.1f}s → TITLE CARD {title_str}")
            elif done[0] % 10 == 0:
                print(f"    [{done[0]:3d}/{total}] cut@{hms(cut_t)} ...")

        return {
            "cut_t": cut_t,
            "sample_t": sample_t,
            "is_title_card": bool(det.get("is_title_card")),
            "title": det.get("title"),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r in as_completed({pool.submit(process, s): s for s in samples}):
            results.append(r.result())

    return sorted(results, key=lambda x: (x["cut_t"], x["sample_t"]))


def group_to_events(detections: list[dict]) -> list[dict]:
    """
    Per cut: if ANY sample is a title card, that cut is a title card event.
    Then merge adjacent events within TITLE_CARD_WINDOW.
    Returns list of {start, end, title}.
    """
    by_cut: dict[float, list[dict]] = defaultdict(list)
    for d in detections:
        by_cut[d["cut_t"]].append(d)

    card_cuts = []
    for cut_t in sorted(by_cut):
        samples = by_cut[cut_t]
        titles = [s["title"] for s in samples if s["is_title_card"] and s["title"]]
        if any(s["is_title_card"] for s in samples):
            best_title = Counter(titles).most_common(1)[0][0] if titles else None
            card_cuts.append({"cut_t": cut_t, "title": best_title})

    if not card_cuts:
        return []

    events = []
    cur = card_cuts[0].copy()
    cur["end_t"] = cur["cut_t"]
    for cc in card_cuts[1:]:
        if cc["cut_t"] - cur["end_t"] <= TITLE_CARD_WINDOW:
            cur["end_t"] = cc["cut_t"]
            if cc["title"] and not cur["title"]:
                cur["title"] = cc["title"]
        else:
            events.append(cur)
            cur = cc.copy()
            cur["end_t"] = cur["cut_t"]
    events.append(cur)

    return [{"start": e["cut_t"], "end": e["end_t"], "title": e["title"] or "Unknown"} for e in events]


def events_to_clips(events: list[dict], duration: float) -> list[dict]:
    clips = []
    for i, ev in enumerate(events):
        clip_start = ev["end"]
        clip_end = events[i + 1]["start"] if i + 1 < len(events) else duration
        if clip_end - clip_start >= MIN_CLIP_SECS:
            clips.append({"title": ev["title"], "start": clip_start, "end": clip_end})
    return clips

# ── Per-video ──────────────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    model: str,
    host: str,
    threshold: float,
    frames_per_cut: int,
    workers: int,
    writer: csv.writer,
    file_handle,
    lock: threading.Lock,
) -> int:
    duration = get_duration(video_path)
    print(f"  Duration  : {hms(duration)}")

    print(f"  Phase 1   : detecting scene changes (threshold={threshold}) ...", end="", flush=True)
    raw_cuts = detect_cuts(video_path, threshold)
    merged = merge_cuts(raw_cuts, MIN_CUT_GAP)
    print(f" {len(raw_cuts)} raw cuts → {len(merged)} candidates")

    if not merged:
        print("  No cuts detected — try lowering --threshold")
        return 0

    print(f"  Phase 2+3 : {frames_per_cut} frame(s)/cut × {len(merged)} cuts"
          f" = {frames_per_cut * len(merged)} Gemma calls ...")
    detections = analyze_cuts(video_path, merged, frames_per_cut, model, host, workers, lock)

    events = group_to_events(detections)
    clips  = events_to_clips(events, duration)

    print(f"  Result    : {len(events)} title card(s) → {len(clips)} clip(s)")
    for clip in clips:
        print(f"    {clip['title']:<45s}  {hms(clip['start'])} → {hms(clip['end'])}")
        writer.writerow([video_path.name, clip["title"], hms(clip["start"]), hms(clip["end"])])

    file_handle.flush()
    return len(clips)

# ── Resume ────────────────────────────────────────────────────────────────────────────────────────

def already_processed(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["video"] for row in csv.DictReader(f)}

# ── Main ────────────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect skit title cards using scene detection + Gemma 4 (Ollama)"
    )
    parser.add_argument("root_folder")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--host",       default=DEFAULT_HOST,
                        help=f"Ollama host (default: {DEFAULT_HOST})")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help=f"Scene change sensitivity 0.0-1.0 (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--workers",    type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel Gemma workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--limit",      type=int, default=None,
                        help="Process only first N videos (testing)")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip videos already in the output CSV")
    args = parser.parse_args()

    root = Path(args.root_folder).resolve()
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        print("No video files found.")
        sys.exit(0)

    csv_path = Path(args.output)
    done_set = already_processed(csv_path) if args.resume else set()

    print(f"Videos     : {len(videos)}")
    print(f"Model      : {args.model} @ {args.host}")
    print(f"Threshold  : {args.threshold}  |  Workers: {args.workers}")
    print(f"Output CSV : {csv_path}")
    if done_set:
        print(f"Resuming   : skipping {len(done_set)} done video(s)")

    lock = threading.Lock()
    total_clips = 0
    mode = "a" if args.resume and csv_path.exists() else "w"

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["video", "clip_name", "start", "end"])

        for idx, video in enumerate(videos, 1):
            if video.name in done_set:
                print(f"\n[{idx}/{len(videos)}] SKIP: {video.name}")
                continue
            print(f"\n[{idx}/{len(videos)}] {video.relative_to(root)}")
            try:
                total_clips += process_video(
                    video, args.model, args.host,
                    args.threshold, FRAMES_PER_CUT, args.workers,
                    writer, f, lock,
                )
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Done! {total_clips} total clips → {csv_path}")


if __name__ == "__main__":
    main()
