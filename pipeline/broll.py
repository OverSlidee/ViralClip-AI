from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import ollama
import requests


BrollPlanItem = Dict[str, Any]


def _safe_json(text: str) -> Any:
    match = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    payload = match[0].strip() if match else text.strip()
    return json.loads(payload)


def plan_broll_keywords(
    words: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    ollama_model: str,
    max_items: int = 3,
) -> List[BrollPlanItem]:
    scoped_words = [w for w in words if float(w["start"]) >= clip_start and float(w["end"]) <= clip_end]
    transcript = " ".join(str(w["word"]) for w in scoped_words)

    prompt = f"""
Create up to {max_items} B-roll ideas for this transcript clip.
Return JSON only with this schema:
[
  {{"start_offset": 5.0, "duration": 2.5, "keywords": ["city", "traffic"]}}
]
Rules:
- start_offset and duration are in seconds relative to clip start.
- Keep duration between 1.5 and 4.0 seconds.
- Items cannot overlap.
Transcript:
{transcript}
""".strip()

    client = ollama.Client(host="http://localhost:11434")
    response = client.chat(
        model=ollama_model,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.4},
    )

    raw = response["message"]["content"]
    data = _safe_json(raw)
    if not isinstance(data, list):
        return []

    out: List[BrollPlanItem] = []
    last_end = -1.0
    for item in data:
        if not isinstance(item, dict):
            continue
        start_offset = float(item.get("start_offset", 0.0))
        duration = max(1.5, min(4.0, float(item.get("duration", 2.0))))
        keywords = item.get("keywords", [])
        if not isinstance(keywords, list):
            continue
        keywords = [str(k).strip().lower() for k in keywords if str(k).strip()]
        if not keywords:
            continue

        if start_offset < 0 or (start_offset + duration) > max(0.0, clip_end - clip_start):
            continue
        if start_offset < last_end:
            continue

        out.append({"start_offset": start_offset, "duration": duration, "keywords": keywords})
        last_end = start_offset + duration

    return out[:max_items]


def choose_local_broll(plan: List[BrollPlanItem], library_dir: str | Path) -> List[Dict[str, Any]]:
    lib = Path(library_dir)
    if not lib.exists():
        return []

    pool = [p for p in lib.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}]
    chosen: List[Dict[str, Any]] = []

    for item in plan:
        picked = None
        joined_keywords = " ".join(item["keywords"])
        for clip in pool:
            name = clip.stem.lower().replace("_", " ").replace("-", " ")
            if any(k in name for k in item["keywords"]) or joined_keywords in name:
                picked = clip
                break
        if picked is None and pool:
            picked = pool[0]

        if picked is not None:
            chosen.append(
                {
                    "path": str(picked),
                    "start": float(item["start_offset"]),
                    "duration": float(item["duration"]),
                }
            )

    return chosen


def fetch_pexels_broll(
    plan: List[BrollPlanItem],
    api_key: str,
    output_dir: str | Path,
) -> List[Dict[str, Any]]:
    if not api_key:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": api_key}
    selected: List[Dict[str, Any]] = []

    for idx, item in enumerate(plan):
        query = " ".join(item["keywords"][:3])
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            params={"query": query, "per_page": 1, "orientation": "portrait"},
            headers=headers,
            timeout=20,
        )
        if resp.status_code != 200:
            continue

        data = resp.json()
        videos = data.get("videos") or []
        if not videos:
            continue

        files = videos[0].get("video_files") or []
        if not files:
            continue

        source_url = files[0].get("link")
        if not source_url:
            continue

        file_path = out_dir / f"pexels_{idx}.mp4"
        dl = requests.get(source_url, timeout=40)
        if dl.status_code != 200:
            continue

        file_path.write_bytes(dl.content)
        selected.append(
            {
                "path": str(file_path),
                "start": float(item["start_offset"]),
                "duration": float(item["duration"]),
            }
        )

    return selected
