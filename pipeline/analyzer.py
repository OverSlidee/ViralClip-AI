from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import ollama


ClipItem = Dict[str, Any]

HOOK_WORDS = {
    "secret",
    "mistake",
    "truth",
    "crazy",
    "insane",
    "why",
    "how",
    "never",
    "always",
    "nobody",
    "everyone",
    "biggest",
    "fast",
    "easy",
}


def _extract_json_payload(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return text.strip()


def _parse_time_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    text = text.replace("seconds", "s").replace("second", "s")
    if text.endswith("s"):
        text = text[:-1].strip()

    try:
        return float(text)
    except ValueError:
        pass

    if ":" in text:
        parts = text.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 2:
            mm, ss = nums
            return mm * 60.0 + ss
        if len(nums) == 3:
            hh, mm, ss = nums
            return hh * 3600.0 + mm * 60.0 + ss

    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    return None


def _extract_clip_list(data: Any) -> Any:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["clips", "moments", "highlights", "results", "data", "items"]:
            value = data.get(key)
            if isinstance(value, list):
                return value
    return data


def _normalize_clip_item(c: Dict[str, Any]) -> ClipItem | None:
    title = str(c.get("title") or c.get("headline") or c.get("name") or "Untitled").strip() or "Untitled"
    reason = str(
        c.get("hook_reason")
        or c.get("reason")
        or c.get("why")
        or c.get("justification")
        or "Strong hook and clear story."
    ).strip()

    score_raw = c.get("hook_score", c.get("score", c.get("viral_score", 7.0)))
    try:
        hook_score = float(score_raw)
    except Exception:
        hook_score = 7.0

    start_time = _parse_time_value(
        c.get("start_time", c.get("start", c.get("from", c.get("startTime", c.get("clip_start")))))
    )
    end_time = _parse_time_value(
        c.get("end_time", c.get("end", c.get("to", c.get("endTime", c.get("clip_end")))))
    )
    duration = _parse_time_value(c.get("duration", c.get("clip_duration")))

    if start_time is None:
        return None
    if end_time is None and duration is not None:
        end_time = start_time + duration
    if end_time is None:
        return None

    length = end_time - start_time
    if length < 20 or length > 120:
        return None

    return {
        "title": title,
        "hook_reason": reason,
        "hook_score": max(1.0, min(10.0, hook_score)),
        "start_time": max(0.0, float(start_time)),
        "end_time": float(end_time),
    }


def _parse_ollama_json(content: str) -> Any:
    candidates: List[str] = []
    payload = _extract_json_payload(content)
    if payload:
        candidates.append(payload)

    raw = content.strip()
    if raw and raw not in candidates:
        candidates.append(raw)

    # Some models wrap JSON with extra commentary. Try to recover first JSON block.
    block_matches = re.findall(r"(\[[\s\S]*\]|\{[\s\S]*\})", raw)
    for b in block_matches:
        b = b.strip()
        if b and b not in candidates:
            candidates.append(b)

    last_error: Exception | None = None
    for text in candidates:
        try:
            return json.loads(text)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Empty Ollama response.")


def _fallback_clips(words: List[Dict[str, Any]], max_clips: int) -> List[ClipItem]:
    if not words:
        return []

    last_end = max(float(w.get("end", w.get("start", 0.0))) for w in words)
    if last_end <= 0:
        return []

    window = max(30.0, min(75.0, last_end / max(1, max_clips)))
    candidates: List[ClipItem] = []

    starts = []
    t = 0.0
    while t + 20.0 <= last_end:
        starts.append(t)
        t += 8.0

    for start in starts:
        end = min(last_end, start + window)
        if end - start < 20.0:
            continue

        scoped = [w for w in words if start <= float(w.get("start", 0.0)) <= end]
        if len(scoped) < 8:
            continue

        hook_hits = 0
        punct = 0
        for w in scoped:
            txt = str(w.get("word", "")).strip().lower()
            if txt.strip(".,!?\"'") in HOOK_WORDS:
                hook_hits += 1
            if any(ch in txt for ch in ["?", "!"]):
                punct += 1

        score = min(10.0, 4.0 + hook_hits * 0.8 + punct * 0.5)
        preview = " ".join(str(w.get("word", "")) for w in scoped[:8]).strip()
        title = preview[:70] if preview else "Auto-selected moment"

        candidates.append(
            {
                "title": title,
                "hook_reason": "Fallback selection from transcript structure.",
                "hook_score": max(1.0, score),
                "start_time": start,
                "end_time": end,
            }
        )

    if not candidates:
        end = min(last_end, 45.0)
        if end > 20.0:
            candidates.append(
                {
                    "title": "Auto-selected opening segment",
                    "hook_reason": "Fallback clip when model output is unusable.",
                    "hook_score": 6.0,
                    "start_time": 0.0,
                    "end_time": end,
                }
            )

    candidates.sort(key=lambda x: float(x.get("hook_score", 0.0)), reverse=True)
    selected: List[ClipItem] = []
    for clip in candidates:
        if len(selected) >= max_clips:
            break
        overlap = False
        for prev in selected:
            ov = min(float(prev["end_time"]), float(clip["end_time"])) - max(
                float(prev["start_time"]), float(clip["start_time"])
            )
            if ov > 10.0:
                overlap = True
                break
        if not overlap:
            selected.append(clip)

    return selected


def _validate_clips(clips: Any) -> List[ClipItem]:
    clips = _extract_clip_list(clips)
    if not isinstance(clips, list):
        raise RuntimeError("Ollama response JSON is not a list.")

    clean: List[ClipItem] = []
    for c in clips:
        if not isinstance(c, dict):
            continue
        normalized = _normalize_clip_item(c)
        if normalized:
            clean.append(normalized)

    if not clean:
        raise RuntimeError("No valid clips produced by Ollama.")

    clean.sort(key=lambda x: float(x["start_time"]))

    deduped: List[ClipItem] = []
    for clip in clean:
        if not deduped:
            deduped.append(clip)
            continue

        prev = deduped[-1]
        overlap = min(float(prev["end_time"]), float(clip["end_time"])) - max(
            float(prev["start_time"]), float(clip["start_time"])
        )
        if overlap > 8.0:
            continue
        deduped.append(clip)

    return deduped


def find_clips(words: List[Dict[str, Any]], ollama_model: str = "llama3.1", max_clips: int = 4) -> List[ClipItem]:
    transcript_lines: List[str] = []
    for item in words:
        transcript_lines.append(f"[{item['start']:.2f}] {item['word']}")

    transcript = " ".join(transcript_lines)

    prompt = f"""
You are a short-form content editor.
Given this timestamped transcript, pick the {max_clips} strongest viral moments.
Rules:
- Each clip must be 30 to 90 seconds when possible.
- Pick clips with a strong hook, emotional insight, conflict, surprise, or practical value.
- Prefer clips that can stand alone with minimal context.
- Do not overlap clips heavily.
- Start near a sentence boundary.
- End on a completed thought.
- Keep thematic diversity across clips.
Return JSON only. No markdown.

Schema:
[
    {{"title": "...", "start_time": 12.3, "end_time": 56.8, "hook_reason": "...", "hook_score": 1-10}}
]

Transcript:
{transcript}
""".strip()

    try:
        client = ollama.Client(host="http://localhost:11434")
        response = client.chat(
            model=ollama_model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )

        content = response["message"]["content"]
        clips = _parse_ollama_json(content)
        validated = _validate_clips(clips)
        if validated:
            return validated[:max_clips]
    except Exception:
        pass

    fallback = _fallback_clips(words, max_clips)
    if fallback:
        return fallback[:max_clips]

    raise RuntimeError("No valid clips produced by Ollama.")


def pick_hook_window(
    words: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    hook_duration: float = 4.0,
) -> Dict[str, float] | None:
    """Pick a high-impact intro window inside a clip.

    The scoring favors question/exclamation punctuation and high-impact words.
    """
    scoped = [w for w in words if clip_start <= float(w.get("start", 0.0)) <= clip_end]
    if not scoped:
        return None

    effective = max(2.0, min(6.0, hook_duration))
    best_score = -1.0
    best_start = clip_start

    for w in scoped:
        start = float(w["start"])
        end = min(clip_end, start + effective)
        score = 0.0

        for token in scoped:
            t = float(token["start"])
            if start <= t <= end:
                text = str(token.get("word", "")).strip().lower()
                score += 1.0
                if any(ch in text for ch in ["?", "!"]):
                    score += 3.0
                if text.strip(".,!?\"'") in HOOK_WORDS:
                    score += 2.5

        if score > best_score:
            best_score = score
            best_start = start

    best_end = min(clip_end, best_start + effective)
    if best_end - best_start < 1.5:
        return None

    if best_start <= clip_start + 1.0:
        return None

    return {"start_time": best_start, "end_time": best_end}


def rank_clips(clips: List[ClipItem], words: List[Dict[str, Any]], top_k: int = 4) -> List[ClipItem]:
    """Score clip candidates with transcript heuristics and return best ones."""

    def _score(clip: ClipItem) -> float:
        c_start = float(clip["start_time"])
        c_end = float(clip["end_time"])
        duration = max(1.0, c_end - c_start)
        scoped = [w for w in words if c_start <= float(w.get("start", 0.0)) <= c_end]
        if not scoped:
            return 0.0

        punct = 0
        hook_hits = 0
        uniq = set()
        for item in scoped:
            txt = str(item.get("word", "")).strip().lower()
            uniq.add(txt)
            if any(ch in txt for ch in ["?", "!", ":"]):
                punct += 1
            if txt.strip(".,!?\"'") in HOOK_WORDS:
                hook_hits += 1

        lexical = len(uniq) / max(1, len(scoped))
        density = len(scoped) / duration
        target = 55.0
        duration_fit = max(0.0, 1.0 - abs(duration - target) / target)
        llm_score = float(clip.get("hook_score", 6.0)) / 10.0

        return (
            (llm_score * 4.0)
            + (duration_fit * 2.5)
            + (lexical * 1.5)
            + (min(2.0, density / 3.2) * 1.0)
            + (min(3.0, punct / 6.0) * 0.8)
            + (min(3.0, hook_hits / 5.0) * 1.0)
        )

    ranked = sorted(clips, key=_score, reverse=True)
    return ranked[:top_k]
