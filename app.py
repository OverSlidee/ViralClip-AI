from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from pipeline.analyzer import find_clips, pick_hook_window, rank_clips
from pipeline.broll import choose_local_broll, fetch_pexels_broll, plan_broll_keywords
from pipeline.captions import build_ass_file
from pipeline.clipper import render_clip
from pipeline.downloader import download_video
from pipeline.face_tracker import detect_face_center_x, detect_face_timeline, suggest_safe_side
from pipeline.transcriber import transcribe


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ViralClip AI")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

TEMPLATE_META = {
    "viral_bold": {
        "name": "Viral Bold",
        "description": "Large high-contrast captions with heavy outline.",
    },
    "neon_pop": {
        "name": "Neon Pop",
        "description": "High-energy neon style for fast-paced clips.",
    },
    "minimal_clean": {
        "name": "Minimal Clean",
        "description": "Clean and subtle subtitles for premium interviews.",
    },
}


class ProcessRequest(BaseModel):
    url: str
    ollama_model: str = "llama3.1"
    whisper_model: str = "base"
    template: str = "viral_bold"
    clip_count: int = Field(default=4, ge=1, le=8)
    min_clip_duration: int = Field(default=30, ge=15, le=120)
    max_clip_duration: int = Field(default=90, ge=20, le=180)
    hook_recut: bool = False
    hook_duration: float = Field(default=4.0, ge=2.0, le=6.0)
    caption_energy: float = Field(default=1.0, ge=0.7, le=1.5)
    caption_position: str = Field(default="bottom", pattern="^(top|middle|bottom)$")
    caption_font_scale: float = Field(default=1.0, ge=0.7, le=1.8)
    caption_max_words_per_line: int = Field(default=0, ge=0, le=10)
    caption_margin_v: int = Field(default=0, ge=0, le=600)
    caption_text_case: str = Field(default="auto", pattern="^(auto|none|upper|lower)$")
    caption_animation: str = Field(default="auto", pattern="^(auto|smooth|punch|pop)$")
    caption_karaoke_speed: float = Field(default=1.0, ge=0.7, le=1.7)
    face_tracking: bool = False
    broll_enabled: bool = False
    broll_source: str = Field(default="local", pattern="^(local|pexels|both)$")
    broll_style: str = Field(default="safe_pip", pattern="^(safe_pip|fullscreen)$")
    ab_test_mode: bool = False
    pexels_api_key: str = ""


JOBS: Dict[str, Dict[str, Any]] = {}


def _emit(job_id: str, stage: str, message: str, extra: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {"stage": stage, "message": message}
    if extra:
        payload.update(extra)
    JOBS[job_id]["events"].append(payload)
    JOBS[job_id]["status"] = stage


def _clip_in_duration_window(clip: Dict[str, Any], min_seconds: int, max_seconds: int) -> bool:
    duration = float(clip["end_time"]) - float(clip["start_time"])
    return min_seconds <= duration <= max_seconds


def _avg_face_x(face_timeline: List[Dict[str, float]]) -> float | None:
    if not face_timeline:
        return None
    return sum(float(s.get("center_x", 0.5)) for s in face_timeline) / len(face_timeline)


def _variant_retention_score(
    variant: str,
    clip: Dict[str, Any],
    broll_items: List[Dict[str, Any]],
    face_timeline: List[Dict[str, float]],
    hook_enabled: bool,
) -> float:
    score = float(clip.get("hook_score", 6.0))
    duration = float(clip.get("end_time", 0.0)) - float(clip.get("start_time", 0.0))

    if hook_enabled:
        score += 1.2

    # Slight preference for mid-length clips around short-form sweet spot.
    score += max(0.0, 1.0 - abs(duration - 52.0) / 52.0)

    if not broll_items:
        return score

    if variant == "safe_pip":
        score += 1.4
        if face_timeline:
            score += 0.9
    else:
        score += 0.6
        if len(broll_items) > 2:
            score -= 0.5

        avg_x = _avg_face_x(face_timeline)
        if avg_x is not None and 0.38 <= avg_x <= 0.62:
            # Centered talking head is likely obscured by fullscreen overlays.
            score -= 0.7

    return score


def _dependency_health() -> Dict[str, Any]:
    ffmpeg_ok = False
    ffmpeg_version = ""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=8)
            ffmpeg_ok = proc.returncode == 0
            ffmpeg_version = (proc.stdout.splitlines() or [""])[0]
        except Exception:
            ffmpeg_ok = False

    ollama_models, ollama_reachable = _ollama_model_names()
    ollama_ok = ollama_reachable and len(ollama_models) > 0

    return {
        "ffmpeg": {
            "ok": ffmpeg_ok,
            "path": ffmpeg_path,
            "version": ffmpeg_version,
        },
        "ollama": {
            "ok": ollama_ok,
            "reachable": ollama_reachable,
            "models": ollama_models,
        },
    }


def _parse_ollama_list_stdout(stdout: str) -> List[str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return []

    models: List[str] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        model_name = parts[0].strip()
        if model_name and model_name.lower() != "name":
            models.append(model_name)
    return models


def _ollama_model_names() -> tuple[List[str], bool]:
    # Primary: Ollama Python client.
    try:
        client = ollama.Client(host="http://localhost:11434")
        data = client.list()
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        if models:
            return models, True
    except Exception:
        pass

    # Fallback: CLI list can include cloud-linked models like qwen3.5:397b-cloud.
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            models = _parse_ollama_list_stdout(proc.stdout)
            return models, True
    except Exception:
        pass

    return [], False


async def _run_pipeline(job_id: str, req: ProcessRequest) -> None:
    job_dir = OUTPUTS_DIR / job_id
    dl_dir = job_dir / "download"
    clips_dir = job_dir / "clips"
    temp_dir = job_dir / "temp"
    dl_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        _emit(job_id, "downloading", "Downloading source video")
        video_path, audio_path, base_id = await asyncio.to_thread(download_video, req.url, dl_dir)

        _emit(job_id, "transcribing", "Generating timestamped transcript")
        words = await asyncio.to_thread(transcribe, audio_path, req.whisper_model)

        _emit(job_id, "analyzing", "Selecting best short clips with Ollama")
        clips = await asyncio.to_thread(find_clips, words, req.ollama_model, req.clip_count)
        filtered = [
            c
            for c in clips
            if _clip_in_duration_window(c, req.min_clip_duration, req.max_clip_duration)
        ]
        if filtered:
            clips = filtered
        clips = await asyncio.to_thread(rank_clips, clips, words, req.clip_count)

        out_clips: List[Dict[str, Any]] = []
        for idx, clip in enumerate(clips, start=1):
            _emit(job_id, f"clipping_{idx}", f"Rendering clip {idx}/{len(clips)}")

            hook_window = None
            montage_segments = [
                {
                    "start_time": float(clip["start_time"]),
                    "end_time": float(clip["end_time"]),
                }
            ]
            if req.hook_recut:
                hook_window = await asyncio.to_thread(
                    pick_hook_window,
                    words,
                    float(clip["start_time"]),
                    float(clip["end_time"]),
                    req.hook_duration,
                )
                if hook_window:
                    montage_segments = [
                        {
                            "start_time": float(hook_window["start_time"]),
                            "end_time": float(hook_window["end_time"]),
                        },
                        {
                            "start_time": float(clip["start_time"]),
                            "end_time": float(clip["end_time"]),
                        },
                    ]

            main_start = float(clip["start_time"])
            main_end = float(clip["end_time"])

            ass_file = temp_dir / f"caption_{idx}.ass"
            await asyncio.to_thread(
                build_ass_file,
                words,
                main_start,
                main_end,
                req.template,
                ass_file,
                req.caption_energy,
                montage_segments,
                req.caption_position,
                req.caption_font_scale,
                req.caption_max_words_per_line,
                req.caption_margin_v,
                req.caption_text_case,
                req.caption_animation,
                req.caption_karaoke_speed,
            )

            face_center_x = None
            face_timeline = []
            if req.face_tracking:
                face_timeline = await asyncio.to_thread(
                    detect_face_timeline,
                    str(video_path),
                    main_start,
                    main_end,
                    5.0,
                )
                face_center_x = await asyncio.to_thread(
                    detect_face_center_x,
                    str(video_path),
                    main_start,
                    main_end,
                    5.0,
                )

            broll_items: List[Dict[str, Any]] = []
            if req.broll_enabled:
                plan = await asyncio.to_thread(
                    plan_broll_keywords,
                    words,
                    main_start,
                    main_end,
                    req.ollama_model,
                    3,
                )
                if req.broll_source in {"local", "both"}:
                    local = await asyncio.to_thread(
                        choose_local_broll,
                        plan,
                        BASE_DIR / "broll_library",
                    )
                    broll_items.extend(local)

                if req.broll_source in {"pexels", "both"} and req.pexels_api_key.strip():
                    pexels = await asyncio.to_thread(
                        fetch_pexels_broll,
                        plan,
                        req.pexels_api_key.strip(),
                        temp_dir / f"pexels_{idx}",
                    )
                    broll_items.extend(pexels)

                hook_offset = 0.0
                if hook_window:
                    hook_offset = float(hook_window["end_time"]) - float(hook_window["start_time"])

                for item in broll_items:
                    start_off = float(item.get("start", 0.0))
                    dur = float(item.get("duration", 2.0))
                    safe_side = "right"
                    if face_timeline:
                        safe_side = await asyncio.to_thread(
                            suggest_safe_side,
                            face_timeline,
                            start_off,
                            start_off + dur,
                            "right",
                        )
                    item["side"] = safe_side
                    item["start"] = start_off + hook_offset

            output_name = f"{base_id}_clip_{idx}.mp4"
            selected_filename = output_name
            selected_variant = req.broll_style
            variant_files: List[Dict[str, Any]] = []

            should_ab = bool(req.ab_test_mode and req.broll_enabled and broll_items)
            if should_ab:
                _emit(job_id, f"clipping_{idx}", f"A/B rendering clip {idx}: safe_pip vs fullscreen")
                variants = ["safe_pip", "fullscreen"]
                scores: Dict[str, float] = {}

                for variant in variants:
                    var_name = f"{base_id}_clip_{idx}_{variant}.mp4"
                    var_file = clips_dir / var_name
                    await asyncio.to_thread(
                        render_clip,
                        str(video_path),
                        {
                            "start_time": main_start,
                            "end_time": main_end,
                        },
                        str(ass_file),
                        str(var_file),
                        face_center_x,
                        broll_items[:3],
                        variant,
                        hook_window,
                    )

                    score = _variant_retention_score(
                        variant,
                        clip,
                        broll_items[:3],
                        face_timeline,
                        bool(hook_window),
                    )
                    scores[variant] = score
                    variant_files.append(
                        {
                            "variant": variant,
                            "filename": var_name,
                            "score": round(score, 3),
                            "preview_url": f"/api/download/{job_id}/{var_name}",
                        }
                    )

                selected_variant = max(scores, key=scores.get)
                selected_filename = f"{base_id}_clip_{idx}_{selected_variant}.mp4"
            else:
                output_file = clips_dir / output_name
                await asyncio.to_thread(
                    render_clip,
                    str(video_path),
                    {
                        "start_time": main_start,
                        "end_time": main_end,
                    },
                    str(ass_file),
                    str(output_file),
                    face_center_x,
                    broll_items[:3],
                    req.broll_style,
                    hook_window,
                )

            rendered_start = main_start
            rendered_end = main_end
            if hook_window:
                rendered_start = float(hook_window["start_time"])
                rendered_end = main_end

            out_clips.append(
                {
                    "title": clip["title"],
                    "hook_reason": clip["hook_reason"],
                    "hook_score": clip.get("hook_score"),
                    "start_time": rendered_start,
                    "end_time": rendered_end,
                    "duration": round(
                        (main_end - main_start)
                        + (
                            (float(hook_window["end_time"]) - float(hook_window["start_time"]))
                            if hook_window
                            else 0.0
                        ),
                        2,
                    ),
                    "original_start_time": main_start,
                    "hook_recut": bool(hook_window),
                    "montage_style": "prepend_hook" if hook_window else "none",
                    "filename": selected_filename,
                    "preview_url": f"/api/download/{job_id}/{selected_filename}",
                    "selected_broll_variant": selected_variant,
                    "ab_test_mode": should_ab,
                    "variant_outputs": variant_files,
                }
            )

        JOBS[job_id]["clips"] = out_clips
        _emit(job_id, "done", "All clips rendered successfully", {"count": len(out_clips)})
    except Exception as exc:
        JOBS[job_id]["error"] = str(exc)
        _emit(job_id, "error", str(exc))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/models")
async def models() -> Dict[str, Any]:
    models, reachable = _ollama_model_names()
    return {"models": models, "reachable": reachable}


@app.get("/api/templates")
async def templates_meta() -> Dict[str, Any]:
    return {"templates": TEMPLATE_META}


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    deps = await asyncio.to_thread(_dependency_health)
    return {"ok": deps["ffmpeg"]["ok"] and deps["ollama"]["ok"], "dependencies": deps}


@app.post("/api/process")
async def process_video(req: ProcessRequest) -> Dict[str, str]:
    if req.min_clip_duration > req.max_clip_duration:
        raise HTTPException(status_code=400, detail="min_clip_duration cannot be greater than max_clip_duration")

    try:
        available_models, reachable = _ollama_model_names()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Ollama is not reachable. Start Ollama and try again.",
        ) from exc

    if not reachable:
        raise HTTPException(
            status_code=400,
            detail="Ollama is not reachable. Start Ollama and try again.",
        )

    if not available_models:
        raise HTTPException(
            status_code=400,
            detail="No Ollama models found. Run: ollama pull llama3.1",
        )

    if req.ollama_model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.ollama_model}' is not installed. Available: {', '.join(available_models)}",
        )

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "events": [{"stage": "queued", "message": "Job queued"}],
        "clips": [],
        "error": None,
    }
    asyncio.create_task(_run_pipeline(job_id, req))
    return {"job_id": job_id}


@app.get("/api/progress/{job_id}")
async def progress(job_id: str) -> StreamingResponse:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        idx = 0
        idle_ticks = 0
        while True:
            events = JOBS[job_id]["events"]
            while idx < len(events):
                payload = json.dumps(events[idx])
                yield f"event: progress\ndata: {payload}\n\n"
                idx += 1
                idle_ticks = 0

            status = JOBS[job_id]["status"]
            if status in {"done", "error"} and idx >= len(events):
                break

            # Heartbeat keeps EventSource connections alive during long stages.
            idle_ticks += 1
            if idle_ticks >= 20:
                yield ": ping\n\n"
                idle_ticks = 0

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/clips/{job_id}")
async def clips(job_id: str) -> Dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": JOBS[job_id]["status"], "clips": JOBS[job_id]["clips"], "error": JOBS[job_id]["error"]}


@app.get("/api/download/{job_id}/{filename}")
async def download_clip(job_id: str, filename: str):
    target = OUTPUTS_DIR / job_id / "clips" / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(target), filename=filename, media_type="video/mp4")
