from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run(cmd: List[str]) -> None:
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "FFmpeg command failed")


def _escape_subtitle_path(path: Path) -> str:
    # ffmpeg subtitles filter requires escaped backslashes and colons on Windows.
    p = str(path.resolve()).replace("\\", "\\\\")
    return p.replace(":", "\\:")


def _build_crop_filter(face_center_x: Optional[float]) -> str:
    # 9:16 crop from source height, then upscale to 1080x1920.
    if face_center_x is None:
        x_expr = "(in_w-ih*9/16)/2"
    else:
        x_expr = f"max(0,min(in_w-ih*9/16,{face_center_x:.4f}*in_w-(ih*9/16)/2))"
    # Commas inside expressions must be escaped for FFmpeg filter option parsing.
    x_expr = x_expr.replace(",", "\\,")
    return f"crop=w=ih*9/16:h=ih:x={x_expr}:y=0,scale=1080:1920"


def _broll_x_from_side(side: str) -> int:
    return 30 if side == "left" else 620


def render_clip(
    video_path: str,
    clip: Dict[str, Any],
    ass_path: str,
    output_path: str,
    face_center_x: Optional[float] = None,
    broll_items: Optional[List[Dict[str, Any]]] = None,
    broll_style: str = "safe_pip",
    prepend_hook: Optional[Dict[str, float]] = None,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    main_start = float(clip["start_time"])
    main_duration = float(clip["end_time"]) - float(clip["start_time"])
    if main_duration <= 0:
        raise RuntimeError("Clip duration must be positive.")

    crop_filter = _build_crop_filter(face_center_x)
    sub = _escape_subtitle_path(Path(ass_path))

    broll_items = broll_items or []

    has_hook = prepend_hook is not None
    hook_duration = 0.0
    if has_hook:
        hook_duration = max(
            0.0,
            float(prepend_hook["end_time"]) - float(prepend_hook["start_time"]),
        )

    if not broll_items and not has_hook:
        vf = f"{crop_filter},subtitles='{sub}'"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{main_start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{main_duration:.3f}",
            "-vf",
            vf,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(out),
        ]
        _run(cmd)
        return out

    cmd = ["ffmpeg", "-y"]

    if has_hook:
        cmd.extend(
            [
                "-ss",
                f"{float(prepend_hook['start_time']):.3f}",
                "-i",
                str(video_path),
                "-ss",
                f"{main_start:.3f}",
                "-i",
                str(video_path),
            ]
        )
    else:
        cmd.extend(["-ss", f"{main_start:.3f}", "-i", str(video_path)])

    for b in broll_items:
        cmd.extend(["-i", str(b["path"])])

    chain: List[str] = []

    if has_hook:
        chain.append(f"[0:v]{crop_filter}[hookv]")
        chain.append(f"[0:a]atrim=0:{hook_duration:.3f},asetpts=PTS-STARTPTS[hooka]")
        chain.append(f"[1:v]{crop_filter}[mainv]")
        chain.append(f"[1:a]atrim=0:{main_duration:.3f},asetpts=PTS-STARTPTS[maina]")
        chain.append("[hookv][hooka][mainv][maina]concat=n=2:v=1:a=1[base0][basea]")
        base_input_count = 2
    else:
        chain.append(f"[0:v]{crop_filter}[base0]")
        chain.append(f"[0:a]atrim=0:{main_duration:.3f},asetpts=PTS-STARTPTS[basea]")
        base_input_count = 1

    current = "base0"
    for idx, b in enumerate(broll_items, start=1):
        st = max(0.0, float(b["start"]))
        dur = max(0.5, float(b["duration"]))
        btag = f"b{idx}"
        otag = f"base{idx}"
        input_idx = base_input_count + idx - 1

        if broll_style == "fullscreen":
            chain.append(
                f"[{input_idx}:v]trim=0:{dur:.3f},setpts=PTS-STARTPTS+{st:.3f}/TB,scale=1080:1920[{btag}]"
            )
            chain.append(
                f"[{current}][{btag}]overlay=0:0:enable='between(t,{st:.3f},{st+dur:.3f})'[{otag}]"
            )
        else:
            side = str(b.get("side", "right"))
            side_x = _broll_x_from_side(side)
            chain.append(
                f"[{input_idx}:v]trim=0:{dur:.3f},setpts=PTS-STARTPTS+{st:.3f}/TB,scale=430:760[{btag}]"
            )
            chain.append(
                f"[{current}][{btag}]overlay=x={side_x}:y=180:enable='between(t,{st:.3f},{st+dur:.3f})'[{otag}]"
            )
        current = otag

    chain.append(f"[{current}]subtitles='{sub}'[vout]")

    total_duration = main_duration + hook_duration

    cmd.extend(
        [
            "-t",
            f"{total_duration:.3f}",
            "-filter_complex",
            ";".join(chain),
            "-map",
            "[vout]",
            "-map",
            "[basea]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(out),
        ]
    )

    _run(cmd)
    return out
