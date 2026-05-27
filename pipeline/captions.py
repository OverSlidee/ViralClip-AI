from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "caption_templates"


def _ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escape_ass_text(text: str) -> str:
    return text.replace("{", "\\{").replace("}", "\\}")


def _text_transform(text: str, mode: str) -> str:
    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    return text


def _line_intro_override(preset: str) -> str:
    if preset == "punch":
        return "{\\fad(30,80)\\blur0.6}"
    if preset == "smooth":
        return "{\\fad(120,120)\\blur1.0}"
    if preset == "pop":
        return "{\\fad(50,60)\\bord4}"
    return ""


def _align_for_position(base_alignment: int, position: str) -> int:
    left = {1, 4, 7}
    center = {2, 5, 8}
    right = {3, 6, 9}

    if base_alignment in left:
        axis = "left"
    elif base_alignment in right:
        axis = "right"
    else:
        axis = "center"

    if position == "top":
        return {"left": 7, "center": 8, "right": 9}[axis]
    if position == "middle":
        return {"left": 4, "center": 5, "right": 6}[axis]
    return {"left": 1, "center": 2, "right": 3}[axis]


def _group_words(words: List[Dict[str, Any]], max_words: int = 5) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for w in words:
        current.append(w)
        if len(current) >= max_words or str(w["word"]).endswith((".", "?", "!")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _load_template(name: str) -> Dict[str, Any]:
    file_path = TEMPLATE_DIR / f"{name}.json"
    if not file_path.exists():
        file_path = TEMPLATE_DIR / "viral_bold.json"

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_ass_file(
    words: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    template_name: str,
    output_path: str | Path,
    caption_energy: float = 1.0,
    segments: List[Dict[str, float]] | None = None,
    caption_position: str = "bottom",
    caption_font_scale: float = 1.0,
    caption_max_words_per_line: int = 0,
    caption_margin_v: int = 0,
    caption_text_case: str = "auto",
    caption_animation: str = "auto",
    caption_karaoke_speed: float = 1.0,
) -> Path:
    template = _load_template(template_name)

    if segments is None:
        segments = [{"start_time": clip_start, "end_time": clip_end}]

    scoped: List[Dict[str, Any]] = []
    cursor = 0.0
    for seg in segments:
        seg_start = float(seg["start_time"])
        seg_end = float(seg["end_time"])
        if seg_end <= seg_start:
            continue

        seg_words = [
            {
                "word": str(w["word"]),
                "start": max(seg_start, float(w["start"])),
                "end": min(seg_end, float(w["end"])),
            }
            for w in words
            if float(w["end"]) >= seg_start and float(w["start"]) <= seg_end
        ]

        for item in seg_words:
            item["start"] = (float(item["start"]) - seg_start) + cursor
            item["end"] = (float(item["end"]) - seg_start) + cursor
            scoped.append(item)

        cursor += seg_end - seg_start

    if not scoped:
        raise RuntimeError("No words found for clip caption range.")

    base_alignment = int(template.get("alignment", 2))
    style_alignment = _align_for_position(base_alignment, caption_position)

    font_size = int(round(float(template["font_size"]) * max(0.7, min(1.8, caption_font_scale))))
    margin_v = int(caption_margin_v) if int(caption_margin_v) > 0 else int(template["margin_v"])

    style_line = (
        "Style: Default,{font},{size},{primary},{secondary},{outline},{back},"
        "0,0,0,0,100,100,0,0,1,{outline_w},{shadow},{align},{l},{r},{v},1"
    ).format(
        font=template["font_name"],
        size=font_size,
        primary=template["primary_color"],
        secondary=template["highlight_color"],
        outline=template["outline_color"],
        back=template["back_color"],
        outline_w=template["outline_width"],
        shadow=template["shadow"],
        align=style_alignment,
        l=template["margin_l"],
        r=template["margin_r"],
        v=margin_v,
    )

    max_words = int(caption_max_words_per_line) if int(caption_max_words_per_line) > 0 else int(template.get("max_words_per_line", 5))
    groups = _group_words(scoped, max_words=max_words)

    preset = str(template.get("animation_preset", "smooth")) if caption_animation == "auto" else caption_animation
    text_case = str(template.get("text_case", "none")) if caption_text_case == "auto" else caption_text_case
    speed = max(
        0.7,
        min(
            1.7,
            float(template.get("karaoke_speed", 1.0))
            * max(0.65, min(1.5, caption_energy))
            * max(0.7, min(1.7, caption_karaoke_speed)),
        ),
    )

    dialogues: List[str] = []
    for group in groups:
        g_start = float(group[0]["start"])
        g_end = float(group[-1]["end"])

        tokens = []
        for word in group:
            base_cs = max(1, int(round((float(word["end"]) - float(word["start"])) * 100)))
            dur_cs = max(1, int(round(base_cs / speed)))
            safe_word = _escape_ass_text(_text_transform(str(word["word"]), text_case))
            tokens.append(f"{{\\kf{dur_cs}}}{safe_word}")

        text = f"{{\\an{style_alignment}}}" + _line_intro_override(preset) + " ".join(tokens)
        dialogues.append(
            f"Dialogue: 0,{_ass_time(g_start)},{_ass_time(g_end)},Default,,0,0,0,,{text}"
        )

    ass = [
        "[Script Info]",
        "Title: ViralClip Captions",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        style_line,
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        *dialogues,
    ]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ass), encoding="utf-8")
    return out
