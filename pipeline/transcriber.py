from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from faster_whisper import WhisperModel


WordItem = Dict[str, float | str]


def transcribe(audio_path: str | Path, model_size: str = "base") -> List[WordItem]:
    """Return a flat word list with timestamps.

    Each item: {"word": str, "start": float, "end": float}
    """
    # Prefer CPU so transcription works on machines without CUDA/CuBLAS.
    # If CPU init fails for any reason, retry with the safest available settings.
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception:
        model = WhisperModel(model_size, compute_type="int8")

    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    words: List[WordItem] = []
    for segment in segments:
        if not segment.words:
            continue
        for w in segment.words:
            if w.start is None or w.end is None:
                continue
            text = (w.word or "").strip()
            if not text:
                continue
            words.append(
                {
                    "word": text,
                    "start": float(w.start),
                    "end": float(w.end),
                }
            )

    if not words:
        raise RuntimeError("No words returned by transcription.")

    return words
