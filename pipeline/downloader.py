from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import yt_dlp
from yt_dlp.utils import DownloadError

# Set this env var on the VPS to a Netscape-format cookies.txt exported from your browser.
# Example: export YT_COOKIES_FILE=/home/user/yt-cookies.txt
_COOKIES_FILE = os.environ.get("YT_COOKIES_FILE", "")


def _extract_with_fallbacks(url: str, options: dict) -> dict:
    """Try downloading; if a cookies file is configured use it directly, otherwise retry with browser cookies."""
    # Cookie file path configured on server — use it directly, skip browser attempts.
    if _COOKIES_FILE and Path(_COOKIES_FILE).exists():
        run_opts = dict(options)
        run_opts["cookiefile"] = _COOKIES_FILE
        with yt_dlp.YoutubeDL(run_opts) as ydl:
            return ydl.extract_info(url, download=True)

    attempts = [None, "edge", "firefox", "chrome"]
    last_error: Exception | None = None

    for browser in attempts:
        run_opts = dict(options)
        if browser:
            run_opts["cookiesfrombrowser"] = (browser,)

        try:
            with yt_dlp.YoutubeDL(run_opts) as ydl:
                return ydl.extract_info(url, download=True)
        except DownloadError as exc:
            last_error = exc
            text = str(exc).lower()
            if (
                "sign in to confirm you're not a bot" not in text
                and "cookies" not in text
                and "cookie database" not in text
            ):
                raise
        except Exception as exc:
            # Some cookie backends fail when browser DB is locked (common on Chrome).
            # Keep trying other browser sources before failing.
            last_error = exc
            text = str(exc).lower()
            if "cookie database" not in text and "cookies" not in text:
                raise

    if last_error is not None:
        raise RuntimeError(
            "YouTube blocked anonymous download. Please sign in to YouTube in Chrome/Edge, then retry."
        ) from last_error

    raise RuntimeError("Failed to download video.")


def download_video(url: str, output_dir: str | Path) -> Tuple[Path, Path, str]:
    """Download a YouTube video and extract best available audio.

    Returns (video_path, audio_path, base_id).
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(target / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    info = _extract_with_fallbacks(url, ydl_opts)

    base_id = info.get("id")
    if not base_id:
        raise RuntimeError("Unable to determine YouTube video id.")

    candidates = sorted(target.glob(f"{base_id}.*"))
    video_path = None
    for p in candidates:
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            video_path = p
            break

    if video_path is None:
        raise RuntimeError("Downloaded video file was not found.")

    audio_path = target / f"{base_id}.m4a"
    if not audio_path.exists():
        # Keep this audio extraction in yt-dlp to avoid extra ffmpeg call from Python.
        audio_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(target / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                    "preferredquality": "192",
                }
            ],
        }
        _extract_with_fallbacks(url, audio_opts)

    if not audio_path.exists():
        # Fallback to mp3 extension if m4a is not available due to source format.
        mp3_path = target / f"{base_id}.mp3"
        if mp3_path.exists():
            audio_path = mp3_path
        else:
            raise RuntimeError("Audio extraction failed.")

    return video_path, audio_path, base_id
