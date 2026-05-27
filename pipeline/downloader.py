from __future__ import annotations

from pathlib import Path
from typing import Tuple

import yt_dlp
from yt_dlp.utils import DownloadError

# VPS-friendly headers that reduce the chance of being fingerprinted as a bot.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_BOT_SIGNALS = (
    "sign in to confirm",
    "cookies",
    "cookie database",
    "blocked",
    "bot",
    "too many requests",
)


def _is_bot_error(text: str) -> bool:
    t = text.lower()
    return any(s in t for s in _BOT_SIGNALS)


def _extract_with_fallbacks(
    url: str,
    options: dict,
    cookies_file: str | Path | None = None,
) -> dict:
    """Try downloading with optional cookies file first, then browser cookie extraction."""
    base_opts = dict(options)
    base_opts["http_headers"] = _BROWSER_HEADERS
    base_opts.setdefault("sleep_interval", 1)
    base_opts.setdefault("max_sleep_interval", 3)
    base_opts.setdefault("socket_timeout", 60)

    # Build attempt list: cookies-file first if supplied, then browser fallbacks on non-VPS.
    attempts: list[dict] = []

    if cookies_file and Path(cookies_file).exists():
        opts = dict(base_opts)
        opts["cookiefile"] = str(cookies_file)
        attempts.append(opts)

    # Plain attempt (no cookies) — may work for unlocked videos.
    attempts.append(dict(base_opts))

    # Browser-cookie fallbacks (only useful on desktops, harmless on VPS — they'll just fail fast).
    for browser in ("edge", "firefox", "chrome"):
        opts = dict(base_opts)
        opts["cookiesfrombrowser"] = (browser,)
        attempts.append(opts)

    last_error: Exception | None = None
    for run_opts in attempts:
        try:
            with yt_dlp.YoutubeDL(run_opts) as ydl:
                return ydl.extract_info(url, download=True)
        except DownloadError as exc:
            last_error = exc
            if not _is_bot_error(str(exc)):
                raise
        except Exception as exc:
            last_error = exc
            if not _is_bot_error(str(exc)):
                raise

    if cookies_file and Path(cookies_file).exists():
        detail = "The supplied cookies.txt was rejected by YouTube. Please export a fresh one."
    else:
        detail = (
            "YouTube blocked the download (bot detection). "
            "Export a cookies.txt from your browser while logged in to YouTube "
            "and upload it in the Cookies section of the app."
        )
    raise RuntimeError(detail) from last_error


def download_video(
    url: str,
    output_dir: str | Path,
    cookies_file: str | Path | None = None,
) -> Tuple[Path, Path, str]:
    """Download a YouTube video and extract best available audio.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save files into.
        cookies_file: Optional path to a Netscape-format cookies.txt file.
                      Required on VPS/server environments to bypass bot detection.

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

    info = _extract_with_fallbacks(url, ydl_opts, cookies_file)

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
        _extract_with_fallbacks(url, audio_opts, cookies_file)

    if not audio_path.exists():
        # Fallback to mp3 extension if m4a is not available due to source format.
        mp3_path = target / f"{base_id}.mp3"
        if mp3_path.exists():
            audio_path = mp3_path
        else:
            raise RuntimeError("Audio extraction failed.")

    return video_path, audio_path, base_id
