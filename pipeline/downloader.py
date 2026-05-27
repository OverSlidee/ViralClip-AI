from __future__ import annotations

import os
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

# Candidate locations where the server admin can drop a cookies.txt file.
# Checked in order; the first existing file wins.
_APP_DIR = Path(__file__).resolve().parent.parent
_COOKIE_CANDIDATES = [
    # 1. Explicit env var — highest priority.
    Path(os.environ["YOUTUBE_COOKIES_FILE"]) if os.environ.get("YOUTUBE_COOKIES_FILE") else None,
    # 2. Uploaded via the admin settings page.
    _APP_DIR / "uploads" / "cookies" / "youtube_cookies.txt",
    # 3. Dropped manually in the app root (simplest VPS setup).
    _APP_DIR / "cookies.txt",
    _APP_DIR / "youtube_cookies.txt",
]


def _server_cookies_file() -> Path | None:
    """Return the first cookies.txt that exists, or None."""
    for p in _COOKIE_CANDIDATES:
        if p is not None and p.exists():
            return p
    return None


def cookies_configured() -> bool:
    """Return True if a cookies file is available for yt-dlp."""
    return _server_cookies_file() is not None


def _is_bot_error(text: str) -> bool:
    t = text.lower()
    return any(s in t for s in _BOT_SIGNALS)


def _extract_with_fallbacks(url: str, options: dict) -> dict:
    """Try downloading, using server-side cookies automatically if present."""
    base_opts = dict(options)
    base_opts["http_headers"] = _BROWSER_HEADERS
    base_opts.setdefault("sleep_interval", 1)
    base_opts.setdefault("max_sleep_interval", 3)
    base_opts.setdefault("socket_timeout", 60)

    attempts: list[dict] = []

    cookies = _server_cookies_file()
    if cookies:
        opts = dict(base_opts)
        opts["cookiefile"] = str(cookies)
        attempts.append(opts)

    # Plain attempt — works for most videos that aren't region/age-locked.
    attempts.append(dict(base_opts))

    # Browser-cookie fallbacks (useful on desktop installs, silently skipped on VPS).
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

    if cookies:
        detail = (
            "YouTube rejected the cookies.txt (it may have expired). "
            "Please export a fresh cookies.txt from your browser while logged in to YouTube "
            "and replace the file on the server."
        )
    else:
        detail = (
            "YouTube blocked the download (bot/datacenter IP detection). "
            "Place a 'cookies.txt' exported from your YouTube-logged-in browser "
            "into the app root folder on the server, or set the "
            "YOUTUBE_COOKIES_FILE environment variable to its path."
        )
    raise RuntimeError(detail) from last_error


def download_video(
    url: str,
    output_dir: str | Path,
) -> Tuple[Path, Path, str]:
    """Download a YouTube video and extract best available audio.

    Cookies are resolved automatically from the server configuration — callers
    do not need to supply them.  See _server_cookies_file() for lookup order.

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
