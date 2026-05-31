"""Video Downloader - handles YouTube, Twitch VODs, and other yt-dlp supported URLs."""

import os
import re
import json
import subprocess
from pathlib import Path
from typing import Optional

from core.config import DOWNLOADS_DIR, DEFAULT_FORMAT


def get_video_info(url: str) -> dict:
    """Fetch video metadata without downloading."""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise Exception(f"Failed to fetch info: {result.stderr.strip()}")
    return json.loads(result.stdout.strip())


def download_video(url: str, job_id: str, progress_callback=None) -> str:
    """Download video to downloads dir. Returns filepath."""
    output_dir = os.path.join(DOWNLOADS_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "source.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", DEFAULT_FORMAT,
        "--output", output_template,
        "--no-warnings",
        "--newline",
        "--progress",
    ]

    # Add Twitch-specific options
    if "twitch.tv" in url:
        cmd += ["--fixup", "force"]

    cmd.append(url)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        line = line.strip()
        if progress_callback and "[download]" in line:
            # Parse progress percentage
            match = re.search(r"(\d+\.?\d*)%", line)
            if match:
                progress_callback(float(match.group(1)))

    process.wait()

    if process.returncode != 0:
        stderr = process.stderr.read()
        raise Exception(f"Download failed: {stderr.strip()}")

    # Find the downloaded file
    for f in os.listdir(output_dir):
        if f.startswith("source."):
            filepath = os.path.join(output_dir, f)
            return filepath

    raise Exception("Download completed but file not found")


def download_audio(url: str, job_id: str) -> str:
    """Download audio-only for Whisper transcription."""
    output_dir = os.path.join(DOWNLOADS_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "audio.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestaudio/best",
        "--output", output_template,
        "--no-warnings",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise Exception(f"Audio download failed: {result.stderr.strip()}")

    for f in os.listdir(output_dir):
        if f.startswith("audio."):
            return os.path.join(output_dir, f)

    raise Exception("Audio download completed but file not found")
