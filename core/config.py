"""Config - shared settings and constants."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Normalize all paths to use the correct separator for the current OS
BASE_DIR = os.path.normpath(BASE_DIR)
DOWNLOADS_DIR = os.path.normpath(DOWNLOADS_DIR)
CLIPS_DIR = os.path.normpath(CLIPS_DIR)
TEMPLATES_DIR = os.path.normpath(TEMPLATES_DIR)
STATIC_DIR = os.path.normpath(STATIC_DIR)

DEFAULT_FORMAT = "bestvideo[height<=1080][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

# Platform presets
PLATFORMS = {
    "tiktok": {
        "name": "TikTok",
        "width": 1080,
        "height": 1920,
        "max_duration": 60,
        "supports_captions": True,
        "emoji": "🎵",
    },
    "youtube_short": {
        "name": "YouTube Shorts",
        "width": 1080,
        "height": 1920,
        "max_duration": 60,
        "supports_captions": True,
        "emoji": "▶️",
    },
    "instagram_reel": {
        "name": "Instagram Reels",
        "width": 1080,
        "height": 1920,
        "max_duration": 90,
        "supports_captions": True,
        "emoji": "📸",
    },
    "twitter": {
        "name": "X / Twitter",
        "width": 720,
        "height": 1280,
        "max_duration": 140,
        "supports_captions": True,
        "emoji": "🐦",
    },
}

# Caption styles
CAPTION_STYLES = {
    "default": {
        "fontsize": 42,
        "fontcolor": "white",
        "box": True,
        "boxcolor": "black@0.6",
        "y_position": 0.75,
    },
    "big": {
        "fontsize": 56,
        "fontcolor": "white",
        "box": True,
        "boxcolor": "black@0.7",
        "y_position": 0.7,
    },
    "outline": {
        "fontsize": 42,
        "fontcolor": "white",
        "borderw": 3,
        "bordercolor": "black",
        "y_position": 0.75,
    },
    "highlight": {
        "fontsize": 44,
        "fontcolor": "yellow",
        "box": True,
        "boxcolor": "black@0.5",
        "y_position": 0.75,
    },
}
