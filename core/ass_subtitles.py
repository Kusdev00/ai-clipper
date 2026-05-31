"""
ASS subtitle generator for AI Clipper.

Generates Advanced SubStation Alpha (.ass) subtitle files from word-level
timing data, then burns them into video via FFmpeg's `ass=` filter.

This approach is far more reliable than FFmpeg drawtext chains on Windows:
- No filter chain length limits
- Proper font rendering via libass
- Supports animations (karaoke, fade, pop, bounce)
- Clean separation of styling from video processing

Inspired by FujiwaraChoki/supoclip's caption system.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Font registry
# ---------------------------------------------------------------------------

FONTS_DIR = Path(__file__).parent.parent / "fonts"


def get_available_fonts() -> list[dict]:
    """Scan the fonts/ directory and return available font info."""
    fonts = []
    if not FONTS_DIR.exists():
        return fonts
    for ext in (".ttf", ".otf"):
        for fp in sorted(FONTS_DIR.glob(f"*{ext}")):
            fonts.append({
                "name": fp.stem,
                "filename": fp.name,
                "path": str(fp),
            })
    return fonts


def resolve_font_path(font_name: str) -> Optional[str]:
    """Resolve a font name to an absolute path with forward slashes."""
    # Try exact match first
    for ext in (".ttf", ".otf"):
        candidate = FONTS_DIR / f"{font_name}{ext}"
        if candidate.exists():
            return str(candidate).replace("\\", "/")
    # Try stem match
    for fp in FONTS_DIR.iterdir():
        if fp.stem.lower() == font_name.lower():
            return str(fp).replace("\\", "/")
    return None


# ---------------------------------------------------------------------------
# Caption templates
# ---------------------------------------------------------------------------

CAPTION_TEMPLATES = {
    "default": {
        "name": "Default",
        "description": "Clean white text with black outline",
        "font_family": "Arial",
        "font_size": 72,
        "font_color": "#FFFFFF",
        "highlight_color": "#FFD700",
        "stroke_color": "#000000",
        "stroke_width": 3,
        "background": False,
        "background_color": None,
        "animation": "none",
        "position_y": 0.78,
    },
    "hormozi": {
        "name": "Hormozi",
        "description": "Bold green highlights like Alex Hormozi",
        "font_family": "Arial",
        "font_size": 88,
        "font_color": "#FFFFFF",
        "highlight_color": "#00FF00",
        "stroke_color": "#000000",
        "stroke_width": 4,
        "background": True,
        "background_color": "#000000AA",
        "animation": "karaoke",
        "position_y": 0.75,
    },
    "mrbeast": {
        "name": "MrBeast",
        "description": "Large yellow text with red highlights",
        "font_family": "Arial",
        "font_size": 96,
        "font_color": "#FFFF00",
        "highlight_color": "#FF0000",
        "stroke_color": "#000000",
        "stroke_width": 5,
        "background": False,
        "background_color": None,
        "animation": "pop",
        "position_y": 0.72,
    },
    "tiktok": {
        "name": "TikTok",
        "description": "TikTok-style with pink highlights",
        "font_family": "Arial",
        "font_size": 80,
        "font_color": "#FFFFFF",
        "highlight_color": "#FE2C55",
        "stroke_color": "#000000",
        "stroke_width": 3,
        "background": False,
        "background_color": None,
        "animation": "karaoke",
        "position_y": 0.78,
    },
    "neon": {
        "name": "Neon",
        "description": "Glowing cyan with magenta highlights",
        "font_family": "Arial",
        "font_size": 84,
        "font_color": "#00FFFF",
        "highlight_color": "#FF00FF",
        "stroke_color": "#000066",
        "stroke_width": 3,
        "background": False,
        "background_color": None,
        "animation": "karaoke",
        "position_y": 0.75,
    },
    "minimal": {
        "name": "Minimal",
        "description": "Clean, subtle with transparent background",
        "font_family": "Arial",
        "font_size": 48,
        "font_color": "#FFFFFF",
        "highlight_color": "#CCCCCC",
        "stroke_color": None,
        "stroke_width": 0,
        "background": True,
        "background_color": "#00000080",
        "animation": "fade",
        "position_y": 0.80,
    },
    "podcast": {
        "name": "Podcast",
        "description": "Professional podcast-style captions",
        "font_family": "Arial",
        "font_size": 52,
        "font_color": "#FFFFFF",
        "highlight_color": "#FFB800",
        "stroke_color": "#333333",
        "stroke_width": 2,
        "background": True,
        "background_color": "#1A1A1ACC",
        "animation": "fade",
        "position_y": 0.78,
    },
    "bold": {
        "name": "Bold",
        "description": "Extra large with thick outline for maximum readability",
        "font_family": "Arial",
        "font_size": 88,
        "font_color": "#FFFFFF",
        "highlight_color": "#FFD700",
        "stroke_color": "#000000",
        "stroke_width": 5,
        "background": False,
        "background_color": None,
        "animation": "pop",
        "position_y": 0.72,
    },
}


def get_template_names() -> list[str]:
    return list(CAPTION_TEMPLATES.keys())


def get_template(template_name: str) -> dict:
    return CAPTION_TEMPLATES.get(template_name, CAPTION_TEMPLATES["default"])


# ---------------------------------------------------------------------------
# ASS subtitle generation
# ---------------------------------------------------------------------------

def _ass_color(hex_color: str) -> str:
    """Convert #RRGGBB or #RRGGBBAA to ASS color format &H00BBGGRR&."""
    if not hex_color:
        return "&H00FFFFFF&"
    val = hex_color.strip().lstrip("#")
    if len(val) == 8:
        # RRGGBBAA → &H00BBGGRR& (ignore alpha for now)
        r, g, b = val[0:2], val[2:4], val[4:6]
    elif len(val) == 6:
        r, g, b = val[0:2], val[2:4], val[4:6]
    else:
        return "&H00FFFFFF&"
    return f"&H00{b}{g}{r}&"


def _ass_timestamp(seconds: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escape_ass_text(text: str) -> str:
    """Escape special characters in ASS text."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def generate_ass_subtitles(
    words: list[dict],
    width: int,
    height: int,
    template_name: str = "default",
    font_path: Optional[str] = None,
) -> str:
    """
    Generate an ASS subtitle file from word-level timing data.

    Each line of text becomes a timed subtitle event. The template controls
    styling (font, colors, animation type).

    Returns the path to the generated .ass file.
    """
    template = get_template(template_name)
    font_family = template["font_family"]
    font_size = template["font_size"]
    font_color = template["font_color"]
    highlight_color = template["highlight_color"]
    stroke_color = template.get("stroke_color", "#000000")
    stroke_width = template.get("stroke_width", 2)
    animation = template.get("animation", "none")
    position_y = template.get("position_y", 0.78)
    words_per_line = 4

    # Scale font size relative to video width (base 720p like SupoClip)
    # For mobile/TikTok, we want larger text. Use 1080p as base for better mobile readability.
    scale = width / 1080
    font_size = max(36, min(120, int(font_size * scale)))

    # Group words into lines
    lines = []
    buf = []
    for w in words:
        buf.append(w)
        if len(buf) >= words_per_line:
            lines.append(buf)
            buf = []
    if buf:
        lines.append(buf)

    # Build ASS header (SupoClip-compatible)
    ass_lines = [
        "[Script Info]",
        "Title: AI Clipper Captions",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "Timer: 100.0000",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]

    # Style for normal text
    bold = 1 if template.get("background") else 0
    shadow_px = 2 if template.get("shadow") else 0
    border_style = 3 if template.get("background") else 1
    ass_lines.append(
        f"Style: Default,{font_family},{font_size},{_ass_color(font_color)},"
        f"{_ass_color(highlight_color)},{_ass_color(stroke_color)},&H00000000&,"
        f"{bold},0,0,0,100,100,0,0,{border_style},{stroke_width},{shadow_px},2,10,10,10,1"
    )

    # Style for highlighted (karaoke) text
    if animation == "karaoke":
        ass_lines.append(
            f"Style: Highlight,{font_family},{font_size},{_ass_color(highlight_color)},"
            f"{_ass_color(highlight_color)},{_ass_color(stroke_color)},&H00000000&,"
            f"{bold},0,0,0,100,100,0,0,1,{stroke_width},0,2,10,10,10,1"
        )

    ass_lines.extend(["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"])

    # Calculate Y position for captions
    base_y = int(height * position_y)
    line_height = int(font_size * 1.5)

    # Generate dialogue events
    for line_idx, line in enumerate(lines):
        if not line:
            continue

        line_start = round(line[0]["start"], 3)
        line_end = round(line[-1]["end"], 3)

        # Ensure minimum duration
        if line_end <= line_start:
            line_end = line_start + 0.5

        # Stack lines upward from base_y
        slot = line_idx % 3
        y_pos = base_y - slot * line_height
        y_pos = max(font_size + 20, y_pos)

        text = " ".join(w.get("word", "") for w in line)
        text = _escape_ass_text(text)

        start_ts = _ass_timestamp(line_start)
        end_ts = _ass_timestamp(line_end)

        if animation == "karaoke":
            # Karaoke: each word gets highlighted in sequence
            word_events = _build_karaoke_events(
                line, start_ts, end_ts, font_size, y_pos,
                font_color, highlight_color, stroke_color, stroke_width,
                font_family, bold, width, height
            )
            ass_lines.extend(word_events)
        else:
            # Static text with optional fade/pop
            margin_v = height - y_pos
            if animation == "fade":
                # Fade in/out using ASS tags
                text = f"{{\\fad(200,200)}}{text}"
            elif animation == "pop":
                # Pop effect using scale
                text = f"{{\\fscx120\\fscy120}}{text}"

            ass_lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}"
            )

    # Write to temp file
    ass_content = "\n".join(ass_lines)
    ass_file = os.path.join(tempfile.gettempdir(), f"captions_{os.getpid()}_{id(words)}.ass")
    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(ass_content)

    logger.info("Generated ASS subtitles: %s (%d lines)", ass_file, len(lines))
    return ass_file


def _build_karaoke_events(
    line: list[dict],
    start_ts: str,
    end_ts: str,
    font_size: int,
    y_pos: int,
    font_color: str,
    highlight_color: str,
    stroke_color: str,
    stroke_width: int,
    font_family: str,
    bold: int,
    width: int,
    height: int,
) -> list[str]:
    """Build karaoke-style word-by-word highlight events."""
    events = []
    margin_v = height - y_pos

    # Build the full text with per-word timing using ASS karaoke tags
    # \kf<duration> = karaoke fill (highlight) duration in centiseconds
    text_parts = []
    for word in line:
        w = _escape_ass_text(word.get("word", ""))
        duration = max(1, int((word.get("end", 0) - word.get("start", 0)) * 100))
        text_parts.append(f"{{\\kf{duration}}}{w}")

    full_text = " ".join(text_parts)

    events.append(
        f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{full_text}"
    )

    return events
