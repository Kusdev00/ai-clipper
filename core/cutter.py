"""
Cutter - OpusClip-grade video editing engine.

Features:
  - AI Reframe / Face-Aware Crop via OpenCV
  - Animated TikTok-style word-by-word captions
  - Brand templates with presets
  - Multi-platform export (TikTok, YouTube Shorts, Instagram Reels, X/Twitter)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from core.config import CLIPS_DIR, PLATFORMS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Detect ffmpeg binary — handle WSL running Windows Python
def _detect_ffmpeg():
    """Find the best available ffmpeg binary."""
    # Check env override first
    env_ff = os.environ.get("FFMPEG_BIN")
    if env_ff:
        return env_ff

    # If running inside WSL, use the Linux ffmpeg
    if "WSL_DISTRO_NAME" in os.environ:
        for candidate in ["/usr/bin/ffmpeg", "ffmpeg"]:
            try:
                r = subprocess.run([candidate, "-version"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return "ffmpeg"

    # Windows native — try to find ffmpeg
    if os.name == "nt":
        # Build candidate list: known paths + search WinGet packages dir
        win_candidates = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            "ffmpeg",  # fallback to PATH
        ]
        # Auto-detect WinGet ffmpeg (version-independent)
        winget_ffmpeg = os.path.expandvars(
            r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
        )
        if os.path.isdir(winget_ffmpeg):
            for entry in os.listdir(winget_ffmpeg):
                if "ffmpeg" in entry.lower():
                    pkg_dir = os.path.join(winget_ffmpeg, entry)
                    # Check direct bin/ffmpeg.exe first
                    candidate = os.path.join(pkg_dir, "bin", "ffmpeg.exe")
                    if os.path.isfile(candidate):
                        win_candidates.insert(0, candidate)
                        break
                    # Check versioned subfolder (e.g. ffmpeg-8.1.1-full_build\bin\ffmpeg.exe)
                    if os.path.isdir(pkg_dir):
                        for sub in os.listdir(pkg_dir):
                            sub_candidate = os.path.join(pkg_dir, sub, "bin", "ffmpeg.exe")
                            if os.path.isfile(sub_candidate):
                                win_candidates.insert(0, sub_candidate)
                                break
                        else:
                            continue
                        break
        for candidate in win_candidates:
            try:
                r = subprocess.run([candidate, "-version"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return "ffmpeg"  # fallback — let it fail with a clear error

    # Linux native
    for candidate in ["/usr/bin/ffmpeg", "ffmpeg"]:
        try:
            r = subprocess.run([candidate, "-version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "ffmpeg"

FFMPEG = _detect_ffmpeg()
# Derive ffprobe path — WinGet packages keep it alongside ffmpeg.exe
if os.name == "nt":
    _ffprobe_candidate = FFMPEG.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    FFPROBE = os.environ.get("FFPROBE_BIN", _ffprobe_candidate if os.path.isfile(_ffprobe_candidate) else "ffprobe")
else:
    FFPROBE = os.environ.get("FFPROBE_BIN", FFMPEG.replace("ffmpeg", "ffprobe"))
if FFPROBE == FFMPEG:
    FFPROBE = "ffprobe"

logger.info("Using FFMPEG: %s", FFMPEG)

# DNN face-detection model files (OpenCV's res10_300x300 SSD)
DNN_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CropRegion:
    """A single crop window (x, y, w, h) at a given timestamp."""
    t: float
    x: int
    y: int
    w: int
    h: int


@dataclass
class WordTiming:
    """A single word with its display window inside a clip."""
    word: str
    start: float   # seconds relative to clip start
    end: float     # seconds relative to clip start


@dataclass
class BrandTemplate:
    """Encapsulates visual branding for a clip."""
    name: str
    font_name: str
    primary_color: str       # FFmpeg colour string, e.g. "white"
    secondary_color: str     # e.g. "black@0.6"
    caption_style: str       # key into CAPTION_STYLE_PRESETS
    logo_path: Optional[str] = None
    logo_position: str = "top-right"   # top-left | top-right | bottom-left | bottom-right
    logo_scale: float = 0.08           # fraction of video width
    logo_opacity: float = 0.85

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "font_name": self.font_name,
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "caption_style": self.caption_style,
            "logo_path": self.logo_path,
            "logo_position": self.logo_position,
            "logo_scale": self.logo_scale,
            "logo_opacity": self.logo_opacity,
        }


# ---------------------------------------------------------------------------
# Caption-style presets  (used by build_animated_caption_filter)
# ---------------------------------------------------------------------------

CAPTION_STYLE_PRESETS: dict[str, dict] = {
    "default": {
        "fontsize_ratio": 25,       # width // ratio
        "fontcolor": "white",
        "box": True,
        "boxcolor": "black@0.6",
        "boxborderw_ratio": 100,    # width // ratio
        "y_position": 0.75,
        "highlight_color": None,
        "borderw": 0,
        "bordercolor": "black",
        "words_per_line": 4,
    },
    "big": {
        "fontsize_ratio": 18,
        "fontcolor": "white",
        "box": True,
        "boxcolor": "black@0.7",
        "boxborderw_ratio": 80,
        "y_position": 0.70,
        "highlight_color": None,
        "borderw": 0,
        "bordercolor": "black",
        "words_per_line": 3,
    },
    "outline": {
        "fontsize_ratio": 25,
        "fontcolor": "white",
        "box": False,
        "boxcolor": "black@0.0",
        "boxborderw_ratio": 100,
        "y_position": 0.75,
        "highlight_color": None,
        "borderw": 3,
        "bordercolor": "black",
        "words_per_line": 4,
    },
    "highlighted": {
        "fontsize_ratio": 24,
        "fontcolor": "white",
        "box": True,
        "boxcolor": "black@0.5",
        "boxborderw_ratio": 100,
        "y_position": 0.75,
        "highlight_color": "yellow",
        "borderw": 0,
        "bordercolor": "black",
        "words_per_line": 4,
    },
    "tiktok": {
        "fontsize_ratio": 22,
        "fontcolor": "white",
        "box": False,
        "boxcolor": "black@0.0",
        "boxborderw_ratio": 100,
        "y_position": 0.72,
        "highlight_color": "red",
        "borderw": 2,
        "bordercolor": "black",
        "words_per_line": 3,
    },
}


# ---------------------------------------------------------------------------
# Brand-template presets
# ---------------------------------------------------------------------------

BRAND_TEMPLATE_PRESETS: dict[str, BrandTemplate] = {
    "minimal": BrandTemplate(
        name="minimal",
        font_name="Arial",
        primary_color="white",
        secondary_color="black@0.5",
        caption_style="default",
    ),
    "bold": BrandTemplate(
        name="bold",
        font_name="Impact",
        primary_color="yellow",
        secondary_color="black@0.7",
        caption_style="big",
    ),
    "neon": BrandTemplate(
        name="neon",
        font_name="Arial",
        primary_color="cyan",
        secondary_color="magenta@0.4",
        caption_style="outline",
    ),
    "corporate": BrandTemplate(
        name="corporate",
        font_name="Helvetica",
        primary_color="white",
        secondary_color="blue@0.6",
        caption_style="default",
    ),
    "tiktok": BrandTemplate(
        name="tiktok",
        font_name="Arial-Bold",
        primary_color="white",
        secondary_color="black@0.0",
        caption_style="tiktok",
    ),
}


# ---------------------------------------------------------------------------
# Platform presets
# ---------------------------------------------------------------------------

PLATFORM_PRESETS: dict[str, dict] = {
    "tiktok": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "max_duration": 60,
        "video_bitrate": "6M",
        "audio_bitrate": "128k",
    },
    "youtube_short": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "max_duration": 60,
        "video_bitrate": "8M",
        "audio_bitrate": "128k",
    },
    "instagram_reel": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "max_duration": 90,
        "video_bitrate": "6M",
        "audio_bitrate": "128k",
    },
    "twitter": {
        "width": 720,
        "height": 1280,
        "fps": 30,
        "max_duration": 140,
        "video_bitrate": "3M",
        "audio_bitrate": "128k",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probe_video(filepath: str) -> dict:
    """Return ffprobe JSON info for the first video stream."""
    cmd = [
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[-400:]}")
    return json.loads(result.stdout)


def _has_audio(filepath: str) -> bool:
    """Check if a video file has an audio stream."""
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        stdout = result.stdout.strip()
        if result.returncode == 0:
            return "audio" in stdout
        # ffprobe failed — log and assume audio exists (safe default)
        logger.warning("_has_audio(%s): ffprobe rc=%d stderr='%s' — assuming audio exists",
                        os.path.basename(filepath), result.returncode, result.stderr.strip()[:100])
        return True
    except FileNotFoundError:
        logger.warning("_has_audio: ffprobe not found at '%s' — assuming audio exists", FFPROBE)
        return True
    except Exception as exc:
        logger.warning("_has_audio(%s) failed: %s — assuming audio exists", os.path.basename(filepath), exc)
        return True


def _detect_face_region(source_path: str, start: float, duration: float) -> Optional[dict]:
    """Detect the dominant face position in a video clip using OpenCV Haar cascade.

    Returns dict with face center (cx, cy) relative to the source video dimensions,
    or None if no face is detected.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for face detection")
        return None

    # Load Haar cascade — ships with OpenCV
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.isfile(cascade_path):
        logger.warning("Haar cascade not found at %s", cascade_path)
        return None

    cap = None
    try:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample 5 evenly-spaced frames from the clip
        clip_start_frame = int(start * fps)
        clip_end_frame = int((start + duration) * fps)
        clip_end_frame = min(clip_end_frame, total_frames)
        sample_count = min(5, max(1, (clip_end_frame - clip_start_frame) // int(fps)))
        step = max(1, (clip_end_frame - clip_start_frame) // sample_count)

        cascade = cv2.CascadeClassifier(cascade_path)
        faces_all = []
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for i in range(sample_count):
            frame_idx = clip_start_frame + i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                faces_all.append((x + w // 2, y + h // 2))  # center of each face

        if not faces_all:
            return None

        # Return average face center
        avg_x = sum(f[0] for f in faces_all) // len(faces_all)
        avg_y = sum(f[1] for f in faces_all) // len(faces_all)
        logger.info("Face detected: (%d,%d) from %d samples", avg_x, avg_y, len(faces_all))
        return {"cx": avg_x, "cy": avg_y, "vid_w": vid_w, "vid_h": vid_h}

    except Exception as exc:
        logger.warning("Face detection failed: %s", exc)
        return None
    finally:
        if cap is not None:
            cap.release()
    """Escape a string for FFmpeg drawtext filter.

    Single quotes break drawtext's text='...' syntax, so we strip them.
    Also escape backslashes, colons, and percent signs.
    """
    # Remove single quotes entirely (they break drawtext quoting)
    text = text.replace("'", "")
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    return text


def _smooth_track(regions: list[CropRegion], window: int = 5) -> list[CropRegion]:
    """Apply a simple moving-average smoother to a list of crop regions."""
    if len(regions) < window:
        return regions
    half = window // 2
    smoothed = []
    for i in range(len(regions)):
        lo = max(0, i - half)
        hi = min(len(regions), i + half + 1)
        avg_x = sum(r.x for r in regions[lo:hi]) / (hi - lo)
        avg_y = sum(r.y for r in regions[lo:hi]) / (hi - lo)
        smoothed.append(CropRegion(
            t=regions[i].t,
            x=int(round(avg_x)),
            y=int(round(avg_y)),
            w=regions[i].w,
            h=regions[i].h,
        ))
    return smoothed


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------


class FaceTracker:
    """
    Detects faces in video frames and produces smooth crop coordinates
    that keep the main speaker centred.

    Uses cv2.dnn with a lightweight Caffe SSD model when model files are
    available; falls back to cv2.CascadeClassifier (Haar) otherwise.
    """

    def __init__(
        self,
        dnn_proto: Optional[str] = None,
        dnn_model: Optional[str] = None,
        confidence_threshold: float = 0.5,
        sample_interval: float = 0.5,   # seconds between samples
        smooth_window: int = 7,
    ):
        self.confidence_threshold = confidence_threshold
        self.sample_interval = sample_interval
        self.smooth_window = smooth_window
        self._net = None
        self._cascade = None
        self._use_dnn = False

        # Try DNN first
        if dnn_proto and dnn_model and os.path.isfile(dnn_proto) and os.path.isfile(dnn_model):
            try:
                self._net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
                self._use_dnn = True
                logger.info("FaceTracker: using DNN face detector")
                return
            except Exception as exc:
                logger.warning("FaceTracker: DNN load failed (%s), falling back to Haar", exc)

        # Fallback: Haar cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.isfile(cascade_path):
            self._cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("FaceTracker: using Haar cascade face detector")
        else:
            logger.warning("FaceTracker: no face detector available")

    # -- detection helpers ---------------------------------------------------

    def _detect_faces_dnn(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), False, False)
        self._net.setInput(blob)
        detections = self._net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < self.confidence_threshold:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def _detect_faces_haar(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return self._cascade.detectMultiScale(gray, 1.3, 5)

    def _detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._use_dnn and self._net is not None:
            return self._detect_faces_dnn(frame)
        if self._cascade is not None:
            return self._detect_faces_haar(frame)
        return []

    @staticmethod
    def _pick_main_face(
        faces: list[tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int,
    ) -> Optional[tuple[int, int, int, int]]:
        """Pick the largest face closest to centre-of-frame."""
        if not faces:
            return None
        cx, cy = frame_w / 2, frame_h / 2

        def score(f):
            fx, fy, fw, fh = f
            area = fw * fh
            dist = ((fx + fw / 2 - cx) ** 2 + (fy + fh / 2 - cy) ** 2) ** 0.5
            return area - dist * 20   # prefer large & centred

        return max(faces, key=score)

    # -- public API ----------------------------------------------------------

    def track(
        self,
        filepath: str,
        start: float,
        end: float,
        target_w: int,
        target_h: int,
    ) -> list[CropRegion]:
        """
        Sample frames between *start* and *end*, detect faces, and return
        a list of :class:`CropRegion` objects that keep the main face centred.
        """
        if self._net is None and self._cascade is None:
            logger.warning("FaceTracker: no detector — returning centre crop")
            info = _probe_video(filepath)
            src_w = int(info["streams"][0]["width"])
            src_h = int(info["streams"][0]["height"])
            return [CropRegion(t=start, x=(src_w - target_w) // 2,
                               y=(src_h - target_h) // 2, w=target_w, h=target_h)]

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {filepath}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = end - start
        num_samples = max(1, int(duration / self.sample_interval))
        sample_times = [start + i * duration / max(num_samples, 1)
                        for i in range(num_samples + 1)]

        raw_regions: list[CropRegion] = []

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            faces = self._detect_faces(frame)
            main = self._pick_main_face(faces, src_w, src_h)

            if main is not None:
                fx, fy, fw, fh = main
                face_cx = fx + fw // 2
                face_cy = fy + fh // 2
                # Place crop so face is centred (with slight upward bias)
                cx = face_cx - target_w // 2
                cy = face_cy - int(target_h * 0.42)  # face at 42% from top
            else:
                cx = (src_w - target_w) // 2
                cy = (src_h - target_h) // 2

            # Clamp
            cx = max(0, min(cx, src_w - target_w))
            cy = max(0, min(cy, src_h - target_h))

            raw_regions.append(CropRegion(t=t, x=cx, y=cy, w=target_w, h=target_h))

        cap.release()

        if not raw_regions:
            return [CropRegion(t=start, x=(src_w - target_w) // 2,
                               y=(src_h - target_h) // 2, w=target_w, h=target_h)]

        return _smooth_track(raw_regions, window=self.smooth_window)

    def build_crop_filter(
        self,
        regions: list[CropRegion],
        src_w: int,
        src_h: int,
        target_w: int,
        target_h: int,
        fps: int,
    ) -> str:
        """
        Build an FFmpeg ``crop`` + ``scale`` filter string that follows the
        tracked face regions using ``enable`` expressions per segment.
        """
        if not regions:
            return f"scale={target_w}:{target_h}:flags=lanczos"

        # Build a piecewise-linear crop-x / crop-y using geq or sendcmd.
        # For broad compatibility we use the ``crop`` filter with
        # ``x``/``y`` expressions driven by ``t`` via ``lerp`` between keyframes.
        #
        # FFmpeg's drawtext/crop x/y can use generic expressions with
        # ``between()`` and linear interpolation.  We emit a chain of
        # ``crop`` filters, each enabled for a time range.

        parts: list[str] = []
        for i, reg in enumerate(regions):
            t0 = reg.t
            t1 = regions[i + 1].t if i + 1 < len(regions) else t0 + 2.0
            parts.append(
                f"crop={reg.w}:{reg.h}:{reg.x}:{reg.y}:"
                f"enable='between(t\\,{t0:.3f}\\,{t1:.3f})'"
            )

        # If there is only one region, just crop unconditionally
        if len(parts) == 1:
            parts = [f"crop={regions[0].w}:{regions[0].h}:{regions[0].x}:{regions[0].y}"]

        # After cropping, scale to target
        scale_part = f"scale={target_w}:{target_h}:flags=lanczos"

        if len(parts) == 1:
            return f"{parts[0]},{scale_part}"

        # Chain: first crop segment, then scale; subsequent segments
        # use sendcmd or overlay approach.  For simplicity and robustness
        # we use a single crop with expression-driven x/y.
        #
        # Build lerp expressions for x(t) and y(t):
        x_expr = self._build_lerp_expr(regions, "x", src_w - target_w)
        y_expr = self._build_lerp_expr(regions, "y", src_h - target_h)
        return f"crop={target_w}:{target_h}:x='{x_expr}':y='{y_expr}',{scale_part}"

    @staticmethod
    def _build_lerp_expr(
        regions: list[CropRegion],
        attr: str,
        max_val: int,
    ) -> str:
        """
        Build an FFmpeg expression that linearly interpolates *attr*
        between keyframe timestamps.
        """
        if len(regions) == 1:
            return str(getattr(regions[0], attr))

        # Use nested if / between expressions
        # FFmpeg expr: if(between(t,t0,t1), lerp(v0,v0,v1, t0,t1), next)
        expr = str(max_val // 2)   # fallback
        for i in range(len(regions) - 1, 0, -1):
            r0 = regions[i - 1]
            r1 = regions[i]
            v0 = getattr(r0, attr)
            v1 = getattr(r1, attr)
            t0 = r0.t
            t1 = r1.t
            # Clamp so we don't extrapolate wildly
            expr = (
                f"if(between(t\\,{t0:.3f}\\,{t1:.3f}),"
                f"{v0}+({v1}-{v0})*(t-{t0:.3f})/max({t1-t0:.3f}\\,0.001),"
                f"{expr})"
            )
        # Wrap with first segment value before first keyframe
        r0 = regions[0]
        expr = f"if(lt(t\\,{r0.t:.3f}),{getattr(r0, attr)},{expr})"
        return expr


# ---------------------------------------------------------------------------
# Module-level face-tracker singleton (lazy)
# ---------------------------------------------------------------------------

_face_tracker: Optional[FaceTracker] = None


def _get_face_tracker() -> FaceTracker:
    global _face_tracker
    if _face_tracker is None:
        _face_tracker = FaceTracker()
    return _face_tracker


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def get_brand_template(name: str) -> dict:
    """
    Return a brand-template dictionary by preset name.

    Raises ValueError if *name* is unknown.
    """
    key = name.lower().strip()
    if key not in BRAND_TEMPLATE_PRESETS:
        raise ValueError(
            f"Unknown brand template '{name}'. "
            f"Available: {list(BRAND_TEMPLATE_PRESETS.keys())}"
        )
    return BRAND_TEMPLATE_PRESETS[key].to_dict()


def build_caption_words(
    transcriptions: list[dict],
    start: float,
    end: float,
) -> list[dict]:
    """
    Extract word-level timestamps for the clip range [*start*, *end*].

    Each returned dict has keys: ``word``, ``start``, ``end`` (relative to
    clip start).
    """
    words: list[dict] = []
    for seg in transcriptions:
        for w in seg.get("words", []):
            ws = w.get("start", 0.0)
            we = w.get("end", 0.0)
            if ws >= start and we <= end:
                words.append({
                    "word": w["word"],
                    "start": round(ws - start, 4),
                    "end": round(we - start, 4),
                })
    # Sort by start time
    words.sort(key=lambda x: x["start"])
    return words


def build_animated_caption_filter(
    words: list[dict],
    width: int,
    height: int,
    style: str = "default",
    brand: Optional[BrandTemplate] = None,
) -> str:
    """
    Build a simple FFmpeg caption filter — one drawtext per line.
    Keeps it lightweight to avoid FFmpeg crashes on Windows.
    """
    if not words:
        return ""

    preset = CAPTION_STYLE_PRESETS.get(style, CAPTION_STYLE_PRESETS["default"])
    fontcolor = brand.primary_color if brand else preset["fontcolor"]
    boxcolor = brand.secondary_color if brand else preset["boxcolor"]
    use_box = preset.get("box", True)
    borderw = preset.get("borderw", 0)
    bordercolor = preset.get("bordercolor", "black")
    words_per_line = preset.get("words_per_line", 4)

    # Use local fonts/ folder (relative path, safe for FFmpeg on Windows)
    font_name = (brand.font_name if brand else None) or "Arial"
    font_file_map = {
        "arial": "arial.ttf",
        "calibri": "calibri.ttf",
        "verdana": "verdana.ttf",
    }
    actual_font = font_file_map.get(font_name.lower().replace(" ", ""), "arial.ttf")
    font_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fonts", actual_font
    )
    # FFmpeg on Windows needs forward slashes in fontfile paths
    # Backslashes get parsed as escape characters by the filter graph
    font_path = font_path.replace("\\", "/")
    if not os.path.isfile(font_path):
        logger.warning("Font not found: %s, skipping captions", font_path)
        return ""
    fontsize = width // preset["fontsize_ratio"]
    boxborderw = width // preset["boxborderw_ratio"] if use_box else 0
    line_height = int(fontsize * 1.4)
    base_y = int(height * preset["y_position"])

    # Group words into lines
    lines: list[list[dict]] = []
    buf: list[dict] = []
    for w in words:
        buf.append(w)
        if len(buf) >= words_per_line:
            lines.append(buf)
            buf = []
    if buf:
        lines.append(buf)

    # Limit to max 5 drawtext nodes to avoid FFmpeg filter chain limits
    MAX_DRAWTEXT = 5
    if len(lines) > MAX_DRAWTEXT:
        logger.info("Too many caption lines (%d), limiting to %d", len(lines), MAX_DRAWTEXT)
        lines = lines[:MAX_DRAWTEXT]

    filters: list[str] = []

    for line_idx, line in enumerate(lines):
        line_start = round(line[0]["start"], 3)
        line_end = round(line[-1]["end"], 3)
        # Skip zero-duration lines (FFmpeg rejects enable=t1..t1 when t1==t2)
        if line_end <= line_start:
            line_end = line_start + 0.5
        # Position from bottom: stack lines upward so they stay on screen
        # Each line gets a slot (0, 1, 2) based on its index
        slot = line_idx % 3
        y = int(height * 0.88) - slot * line_height
        y = max(fontsize + 10, y)  # keep on screen
        line_text = " ".join(_escape_drawtext(w["word"]) for w in line)

        # Single drawtext per line — simple and stable
        dt = (
            f"drawtext=text='{line_text}':"
            f"fontsize={fontsize}:"
            f"fontcolor={fontcolor}:"
            f"x=(w-text_w)/2:y={y}:"
        )
        if use_box:
            dt += f"box=1:boxcolor={boxcolor}:boxborderw={boxborderw}:"
        dt += f"borderw={borderw}:bordercolor={bordercolor}:"
        if font_path and os.path.isfile(font_path):
            # Use forward slashes — works fine for local relative paths
            dt += f"fontfile='{font_path}':"
        dt += f"enable='between(t\\,{line_start}\\,{line_end})'"
        filters.append(dt)

    return ",".join(filters)


def get_face_crop_coordinates(
    filepath: str,
    start: float,
    end: float,
    target_w: int,
    target_h: int,
) -> list[dict]:
    """
    Analyse *filepath* between *start* and *end* and return a list of
    crop-coordinate dicts (keys: t, x, y, w, h) that track the main face.

    This is a convenience wrapper around :class:`FaceTracker`.
    """
    tracker = _get_face_tracker()
    regions = tracker.track(filepath, start, end, target_w, target_h)
    return [{"t": r.t, "x": r.x, "y": r.y, "w": r.w, "h": r.h} for r in regions]


def _get_pil_font(font_size: int):
    """Try to load a PIL font, fallback to default.

    Uses local fonts/ folder (copied from Windows) to avoid font path
    parsing issues with FFmpeg on Windows.
    """
    try:
        from PIL import ImageFont
        # Use local fonts folder first (avoids Windows Fonts path issues)
        local_font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
        font_paths = [
            os.path.join(local_font_dir, "arial.ttf"),
            os.path.join(local_font_dir, "verdana.ttf"),
            os.path.join(local_font_dir, "calibri.ttf"),
            # Fallback to absolute Windows paths
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\verdana.ttf",
        ]
        for fp in font_paths:
            if os.path.isfile(fp):
                return ImageFont.truetype(fp, font_size)
        return ImageFont.load_default()
    except ImportError:
        return None


def _parse_box_color(boxcolor: str):
    """Parse 'black@0.6' → ((0,0,0), 153)."""
    import re
    m = re.match(r"(\w+)@([\d.]+)", boxcolor)
    if m:
        color_name = m.group(1).lower()
        alpha = int(float(m.group(2)) * 255)
    else:
        color_name = "black"
        alpha = 153
    color_map = {
        "black": (0, 0, 0), "white": (255, 255, 255),
        "red": (0, 0, 255), "blue": (255, 0, 0),
        "green": (0, 255, 0), "yellow": (0, 255, 255),
    }
    return color_map.get(color_name, (0, 0, 0)), alpha


def _draw_caption_line(
    frame, text: str, y: int, font, fontcolor: str,
    use_box: bool, bc_rgb: tuple, bc_alpha: int, width: int
):
    """Draw a single caption line onto a frame using PIL."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw

        if font is None:
            return frame

        # Convert OpenCV BGR frame to PIL RGB
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Measure text
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width - tw) // 2

        # Draw box background
        if use_box:
            pad = max(8, width // 100)
            overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [x - pad, y - pad, x + tw + pad, y + th + pad],
                fill=(*bc_rgb[::-1], bc_alpha),  # BGR→RGBA
            )
            pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(pil_img)

        # Map fontcolor name to RGB
        color_map = {
            "white": (255, 255, 255), "black": (0, 0, 0),
            "yellow": (255, 255, 0), "red": (0, 0, 255),
        }
        fc_rgb = color_map.get(fontcolor, (255, 255, 255))
        draw.text((x, y), text, fill=fc_rgb, font=font)

        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    except Exception as exc:
        logger.warning("Caption draw failed: %s", exc)
        return frame


def _burn_captions_opencv(
    video_path: str,
    output_path: str,
    captions: list[dict],
    width: int,
    height: int,
    caption_style: str = "default",
    progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """Burn captions onto a video using OpenCV + PIL for text rendering.

    This avoids FFmpeg's broken drawtext fontfile path parsing on Windows.
    Reads each frame, draws active captions with PIL (better font support),
    then writes to the output file with OpenCV's VideoWriter.
    """
    preset = CAPTION_STYLE_PRESETS.get(caption_style, CAPTION_STYLE_PRESETS["default"])
    fontcolor = preset.get("fontcolor", "white")
    boxcolor = preset.get("boxcolor", "black@0.6")
    use_box = preset.get("box", True)
    y_position = preset.get("y_position", 0.75)
    words_per_line = preset.get("words_per_line", 4)

    # Resolve font
    font_size = width // preset.get("fontsize_ratio", 25)
    font = _get_pil_font(font_size)

    # Group captions into lines with timing
    lines = []
    buf = []
    for w in captions or []:
        buf.append(w)
        if len(buf) >= words_per_line:
            lines.append(buf)
            buf = []
    if buf:
        lines.append(buf)

    # Build line data: (start_time, end_time, text, slot_index)
    # We have 3 visible slots. Each line gets a slot based on its index mod 3.
    # Lines are drawn bottom-up: slot 0 = bottom, slot 2 = top
    line_height = int(font_size * 1.4)
    visible_slots = 3
    bottom_y = int(height * 0.88)  # bottom-most line at 88% of screen height
    line_data = []
    for idx, line in enumerate(lines):
        t_start = line[0]["start"]
        t_end = line[-1]["end"]
        text = " ".join(w["word"] for w in line)
        slot = idx % visible_slots
        y = bottom_y - slot * line_height
        y = max(font_size + 10, y)  # keep on screen
        line_data.append((t_start, t_end, text, y))

    # Open video with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for caption burning: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Use raw AVI intermediate (uncompressed) then FFmpeg to encode properly
    # mp4v codec on Windows is unreliable and produces corrupted files
    tmp_raw = output_path + ".tmp.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Motion-JPEG, reliable on Windows
    out = cv2.VideoWriter(tmp_raw, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for: {output_path}")

    # Parse box color
    bc_rgb, bc_alpha = _parse_box_color(boxcolor)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            # Draw all active caption lines on this frame
            for (t_start, t_end, text, y) in line_data:
                if t_start <= current_time <= t_end:
                    frame = _draw_caption_line(
                        frame, text, y, font, fontcolor,
                        use_box, bc_rgb, bc_alpha, width
                    )

            out.write(frame)
            frame_idx += 1

            if progress_callback and frame_idx % 30 == 0:
                pct = min(100.0, frame_idx / total_frames * 100) if total_frames > 0 else 0
                progress_callback(round(pct, 1))

    finally:
        cap.release()
        out.release()

    # Re-encode AVI → MP4 with FFmpeg (proper H.264 encoding)
    if frame_idx > 0 and os.path.isfile(tmp_raw):
        try:
            ffmpeg_cmd = [
                FFMPEG, "-y",
                "-i", tmp_raw,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ]
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.warning("FFmpeg re-encode failed: %s", result.stderr[-500:])
        except Exception as e:
            logger.warning("FFmpeg re-encode error: %s", e)
        finally:
            # Clean up temp AVI
            try:
                os.remove(tmp_raw)
            except OSError:
                pass
    else:
        logger.warning("No frames written, skipping re-encode")

    logger.info("captions burned: %s (%d frames)", output_path, frame_idx)
    return output_path



def cut_clip(
    source_path: str,
    job_id: str,
    clip_id: int,
    start: float,
    end: float,
    platform: str = "tiktok",
    captions: Optional[list[dict]] = None,
    caption_style: str = "default",
    brand_template: Optional[str] = None,
    face_track: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
    crop_mode: str = "blur_bg",
) -> str:
    """
    Extract a clip from *source_path*, reformat for *platform*, optionally
    add animated captions and face-aware cropping.

    Parameters
    ----------
    source_path : str
        Path to the source video file.
    job_id : str
        Identifier for this processing job (used for output directory).
    clip_id : int
        Sequential clip number (used in the output filename).
    start, end : float
        Clip boundaries in seconds.
    platform : str
        One of ``tiktok``, ``youtube_short``, ``instagram_reel``, ``twitter``.
    captions : list[dict] | None
        Word-level timing data (as returned by :func:`build_caption_words`).
    caption_style : str
        Caption visual style (see ``CAPTION_STYLE_PRESETS``).
    brand_template : str | None
        Name of a brand template preset (see ``BRAND_TEMPLATE_PRESETS``).
    face_track : bool
        If True, use face-aware cropping instead of centre crop.
    crop_mode : str
        How to handle aspect ratio conversion to 9:16:
        - "blur_bg" (default) — blurred background fill, no black bars, no lost content
        - "center_crop" — classic center crop (may cut off edges)
        - "face_track" — detect faces and keep them centered
    progress_callback : callable | None
        Called with a float 0-100 indicating progress.

    Returns
    -------
    str
        Absolute path to the rendered clip.
    """
    # --- Validate platform ------------------------------------------------
    if platform not in PLATFORM_PRESETS:
        raise ValueError(
            f"Unknown platform '{platform}'. "
            f"Choose from: {list(PLATFORM_PRESETS.keys())}"
        )

    preset = PLATFORM_PRESETS[platform]
    w, h = preset["width"], preset["height"]
    # Ensure even dimensions for H.264 compatibility
    w = w - (w % 2)
    h = h - (h % 2)
    fps = preset["fps"]
    duration = end - start

    # Enforce max duration
    if duration > preset["max_duration"]:
        end = start + preset["max_duration"]
        duration = preset["max_duration"]
        logger.info("Clip truncated to %ss (platform limit)", preset["max_duration"])

    if duration < 3:
        raise ValueError("Clip too short (minimum 3 seconds)")

    # --- Brand template ---------------------------------------------------
    brand = None
    if brand_template:
        if isinstance(brand_template, dict):
            bt_dict = brand_template
        else:
            bt_dict = get_brand_template(brand_template)
        brand = BrandTemplate(**bt_dict)
        if not caption_style or caption_style == "default":
            caption_style = bt_dict.get("caption_style", caption_style)

    # --- Output path ------------------------------------------------------
    clip_dir = os.path.join(CLIPS_DIR, job_id)
    os.makedirs(clip_dir, exist_ok=True)
    output_filename = f"clip_{clip_id:02d}_{platform}.mp4"
    output_path = os.path.join(clip_dir, output_filename)

    # --- Build video filter chain -----------------------------------------
    # Crop mode: "blur_bg" (default) | "center_crop" | "face_track"

    # Read source aspect ratio to decide scaling strategy
    src_info = _probe_video(source_path)
    src_w = int(src_info["streams"][0].get("width", 1920))
    src_h = int(src_info["streams"][0].get("height", 1080))
    src_aspect = src_w / src_h
    dst_aspect = w / h

    vf_string = None  # set by each branch below
    is_complex = False

    if crop_mode == "blur_bg":
        # Blurred background: full video visible, edges filled with blurred copy
        bg_scale = f"scale={w}*1.15:{h}*1.15:flags=lanczos"
        bg_blur = "boxblur=20:20"
        bg_dark = "eq=brightness=-0.12:contrast=1.05"
        if src_aspect > dst_aspect:
            fg_scale = f"scale={w}*{min(1, dst_aspect/src_aspect):.2f}:{h}:flags=lanczos"
        else:
            fg_scale = f"scale={w}:{h}*{min(1, src_aspect/dst_aspect):.2f}:flags=lanczos"
        vf_string = (
            f"[0:v]split=2[bg_src][fg_src];"
            f"[bg_src]{bg_scale},{bg_blur},{bg_dark}[bg];"
            f"[fg_src]{fg_scale}[fg];"
            f"[bg][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:format=auto,"
            f"fps={fps},format=yuv420p"
        )
        is_complex = True

    elif crop_mode == "face_track":
        face_data = _detect_face_region(source_path, start, duration)
        if face_data:
            face_cx = face_data["cx"]
            scale_f = f"scale=-2:{h}:flags=lanczos"
            x_off = max(0, min(face_cx - w // 2, src_w - w))
            crop_f = f"crop={w}:{h}:{x_off}:0"
            vf_string = f"{scale_f},{crop_f},fps={fps},format=yuv420p"
            logger.info("Face track: face at (%d), crop x_offset=%d", face_cx, x_off)
        else:
            logger.info("Face track: no face detected, falling back to blur_bg")
            crop_mode = "blur_bg"  # fall through to blur_bg below
            # re-use blur_bg code by setting is_complex after this block

    if crop_mode in ("center_crop",) or (crop_mode == "face_track" and vf_string is None):
        # Classic center crop (original behavior) OR face_track fallback
        scale_f = f"scale=-2:{h}:flags=lanczos"
        crop_f = f"crop={w}:{h}:(in_w-out_w)/2:0"
        vf_string = f"{scale_f},{crop_f},fps={fps},format=yuv420p"

    # Safety: if somehow vf_string wasn't set, fall back to center crop
    if vf_string is None:
        scale_f = f"scale=-2:{h}:flags=lanczos"
        crop_f = f"crop={w}:{h}:(in_w-out_w)/2:0"
        vf_string = f"{scale_f},{crop_f},fps={fps},format=yuv420p"

    # --- Captions (optional) -----------------------------------------------
    # Use ASS subtitles instead of drawtext — far more reliable on Windows.
    # No filter chain length limits, proper font rendering via libass.
    ass_file = None
    if captions:
        try:
            from core.ass_subtitles import generate_ass_subtitles
            ass_file = generate_ass_subtitles(
                words=captions,
                width=w,
                height=h,
                template_name=caption_style,
            )
            logger.info("ASS subtitles generated: %s", ass_file)
        except Exception as exc:
            logger.warning("ASS subtitle generation failed (%s), skipping captions", exc)

    # Add ASS subtitle burning to filter chain
    if ass_file and os.path.isfile(ass_file):
        ass_path = ass_file.replace("\\", "/")
        ass_path_escaped = ass_path.replace(":", "\\:")
        fonts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
        fonts_dir_escaped = fonts_dir.replace("\\", "/").replace(":", "\\:")
        subtitle_filter = f"subtitles=filename='{ass_path_escaped}':fontsdir='{fonts_dir_escaped}'"
        if is_complex:
            # Insert subtitle filter before the fps/format ending
            vf_string = vf_string.replace(f",fps={fps},format=yuv420p", f",{subtitle_filter},fps={fps},format=yuv420p")
        else:
            vf_string = vf_string + "," + subtitle_filter
        logger.info("ASS subtitles added to filter chain")

    logger.info("cut_clip vf: %s", vf_string[:200])

    # --- Build FFmpeg command ---------------------------------------------
    # Use a clean environment to avoid fontconfig issues on Windows
    ffmpeg_env = os.environ.copy()
    ffmpeg_env.pop("FONTCONFIG_FILE", None)
    ffmpeg_env.pop("FONTCONFIG_PATH", None)
    ffmpeg_env.pop("FONTCONFIG_SYSROOT", None)

    # Write filter chain to temp file if it's too long for command line
    # FFmpeg has a limit on -vf argument length (~8KB on Windows)
    vf_file = None
    if len(vf_string) > 4000:
        import tempfile
        vf_file = os.path.join(tempfile.gettempdir(), f"vf_{job_id}_{clip_id}.txt")
        with open(vf_file, "w", encoding="utf-8") as f:
            f.write(vf_string)
        logger.info("Filter chain too long (%d chars), using temp file: %s", len(vf_string), vf_file)

    # Check if source has audio
    source_has_audio = _has_audio(source_path)

    # Determine if this is a complex filter (uses labelled pins like [bg],[fg], overlay, split, etc.)
    # Complex filters require -filter_complex instead of -vf
    is_complex_filter = "[bg]" in vf_string or "[fg]" in vf_string or "overlay=" in vf_string or "split=" in vf_string

    cmd = [
        FFMPEG, "-y",
        "-ss", str(start),
        "-i", source_path,
        "-t", str(duration),
    ]
    if is_complex_filter:
        if vf_file:
            cmd += ["-filter_complex_script", vf_file]
        else:
            cmd += ["-filter_complex", vf_string]
    elif vf_file:
        cmd += ["-filter_complex_script", vf_file]
    else:
        cmd += ["-vf", vf_string]
    cmd += [
        "-c:v", "libx264",
        "-preset", "fast",
        "-b:v", preset["video_bitrate"],
    ]
    # Only add audio encoding if source has audio
    if source_has_audio:
        cmd += ["-c:a", "aac", "-b:a", preset["audio_bitrate"]]
    else:
        cmd += ["-an"]
        logger.info("Source has no audio stream, skipping audio encoding")
    cmd += [
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-threads", "0",
        output_path,
    ]

    logger.info("cut_clip: %s", " ".join(cmd[:10]) + " ...")

    # --- Run FFmpeg --------------------------------------------------------
    if progress_callback:
        progress_callback(0.0)

    logger.info("cut_clip: source=%s output=%s ffmpeg=%s", source_path, output_path, FFMPEG)
    logger.info("cut_clip: source exists=%s dir exists=%s", os.path.exists(source_path), os.path.isdir(clip_dir))

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=ffmpeg_env,
        )
        total_us = int(duration * 1_000_000)

        # Use communicate() to avoid deadlocks from simultaneous stdout/stderr reads
        try:
            stdout_data, stderr_data = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"FFmpeg cut timed out after 600s")

        # Log full stderr for debugging
        if stderr_data:
            logger.debug("FFmpeg stderr:\n%s", stderr_data[-2000:])

        # Parse progress from stderr (best-effort)
        if progress_callback and stderr_data:
            for line in stderr_data.splitlines():
                line = line.strip()
                if line.startswith("out_time_us="):
                    try:
                        cur_us = int(line.split("=", 1)[1])
                        pct = min(100.0, cur_us / total_us * 100)
                        progress_callback(round(pct, 1))
                    except (ValueError, ZeroDivisionError):
                        pass

        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg cut failed (rc={proc.returncode}) for {output_path}:\n{stderr_data[-3000:] if stderr_data else 'no stderr'}"
            )

    except RuntimeError:
        raise  # re-raise our own errors
    except Exception as exc:
        raise RuntimeError(f"FFmpeg error: {exc}")
    finally:
        # Clean up temp files
        if vf_file and os.path.isfile(vf_file):
            try:
                os.remove(vf_file)
            except OSError:
                pass
        if ass_file and os.path.isfile(ass_file):
            try:
                os.remove(ass_file)
            except OSError:
                pass

    if progress_callback:
        progress_callback(100.0)

    logger.info("cut_clip: output saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_font_path(font_name: str) -> Optional[str]:
    """
    Try to resolve *font_name* to an absolute file path suitable for
    FFmpeg's ``fontfile`` parameter.  Returns None if not found.
    """
    # Common font directories
    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        "/mnt/c/Windows/Fonts",
    ]

    # Normalise name
    name_lower = font_name.lower().replace("-", "").replace(" ", "")

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                fl = f.lower().replace("-", "").replace(" ", "")
                if fl.startswith(name_lower) and f.endswith((".ttf", ".otf")):
                    return os.path.join(root, f)
    return None


def _escape_drawtext(text: str) -> str:
    """Escape a string for FFmpeg drawtext filter.

    Single quotes break drawtext's text='...' syntax, so we strip them.
    Also escape backslashes and colons.
    """
    return text.replace("\\", "\\\\").replace("'", "").replace(":", "\\:").replace("%", "%%")


def _build_logo_filter(brand: BrandTemplate, vid_w: int, vid_h: int) -> Optional[str]:
    """Build an overlay filter string for a brand logo watermark."""
    if not brand.logo_path or not os.path.isfile(brand.logo_path):
        return None

    logo_w = int(vid_w * brand.logo_scale)
    pad = int(vid_w * 0.02)

    # Position
    pos = brand.logo_position
    if "right" in pos:
        x_expr = f"main_w-overlay_w-{pad}"
    else:
        x_expr = str(pad)

    if "bottom" in pos:
        y_expr = f"main_h-overlay_h-{pad}"
    else:
        y_expr = str(pad)

    return (
        f"[v]overlay={x_expr}:{y_expr}:format=auto:alpha={brand.logo_opacity}"
    )


def hex_to_ass_color(name: str) -> str:
    """
    Convert a simple colour name to an ASS hex string (BBGGRR).
    """
    colours = {
        "white": "FFFFFF",
        "black": "000000",
        "red": "0000FF",
        "green": "00FF00",
        "blue": "FF0000",
        "yellow": "00FFFF",
        "cyan": "FFFF00",
        "magenta": "FF00FF",
        "orange": "0080FF",
    }
    return colours.get(name.lower(), "FFFFFF")
