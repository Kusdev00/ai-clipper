"""
Analyzer - OpusClip-grade AI analysis engine for viral clip detection.

Provides:
    - Audio transcription (faster-whisper)
    - Audio analysis (librosa: energy, excitement, laughter, shouting)
    - Scene detection (FFmpeg scene change filter)
    - Speaker change detection (energy + pitch heuristics)
    - Hook detection (attention-grabbing phrase scoring)
    - Topic change detection (keyword clustering)
    - Viral score calculation (weighted multi-signal)
    - Highlight clip detection (main orchestrator)
    - Chapter / topic detection (Ollama LLM or heuristic fallback)
    - Hot take scoring (Ollama LLM or heuristic fallback)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from core.config import DOWNLOADS_DIR

log = logging.getLogger(__name__)

# ── Weights for viral score components ──────────────────────────────────────

WEIGHT_EXCITEMENT_WORDS = 0.2
WEIGHT_AUDIO_ENERGY = 0.2
WEIGHT_SPEECH_DENSITY = 0.15
WEIGHT_HOOK_QUALITY = 0.15
WEIGHT_SCENE_DYNAMICS = 0.1
WEIGHT_CAPS_SHOUTING = 0.1
WEIGHT_QUESTIONS = 0.1

# ── Pattern definitions ─────────────────────────────────────────────────────

EXCITEMENT_WORDS: dict[str, float] = {
    "worst": 1.0, "first": 0.5, "last": 0.8, "only": 0.6,
    "biggest": 0.9, "smallest": 0.7, "big": 0.4, "tiny": 0.3,
    "diff": 0.2,
}

HOOK_PHRASES: list[str] = [
    "watch this", "you won't believe", "you will not believe",
    "wait for it", "this is crazy", "this is insane", "listen up",
    "pay attention", "here's the thing", "the thing is",
    "here's what", "let me tell you", "i promise you", "trust me",
    "believe me", "you need to", "you have to", "you must",
    "don't miss", "look at this", "check this out",
    "this changed everything", "nobody talks about", "the secret",
    "the truth about", "what happens next", "the reason why",
    "here's why", "this is why", "stop scrolling", "wait until",
    "just wait", "it gets better", "but here's", "plot twist",
]

QUESTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(what|why|how|when|where|who|which|can you|could you|would you|do you|are you|is this|will this)\b.*\?",
        r"\b(ever wonder|have you ever|did you know|what if|imagine if)\b",
    ]
]

LAUGHTER_WORDS = ["haha", "lol", "lmao", "rofl", "hahaha", "laughing"]
SHOUTING_PATTERNS = [re.compile(r"\b[A-Z]{3,}\b")]

TOPIC_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "and", "but", "or", "not", "no", "yes", "so", "if", "then",
    "that", "this", "it", "its", "i", "me", "my", "we", "us", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "about", "up", "out",
    "yeah", "yes", "okay", "ok", "like", "gonna", "wanna", "gotta",
    "uh", "um", "hmm", "oh", "ah", "well", "really", "actually", "basically",
    "know", "get", "got", "go", "going", "come", "came", "make", "made",
    "think", "thought", "see", "saw", "look", "looked", "want", "say", "said",
}


# ── Utility Functions ───────────────────────────────────────────────────────

def _job_dir(job_id: str) -> Path:
    """Return the download directory path for a given job."""
    return Path(os.path.join(DOWNLOADS_DIR, job_id))


def _audio_path(job_id: str) -> str:
    """Return the WAV audio path for a given job."""
    return os.path.join(_job_dir(job_id), "audio.wav")


def _video_path(job_id: str) -> str:
    """Return the video file path for a given job (best-effort)."""
    d = job_id
    video_ext = (".mp4", ".mkv", ".webm", ".avi", ".mov")
    for p in video_ext:
        f = os.path.join(d, f"video{p}")
        if os.path.exists(f):
            return f
    # Fallback: search directory
    jdir = _job_dir(job_id)
    if os.path.isdir(jdir):
        for f in os.listdir(jdir):
            ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
            if ext in ("mp4", "mkv", "webm", "avi", "mov"):
                return os.path.join(str(jdir), f)
    return ""


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a subprocess command and return the result."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ── Audio Duration ──────────────────────────────────────────────────────────

def get_duration(filepath: str) -> float:
    """Get video duration in seconds using ffprobe.

    Args:
        filepath: Path to the video file.

    Returns:
        Duration in seconds as a float.

    Raises:
        Exception: If ffprobe fails.
    """
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath]
    result = _run(cmd, timeout=30)
    if result.returncode != 0:
        raise Exception(f"ffprobe failed: {result.stderr}")
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


# ── Transcription ───────────────────────────────────────────────────────────

def transcribe_audio(job_id: str, model_size: str = "base") -> list[dict]:
    """Transcribe audio using faster-whisper with word-level timestamps.

    Args:
        job_id: Job identifier (used to locate audio.wav in downloads dir).
        model_size: Whisper model size (tiny, base, small, medium, large-v3).

    Returns:
        List of segment dicts with keys: start, end, text, words.
        Each word dict has: word, start, end, confidence.

    Raises:
        Exception: If audio file not found or faster-whisper not installed.
    """
    ap = _audio_path(job_id)
    if not os.path.exists(ap):
        raise Exception("Audio file not found. Download audio first.")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise Exception("faster-whisper not installed. Run: pip install faster-whisper")

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(ap, word_timestamps=True, beam_size=5)

    results: list[dict] = []
    for segment in segments:
        words: list[dict] = []
        if segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "confidence": word.probability,
                })
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words,
        })

    return results


# ── Audio Analysis ───────────────────────────────────────────────────────────

def analyze_audio(job_id: str) -> Optional[dict]:
    """Analyze audio with librosa to detect excitement, loudness, and reactions.

    Extracts RMS energy, spectral centroid, zero-crossing rate, and tempo
    to identify exciting moments, loud segments, and laughter/shouting patterns.

    Args:
        job_id: Job identifier.

    Returns:
        Dict with keys:
            - excitement_moments: list of {start, end, energy, centroid}
            - loud_segments: list of {start, end, rms_db}
            - laughter_segments: list of {start, end, confidence}
            - shouting_segments: list of {start, end, confidence}
            - overall_energy: float (mean RMS)
            - energy_profile: list of {time, rms} sampled per second
    """
    ap = _audio_path(job_id)
    if not os.path.exists(ap):
        return None

    try:
        import librosa
        import numpy as np
    except ImportError:
        return None

    y, sr = librosa.load(ap, sr=22050, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    hop_length = 512
    frame_length = 2048

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(rms, sr=sr, hop_length=hop_length)
    rms_db = 20 * np.log10(rms + 1e-10)
    overall_energy = float(np.mean(rms_db))

    # Build energy profile (per-second sampling)
    energy_profile: list[dict] = []
    for t in range(int(duration) + 1):
        mask = (rms_times >= t) & (rms_times < t + 1)
        if np.any(mask):
            energy_profile.append({"time": t, "rms": float(np.mean(rms_db[mask]))})
        else:
            energy_profile.append({"time": t, "rms": -80.0})

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid_times = librosa.frames_to_time(centroid, sr=sr, hop_length=hop_length)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    zcr_times = librosa.frames_to_time(zcr, sr=sr, hop_length=hop_length)

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_env, sr=sr, hop_length=hop_length)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Loud segments (above mean + 0.5 * std)
    loud_threshold = float(np.mean(rms_db) + 0.5 * np.std(rms_db))
    loud_segments: list[dict] = []
    in_loud = False
    loud_start = 0.0
    for i, db in enumerate(rms_db):
        if db > loud_threshold and not in_loud:
            in_loud = True
            loud_start = float(rms_times[i])
        elif db <= loud_threshold and in_loud:
            in_loud = False
            loud_segments.append({"start": round(loud_start, 1), "end": round(float(rms_times[i]), 1), "rms_db": round(db, 1)})

    # Excitement moments (high energy + high centroid)
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-10)
    centroid_norm = (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid) + 1e-10)
    onset_norm = (onset_env - np.min(onset_env)) / (np.max(onset_env) - np.min(onset_env) + 1e-10)

    min_frames = 4
    excitement_signal = rms_norm * 0.4 + centroid_norm * 0.3 + onset_norm * 0.3
    excite_times = librosa.frames_to_time(range(len(excitement_signal)), sr=sr, hop_length=hop_length)
    excite_threshold = np.mean(excitement_signal) + 0.8 * np.std(excitement_signal)

    excitement_moments: list[dict] = []
    in_excite = False
    excite_start = 0.0
    for i, val in enumerate(excitement_signal):
        if val > excite_threshold and not in_excite:
            in_excite = True
            excite_start = float(excite_times[i])
        elif val <= excite_threshold and in_excite:
            in_excite = False
            idx_start = max(0, i - 2)
            excitement_moments.append({
                "start": round(excite_start, 1),
                "end": round(float(excite_times[i]), 1),
                "energy": round(float(np.mean(excitement_signal[idx_start:i])), 3),
                "centroid": round(float(np.mean(centroid_norm[idx_start:i])), 3),
            })

    # Laughter detection (high ZCR + moderate energy)
    laughter_segments: list[dict] = []
    zcr_threshold = np.mean(zcr) + 1.5 * np.std(zcr)
    in_laugh = False
    laugh_start = 0.0
    for i, val in enumerate(zcr):
        if val > zcr_threshold and not in_laugh:
            in_laugh = True
            laugh_start = float(zcr_times[i])
        elif val <= zcr_threshold and in_laugh:
            in_laugh = False
            min_len = 0.2
            if zcr_times[i] - laugh_start >= min_len:
                laughter_segments.append({
                    "start": round(laugh_start, 1),
                    "end": round(float(zcr_times[i]), 1),
                    "confidence": round(float(np.mean(zcr[max(0, i-5):i])), 3),
                })

    # Shouting detection (high RMS + high centroid)
    shouting_segments: list[dict] = []
    shout_threshold_rms = np.mean(rms_db) + 1.0 * np.std(rms_db)
    shout_threshold_centroid = float(np.mean(centroid_norm) + 0.7 * np.std(centroid_norm))
    in_shout = False
    shout_start = 0.0
    for i in range(len(rms_db)):
        rms_val = rms_db[i]
        cent_idx = min(i, len(centroid_norm) - 1)
        cent_val = centroid_norm[cent_idx]
        if rms_val > shout_threshold_rms and cent_val > shout_threshold_centroid and not in_shout:
            in_shout = True
            shout_start = float(rms_times[i])
        elif (rms_val <= shout_threshold_rms or cent_val <= shout_threshold_centroid) and in_shout:
            in_shout = False
            if rms_times[i] - shout_start >= 0.5:
                shouting_segments.append({
                    "start": round(shout_start, 1),
                    "end": round(float(rms_times[i]), 1),
                    "confidence": round(float(cent_val), 3),
                })

    return {
        "excitement_moments": excitement_moments,
        "loud_segments": loud_segments,
        "laughter_segments": laughter_segments,
        "shouting_segments": shouting_segments,
        "overall_energy": round(overall_energy, 2),
        "energy_profile": energy_profile,
        "tempo": float(tempo),
    }


# ── Scene Detection ────────────────────────────────────────────────────────

def detect_scenes(filepath: str, threshold: float = 0.3) -> list[dict]:
    """Detect scene changes in video using FFmpeg's scene change filter.

    Args:
        filepath: Path to the video file.
        threshold: Scene change sensitivity (0.0-1.0). Lower = more sensitive.

    Returns:
        List of scene dicts with keys: time, score.
    """
    cmd = [
        "ffmpeg", "-i", filepath,
        "-an",
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    result = _run(cmd, timeout=600)
    scenes: list[dict] = []
    for match in re.finditer(r"pts_time:(\d+\.?\d*)", result.stderr or ""):
        scenes.append({
            "time": round(float(match.group(1)), 2),
            "score": threshold,
        })
    return scenes


# ── Speaker Change Detection ────────────────────────────────────────────────

def detect_speaker_changes(job_id: str) -> list[dict]:
    """Detect speaker changes using energy + pitch heuristics (simplified pyannote-style).

    Analyzes audio in short windows, extracts energy and pitch features,
    and detects significant changes that indicate a different speaker.

    Args:
        job_id: Job identifier.

    Returns:
        List of speaker segment dicts with keys: start, end, speaker_id, confidence.
    """
    ap = _audio_path(job_id)
    if not os.path.exists(ap):
        return []

    try:
        import librosa
        import numpy as np
    except ImportError:
        return []

    y, sr = librosa.load(ap, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    window_sec = 0.5
    hop_sec = 0.25
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    features: list[dict] = []
    for start_sample in range(0, len(y) - window_samples, hop_samples):
        t = start_sample / sr
        window = y[start_sample:start_sample + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(window, fmin=50, fmax=500, sr=sr)
            valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
            pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0
        except Exception:
            pitch_mean = 0.0
            pitch_std = 0.0

        spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=window, sr=sr)))
        features.append({
            "time": t,
            "rms": rms,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "spectral_centroid": spec_cent,
        })

    if len(features) < 3:
        return []

    # Calculate change scores
    change_scores: list[float] = [0.0]
    for i in range(1, len(features)):
        prev = features[i - 1]
        curr = features[i]
        rms_diff = abs(np.log10(curr["rms"] + 1e-10) - np.log10(prev["rms"] + 1e-10))
        pitch_diff = abs(curr["pitch_mean"] - prev["pitch_mean"]) / 50.0
        cent_diff = abs(curr["spectral_centroid"] - prev["spectral_centroid"]) / 1000.0
        change = rms_diff * 0.3 + pitch_diff * 0.4 + cent_diff * 0.3
        change_scores.append(min(1.0, change))

    mean_change = float(np.mean(change_scores))
    std_change = float(np.std(change_scores))
    threshold = mean_change + 1.5 * std_change

    change_points: list[dict] = []
    for i, score in enumerate(change_scores):
        if score > threshold:
            change_points.append({"time": features[i]["time"], "score": round(score, 3)})

    # Build speaker segments from change points
    boundaries = [0.0] + [cp["time"] for cp in change_points] + [duration]
    speaker_segments: list[dict] = []
    for i in range(len(boundaries) - 1):
        mid_idx = min(i, len(features) - 1)
        conf = min(1.0, change_scores[mid_idx] / max(threshold, 0.01)) if mid_idx < len(change_scores) else 0.5
        speaker_segments.append({
            "start": round(boundaries[i], 1),
            "end": round(boundaries[i + 1], 1),
            "speaker_id": i,
            "confidence": round(conf, 2),
        })

    return speaker_segments


# ── Hook Detection ──────────────────────────────────────────────────────────

def detect_hooks(
    transcriptions: list[dict],
    audio_data: Optional[dict],
    scenes: list[dict],
) -> dict[float, float]:
    """Analyze the first 3 seconds of each transcript segment for hook quality.

    Scores each segment's opening for attention-grabbing potential based on:
    - Hook phrases (e.g., "watch this", "you won't believe")
    - Question marks and curiosity triggers
    - Excitement words in the first 3 seconds
    - Audio energy in the first 3 seconds
    - Scene changes near the start
    - Laughter/shouting in the first 3 seconds

    Args:
        transcriptions: List of transcript segments.
        audio_data: Output from analyze_audio().
        scenes: List of scene change dicts.

    Returns:
        Dict mapping segment start time -> hook score (0.0-1.0).
    """
    hook_scores: dict[float, float] = {}

    excitement_intervals = audio_data.get("excitement_moments", []) if audio_data else []
    laughter_intervals = audio_data.get("laughter_segments", []) if audio_data else []
    shouting_intervals = audio_data.get("shouting_segments", []) if audio_data else []
    scene_times = [s["time"] for s in scenes]

    for seg in transcriptions:
        seg_start = seg["start"]
        seg_end = seg["end"]
        text = seg.get("text", "")
        text_lower = text.lower()
        words = seg.get("words", [])

        score = 0.0
        max_score = 0.0

        # Hook phrases
        phrase_score = 0.0
        for phrase in HOOK_PHRASES:
            if phrase in text_lower:
                phrase_score = max(phrase_score, 0.3)
        score += phrase_score
        max_score += 0.3

        # Questions
        question_score = 0.0
        for pattern in QUESTION_PATTERNS:
            if pattern.search(text):
                question_score = max(question_score, 0.2)
                break
        score += question_score
        max_score += 0.2

        # Excitement words in first 3 seconds
        early_words = [w for w in words if w.get("start", 0) - seg_start <= 3.0]
        early_text = " ".join(w.get("word", "") for w in early_words).lower()
        excite_score = 0.0
        for word, intensity in EXCITEMENT_WORDS.items():
            if word in early_text:
                excite_score = max(excite_score, intensity * 0.15)
        score += excite_score
        max_score += 0.15

        # Audio energy in first 3 seconds
        if audio_data and "energy_profile" in audio_data:
            energy_score = 0.0
            energies_in_window = [
                e["rms"] for e in audio_data["energy_profile"]
                if seg_start <= e["time"] <= seg_start + 3.0
            ]
            if energies_in_window:
                avg_energy = sum(energies_in_window) / len(energies_in_window)
                normalized = (avg_energy + 80) / 80  # normalize from [-80, 0] to [0, 1]
                energy_score = max(0, min(0.15, normalized * 0.15))
            score += energy_score
            max_score += 0.15

        # Scene changes near start
        scene_score = 0.0
        for st in scene_times:
            if 0 <= st - seg_start <= 2.0:
                scene_score = max(scene_score, 0.1)
        score += scene_score
        max_score += 0.1

        # Laughter/shouting in first 3 seconds
        reaction_score = 0.0
        for interval in laughter_intervals + shouting_intervals:
            if seg_start <= interval["start"] <= seg_start + 3.0:
                reaction_score = max(reaction_score, 0.1)
        score += reaction_score
        max_score += 0.1

        hook_scores[seg_start] = round(min(1.0, score / max(max_score, 0.01)), 3)

    return hook_scores


# ── Topic Change Detection ──────────────────────────────────────────────────

def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, removing stop words."""
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w not in TOPIC_STOP_WORDS and len(w) > 3}


def _keyword_similarity(kw1: set[str], kw2: set[str]) -> float:
    """Jaccard-like similarity between two keyword sets."""
    intersection = kw1 & kw2
    union = kw1 | kw2
    return len(intersection) / len(union) if union else 0.0


def detect_topic_changes(
    transcriptions: list[dict],
    window_size: int = 5,
) -> list[dict]:
    """Detect topic shifts using keyword clustering across transcript segments.

    Slides a window over the transcript, extracts keywords, and detects
    significant drops in keyword similarity between consecutive windows.

    Args:
        transcriptions: List of transcript segments.
        window_size: Number of segments per analysis window.

    Returns:
        List of topic change dicts with keys: time, similarity_drop, keywords_before, keywords_after.
    """
    if len(transcriptions) < window_size * 2:
        return []

    changes: list[dict] = []
    for i in range(len(transcriptions) - window_size * 2 + 1):
        window1 = " ".join(t.get("text", "") for t in transcriptions[i:i + window_size])
        text1 = window1
        kw1 = _extract_keywords(text1)

        window2 = " ".join(t.get("text", "") for t in transcriptions[i + window_size:i + window_size * 2])
        text2 = window2
        kw2 = _extract_keywords(text2)

        similarity = _keyword_similarity(kw1, kw2)
        if similarity < 0.15:
            change_time = transcriptions[i + window_size].get("start", 0)
            changes.append({
                "time": round(change_time, 1),
                "similarity_drop": round(1.0 - similarity, 3),
                "keywords_before": list(kw1)[:10],
                "keywords_after": list(kw2)[:10],
            })

    return changes


# ── Viral Score Calculation ─────────────────────────────────────────────────

def calculate_viral_score(
    segment: dict,
    transcript: dict,
    audio_features: dict,
    hook_score: float,
    scene_data: list[dict],
) -> dict:
    """Compute a viral score (1-100) for a clip segment using weighted multi-signal analysis.

    Components (weights sum to 1.0):
        - Excitement words (20%): Intensity-weighted keyword matching
        - Audio energy (20%): RMS energy within the clip window
        - Speech density (15%): Words per second
        - Hook quality (15%): Opening 3-second hook score
        - Scene dynamics (10%): Scene changes within/near clip
        - Caps/shouting (10%): ALL CAPS ratio + audio shouting detection
        - Questions/curiosity (10%): Question marks + curiosity patterns

    Args:
        segment: Clip segment dict with start, end keys.
        transcript: Matching transcript segment with text, words.
        audio_features: Output from analyze_audio().
        hook_score: Hook score (0-1) from detect_hooks().
        scene_data: List of scene change dicts.

    Returns:
        Dict with keys: viral_score (int 1-100), component_scores (dict), reasons (list[str]).
    """
    text = transcript.get("text", "")
    text_lower = text.lower()
    words = transcript.get("words", [])
    seg_start = segment.get("start", 0)
    seg_end = segment.get("end", seg_start + 30)
    seg_duration = max(seg_end - seg_start, 0.1)

    component_scores: dict[str, float] = {}
    reasons: list[str] = []

    # Excitement words
    excite_score = 0.0
    for word, intensity in EXCITEMENT_WORDS.items():
        if word in text_lower:
            excite_score = max(excite_score, intensity)
    component_scores["excitement_words"] = round(excite_score, 3)
    if excite_score > 0.5:
        reasons.append(f"excitement words")

    # Audio energy
    energy_score = 0.0
    if audio_features and "energy_profile" in audio_features:
        energies = [e["rms"] for e in audio_features["energy_profile"] if seg_start <= e["time"] <= seg_end]
        if energies:
            avg_energy = sum(energies) / len(energies)
            energy_score = max(0, min(1.0, (avg_energy + 80) / 80))
    component_scores["audio_energy"] = round(energy_score, 3)
    if energy_score > 0.7:
        reasons.append("High audio energy")

    # Speech density
    word_count = len(words) if words else len(text.split())
    density = word_count / seg_duration
    density_score = min(1.0, density / 4.0)  # 4 words/sec = max
    component_scores["speech_density"] = round(density_score, 3)
    if density > 3.5:
        reasons.append("Fast speech")

    # Hook quality
    hook_normalized = min(1.0, hook_score)
    component_scores["hook_quality"] = round(hook_normalized, 3)
    if hook_normalized > 0.5:
        reasons.append("Strong hook opening")

    # Scene dynamics
    scenes_in_clip = sum(1 for s in scene_data if seg_start <= s["time"] <= seg_end)
    scenes_nearby = sum(1 for s in scene_data if abs(s["time"] - seg_start) <= 2.0)
    total_scenes = scenes_in_clip + scenes_nearby
    scene_score = min(1.0, total_scenes / 3.0)
    component_scores["scene_dynamics"] = round(scene_score, 3)
    if total_scenes > 0:
        reasons.append(f"{total_scenes} scene changes")

    # Caps/shouting
    caps_score = 0.0
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.5:
        caps_score = 0.5
        reasons.append("ALL CAPS detected")
    if audio_features and "shouting_segments" in audio_features:
        for si in audio_features["shouting_segments"]:
            if seg_start <= si["start"] <= seg_end:
                caps_score = max(caps_score, si.get("confidence", 0.5))
                reasons.append("Audio shouting detected")
    component_scores["caps_shouting"] = round(caps_score, 3)

    # Questions
    question_score = 0.0
    for pattern in QUESTION_PATTERNS:
        if pattern.search(text):
            question_score = 0.5
            break
    q_count = text.count("?")
    if q_count > 0:
        question_score = min(1.0, question_score + 0.2 * q_count)
    curiosity_words = ("imagine", "what if", "ever wonder", "did you know", "you ever")
    for cw in curiosity_words:
        if cw in text_lower:
            question_score = min(1.0, question_score + 0.3)
            reasons.append("Curiosity trigger")
            break
    component_scores["questions"] = round(question_score, 3)
    if question_score > 0.3:
        reasons.append("Question/curiosity trigger")

    # Weighted sum
    viral_score = (
        component_scores["excitement_words"] * WEIGHT_EXCITEMENT_WORDS
        + component_scores["audio_energy"] * WEIGHT_AUDIO_ENERGY
        + component_scores["speech_density"] * WEIGHT_SPEECH_DENSITY
        + component_scores["hook_quality"] * WEIGHT_HOOK_QUALITY
        + component_scores["scene_dynamics"] * WEIGHT_SCENE_DYNAMICS
        + component_scores["caps_shouting"] * WEIGHT_CAPS_SHOUTING
        + component_scores["questions"] * WEIGHT_QUESTIONS
    )

    viral_score_int = max(1, min(100, int(viral_score * 100)))
    if not reasons:
        reasons.append("Interesting segment")

    return {
        "viral_score": viral_score_int,
        "component_scores": component_scores,
        "reasons": reasons,
    }


# ── Main Highlight Detection ────────────────────────────────────────────────

def find_highlights(
    transcriptions: list[dict],
    scenes: list[dict],
    audio_data: Optional[dict],
    duration: float,
    num_clips: int = 5,
) -> list[dict]:
    """Main orchestrator: find the best viral clips from a video.

    Combines all analysis signals to generate non-overlapping highlight clips
    ranked by viral score.

    Args:
        transcriptions: List of transcript segments from transcribe_audio().
        scenes: List of scene change dicts from detect_scenes().
        audio_data: Audio analysis dict from analyze_audio().
        duration: Total video duration in seconds.
        num_clips: Maximum number of clips to return (5-20).

    Returns:
        List of highlight dicts sorted by viral_score descending, each with:
            - id: int
            - start: float (seconds)
            - end: float (seconds)
            - viral_score: int (1-100)
            - hook_score: float (0-1)
            - reason: str
            - text: str (transcript text)
            - audio_features: dict (energy, excitement, etc. for this clip)
            - speaker_changes: int (count within clip)
            - scene_changes: int (count within clip)
    """
    num_clips = max(5, min(20, num_clips))
    highlights: list[dict] = []

    if not transcriptions:
        return []

    # Detect hooks
    hook_scores = detect_hooks(transcriptions, audio_data, scenes)

    # Detect topic changes
    topic_changes = detect_topic_changes(transcriptions)

    # Generate candidate clips around exciting moments
    candidates: list[dict] = []
    MIN_CLIP_SPACING = max(5.0, min(15.0, duration * 0.2))  # at least 5s, at most 15s, or 20% of duration

    for i, scene in enumerate(scenes):
        st = scene["time"]
        et = st + 47  # ~47 second clips

        # Find matching transcript
        transcript = {}
        for t in transcriptions:
            if t["start"] <= st < t["end"]:
                transcript = t
                break

        clip_start = max(0, st - 3)
        clip_end = min(duration, et)

        # Extend to include nearby transcript
        for j in range(i, min(i + 5, len(transcriptions))):
            next_seg = transcriptions[j]
            if next_seg["start"] <= clip_end + 3.0:
                clip_end = min(duration, next_seg["end"])

        # Find nearby excitement
        next_excite = None
        if audio_data and "excitement_moments" in audio_data:
            for exc in audio_data["excitement_moments"]:
                if exc["start"] >= clip_start:
                    next_excite = exc
                    break

        hook_score = hook_scores.get(transcript.get("start", 0), 0.0)

        # Calculate viral score
        score_result = calculate_viral_score(
            {"start": clip_start, "end": clip_end},
            transcript,
            audio_data or {},
            hook_score,
            scenes,
        )

        # Count speaker changes and scene changes
        spk_changes = 0
        scn_changes = sum(1 for s in scenes if clip_start <= s["time"] <= clip_end)

        # Topic change bonus
        topic_bonus = 0.0
        for tc in topic_changes:
            if abs(tc["time"] - clip_start) <= 5.0:
                topic_bonus = 1.0
                score_result["reasons"].append("Topic change nearby")

        # Audio features for this clip
        clip_audio: dict = {}
        if audio_data and "energy_profile" in audio_data:
            clip_energies = [e["rms"] for e in audio_data["energy_profile"] if clip_start <= e["time"] <= clip_end]
            clip_audio["avg_energy"] = round(sum(clip_energies) / len(clip_energies), 2) if clip_energies else 0
            clip_audio["max_energy"] = round(max(clip_energies), 2) if clip_energies else 0
            clip_audio["excitement_moments"] = [e for e in audio_data.get("excitement_moments", []) if clip_start <= e["start"] <= clip_end]
            clip_audio["laughter_segments"] = [e for e in audio_data.get("laughter_segments", []) if clip_start <= e["start"] <= clip_end]
            clip_audio["shouting_segments"] = [e for e in audio_data.get("shouting_segments", []) if clip_start <= e["start"] <= clip_end]

        viral_score = min(100, score_result["viral_score"] + int(topic_bonus * 10))

        candidates.append({
            "id": i,
            "start": round(clip_start, 1),
            "end": round(clip_end, 1),
            "viral_score": viral_score,
            "hook_score": hook_score,
            "reason": ", ".join(score_result["reasons"]),
            "text": transcript.get("text", ""),
            "audio_features": clip_audio,
            "speaker_changes": spk_changes,
            "scene_changes": scn_changes,
            "_component_scores": score_result["component_scores"],
        })

    # Fallback: if no scenes detected, generate candidates from transcript segments
    if not candidates and transcriptions:
        log.info("No scenes detected — generating candidates from transcripts (%d segments)", len(transcriptions))
        clip_duration = 47.0
        num_candidates = min(num_clips * 3, len(transcriptions))
        step = max(1, len(transcriptions) // num_candidates)
        for idx in range(0, len(transcriptions), step):
            seg = transcriptions[idx]
            clip_start = max(0, seg["start"] - 3)
            clip_end = min(duration, clip_start + clip_duration)
            if clip_end - clip_start < 10:
                clip_end = min(duration, clip_start + 30)

            hook_score = hook_scores.get(seg.get("start", 0), 0.0)
            score_result = calculate_viral_score(
                {"start": clip_start, "end": clip_end},
                seg,
                audio_data or {},
                hook_score,
                scenes,
            )
            spk_changes = 0
            scn_changes = 0
            topic_bonus = 0.0
            for tc in topic_changes:
                if abs(tc["time"] - clip_start) <= 5.0:
                    topic_bonus = 1.0
                    score_result["reasons"].append("Topic change nearby")

            viral_score = min(100, score_result["viral_score"] + int(topic_bonus * 10))
            candidates.append({
                "id": idx,
                "start": round(clip_start, 1),
                "end": round(clip_end, 1),
                "viral_score": viral_score,
                "hook_score": hook_score,
                "reason": ", ".join(score_result["reasons"]) or "Transcript segment",
                "text": seg.get("text", ""),
                "audio_features": {},
                "speaker_changes": spk_changes,
                "scene_changes": scn_changes,
                "_component_scores": score_result["component_scores"],
            })
            if len(candidates) >= num_clips * 3:
                break

    # Sort by viral score and filter overlapping
    candidates.sort(key=lambda c: c["viral_score"], reverse=True)

    filtered: list[dict] = []
    used_ranges: list[tuple[float, float]] = []

    for c in candidates:
        overlaps = False
        for us, ue in used_ranges:
            if abs(c["start"] - us) < MIN_CLIP_SPACING:
                overlaps = True
                break
        if not overlaps:
            filtered.append(c)
            used_ranges.append((c["start"], c["end"]))
        if len(filtered) >= num_clips:
            break

    # Sort by start time for final output
    filtered.sort(key=lambda c: c["start"])

    # Clean up internal fields
    final: list[dict] = []
    for h in filtered:
        h.pop("_component_scores", None)
        h.pop("id", None)
        final.append(h)

    return final


# ── Cleanup ─────────────────────────────────────────────────────────────────

def cleanup_job(job_id: str) -> None:
    """Remove downloaded files for a job.

    Args:
        job_id: Job identifier whose files should be removed.
    """
    import shutil
    jdir = _job_dir(job_id)
    if os.path.exists(jdir):
        shutil.rmtree(str(jdir), ignore_errors=True)


# ── Ollama-powered Features ─────────────────────────────────────────────────

def detect_chapters(
    transcript: list[dict],
    duration: float,
    use_ollama: bool = True,
) -> list[dict]:
    """
    Split a transcript into chapters by topic.

    Args:
        transcript: List of segment dicts with 'start', 'end', 'text' keys.
        duration: Total video duration in seconds.
        use_ollama: Use Ollama if available, else heuristic.

    Returns:
        List of chapter dicts:
            [{"start": 0.0, "end": 45.2, "title": "Intro", "summary": "..."}, ...]
    """
    if not transcript or duration <= 0:
        return []

    # Build timestamped transcript text
    lines = []
    for seg in transcript:
        ts = _format_time(seg["start"])
        lines.append(f"[{ts}] {seg['text'].strip()}")
    full_text = "\n".join(lines)

    if use_ollama:
        try:
            from core.ollama_client import chat, is_available
            if is_available():
                chapters = _detect_chapters_ollama(chat, full_text, duration)
                if chapters:
                    log.info("Ollama detected %d chapters", len(chapters))
                    return chapters
        except Exception as e:
            log.warning("Ollama chapter detection failed: %s", e)

    # Fallback: heuristic topic detection
    chapters = _detect_chapters_heuristic(transcript, duration)
    log.info("Heuristic detected %d chapters", len(chapters))
    return chapters


def score_hot_takes(
    clips: list[dict],
    use_ollama: bool = True,
) -> list[dict]:
    """
    Score each clip for "hot take" potential.

    Args:
        clips: List of clip dicts with 'text' key (transcript text of the clip).
        use_ollama: Use Ollama if available, else heuristic.

    Returns:
        Same clips list with added 'hot_take_score' key (0-10).
    """
    if use_ollama:
        try:
            from core.ollama_client import chat, is_available
            if is_available():
                return _score_hot_takes_ollama(chat, clips)
        except Exception as e:
            log.warning("Ollama hot take scoring failed: %s, using heuristic", e)
    return _score_hot_takes_heuristic(clips)


# ── Ollama Implementations ──────────────────────────────────────────────────

def _detect_chapters_ollama(chat_fn, full_text: str, duration: float) -> list[dict]:
    """Use LLM to detect topic changes and generate chapter titles."""
    import json as _json

    # Send full transcript if reasonable, otherwise sample evenly
    # ~8000 chars fits safely in 8192 token context with the prompt
    if len(full_text) > 8000:
        lines = full_text.split("\n")
        step = max(1, len(lines) // 150)  # ~150 lines covers full video
        sampled = lines[::step]
        transcript_for_prompt = "\n".join(sampled)
        log.info("Chapter detection: sampled %d/%d transcript lines for Ollama", len(sampled), len(lines))
    else:
        transcript_for_prompt = full_text

    # Scale chapter count to video duration (1 chapter per ~10 min, min 5, max 20)
    import math
    target_chapters = max(5, min(20, math.ceil(duration / 600)))

    prompt = f"""You are a podcast episode analyzer. Split this transcript into logical chapters by topic.

Transcript length: {duration/60:.0f} minutes
Timestamped transcript:
{transcript_for_prompt}

Rules:
- Provide {target_chapters} chapters (approximately one per {max(3, int(duration/60/target_chapters))} minutes)
- Each chapter should be at least 2 minutes long
- Focus on major topic changes, not speaker changes
- Use the timestamps in the transcript to determine start times

Respond in JSON format:
[{{"start_time": "0:00", "title": "Introduction", "summary": "The hosts introduce the topic."}}]

Provide EXACTLY {target_chapters} chapters covering the ENTIRE {duration/60:.0f}-minute video."""

    try:
        resp = chat_fn(prompt, temperature=0.2, timeout=60, num_ctx=4096)
        match = re.search(r"\[.*\]", resp, re.DOTALL)
        if not match:
            return []

        chapters = _json.loads(match.group(0))

        result = []
        for i, ch in enumerate(chapters):
            start_sec = _parse_time(ch.get("start_time", "0:00"))
            if i + 1 < len(chapters):
                end_sec = _parse_time(chapters[i + 1].get("start_time", "0:00"))
            else:
                end_sec = duration
            result.append({
                "start": start_sec,
                "end": end_sec,
                "title": ch.get("title", f"Chapter {i+1}"),
                "summary": ch.get("summary", ""),
            })
        return result

    except Exception as e:
        log.warning("Ollama chapter parse failed: %s", e)
        return []


def _score_hot_takes_ollama(chat_fn, clips: list[dict]) -> list[dict]:
    """Use LLM to rate each clip's hot-take potential."""
    prompt_parts = [
        "Rate how much of a 'hot take' each clip is (0-10 scale).\n"
        "A hot take = bold, controversial, surprising, or provocative statement.\n"
        "Consider: strong opinions, challenging conventional wisdom, provocative claims.\n"
        "Reply with just numbers, one per line.\n"
    ]

    for i, clip in enumerate(clips):
        text = clip.get("text", "")[:300]
        prompt_parts.append(f"\nClip {i+1}: {text}")

    try:
        resp = chat_fn("\n".join(prompt_parts), temperature=0.2, timeout=30)
        scores = []
        for line in resp.strip().split("\n"):
            nums = re.findall(r"\d+", line)
            if nums:
                scores.append(min(10, max(0, int(nums[0]))))
            else:
                scores.append(0)

        for i, clip in enumerate(clips):
            clip["hot_take_score"] = scores[i] if i < len(scores) else 0
        return clips

    except Exception as e:
        log.warning("Ollama hot take scoring failed: %s", e)
        return _score_hot_takes_heuristic(clips)


# ── Heuristic Fallbacks ─────────────────────────────────────────────────────

HOT_TAKE_KEYWORDS = [
    "hot take", "unpopular opinion", "controversial", "bold take",
    "truth is", "real talk", "hear me out", "i'll say what",
    "everyone thinks", "but actually", "the problem is",
    "nobody talks about", "let's be honest", "unpopular but",
    "i'm gonna say it", "controversy", "takes courage",
    "most people don't", "the real reason", "what if i said",
]

HOT_TAKE_PATTERNS = [
    re.compile(r"\bi (?:think|believe|feel) (?:that )?everyone", re.IGNORECASE),
    re.compile(r"\bthe truth is\b", re.IGNORECASE),
    re.compile(r"\bno one (?:talks?|wants?) (?:about|to)", re.IGNORECASE),
    re.compile(r"\blet me be (?:honest|real|frank)", re.IGNORECASE),
    re.compile(r"\bhot take\b", re.IGNORECASE),
]


def _score_hot_takes_heuristic(clips: list[dict]) -> list[dict]:
    """Score hot takes using keyword matching and text analysis."""
    for clip in clips:
        text = clip.get("text", "").lower()
        score = 0

        for kw in HOT_TAKE_KEYWORDS:
            if kw in text:
                score += 2

        for pat in HOT_TAKE_PATTERNS:
            if pat.search(text):
                score += 3

        sentences = re.split(r"[.!?]+", text)
        avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_len < 8:
            score += 1

        if re.search(r"\bi (?:think|believe|feel|know)\b", text):
            score += 1

        clip["hot_take_score"] = min(10, score)

    return clips


def _detect_chapters_heuristic(transcript: list[dict], duration: float) -> list[dict]:
    """Fallback: detect chapters by looking for long pauses and topic-keyword shifts."""
    if not transcript:
        return []

    window_size = 120  # seconds
    chapters: list[dict] = []
    current_start = 0.0
    current_keywords: set[str] = set()

    for i, seg in enumerate(transcript):
        if seg["start"] - current_start >= window_size:
            new_keywords = _extract_keywords_simple(seg["text"])
            overlap = len(current_keywords & new_keywords) / max(len(current_keywords | new_keywords), 1)

            if overlap < 0.3 and current_keywords:
                topic = " / ".join(list(current_keywords)[:2]).title()
                chapters.append({
                    "start": current_start,
                    "end": seg["start"],
                    "title": topic or f"Chapter {len(chapters)+1}",
                    "summary": "",
                })
                current_start = seg["start"]
            current_keywords = new_keywords
        else:
            current_keywords |= _extract_keywords_simple(seg["text"])

    if duration - current_start > 10:
        topic = " / ".join(list(current_keywords)[:2]).title()
        chapters.append({
            "start": current_start,
            "end": duration,
            "title": topic or f"Chapter {len(chapters)+1}",
            "summary": "",
        })

    if not chapters:
        chapters = [{"start": 0, "end": duration, "title": "Full Episode", "summary": ""}]

    return chapters


def _extract_keywords_simple(text: str, top_n: int = 10) -> set[str]:
    """Simple keyword extraction."""
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    words = [w for w in words if w not in TOPIC_STOP_WORDS]
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return {w for w, _ in sorted_words[:top_n]}


# ── Time Utilities ──────────────────────────────────────────────────────────

def _format_time(seconds: float) -> str:
    """Format seconds as m:ss or mm:ss."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _parse_time(ts: str) -> float:
    """Parse 'm:ss' or 'mm:ss' or 'h:mm:ss' to seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0
