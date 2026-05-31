"""
Ollama API client for local LLM inference.

Provides topic detection and hot take scoring using a local Ollama model.
Falls back gracefully if Ollama is not running.

Configuration via environment variables:
    OLLAMA_URL     - Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL   - Model name (default: llama3.1:8b)
    OLLAMA_TIMEOUT - Request timeout in seconds (default: 120)
"""

from __future__ import annotations

import os
import re
import logging
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))


def is_available() -> bool:
    """Check if Ollama server is reachable and responsive."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def chat(prompt: str, system: str | None = None, temperature: float = 0.3, timeout: int | None = None, num_ctx: int = 8192) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    if system:
        payload["system"] = system

    logger.debug("Ollama request: model=%s prompt_len=%d", OLLAMA_MODEL, len(prompt))
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=timeout or OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["response"]


def generate(prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
    """Alias for chat()."""
    return chat(prompt, system=system, temperature=temperature)


def generate_tiktok_caption(transcript_text: str, video_title: str = "") -> str:
    """
    Generate a short, engaging TikTok caption from transcript text using Ollama.

    Returns a 1-2 sentence caption that's grammatically correct and sounds natural.
    Falls back to a cleaned-up snippet if Ollama is unavailable.
    """
    if not transcript_text or not transcript_text.strip():
        return ""

    # Truncate to keep the prompt small — we only need the gist
    text_preview = transcript_text[:600].strip()

    system = (
        "You write TikTok captions. Rules:\n"
        "- 1-2 short sentences max\n"
        "- Sound natural and conversational, like a real person talking\n"
        "- NO hashtags, NO emojis, NO all-caps shouting\n"
        "- Fix any grammar issues from the transcript\n"
        "- If the transcript is gibberish or too fragmented, write a generic hook like 'Wait for it…' or 'This part hit different'\n"
        "- Output ONLY the caption text, nothing else"
    )

    user_prompt = (
        f"Transcript: \"{text_preview}\"\n"
        f"{'Original video title: ' + video_title if video_title else ''}\n\n"
        "Write a short TikTok caption that captures the best part:"
    )

    try:
        caption = chat(
            user_prompt,
            system=system,
            temperature=0.7,
            timeout=15,
            num_ctx=4096,
        ).strip()
        # Clean up: remove quotes if the model wraps it
        caption = caption.strip('"').strip("'")
        # Truncate to TikTok caption limit (2200 chars, but keep it short)
        if len(caption) > 300:
            caption = caption[:297] + "…"
        # If output looks weird (empty, just hashtags, etc.), fall back
        if len(caption) < 3:
            raise ValueError("Model returned empty/too short caption")
        return caption
    except Exception as exc:
        logger.warning("Ollama caption generation failed: %s, using fallback", exc)
        # Fallback: take first sentence, clean it up
        sentences = re.split(r'[.!?]+', transcript_text.strip())
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                # Capitalize first letter, add period
                s = s[0].upper() + s[1:]
                if not s.endswith(('.', '!', '?')):
                    s += '.'
                return s[:200]
        return transcript_text.strip()[:100]


def generate_hashtags(transcript_text: str, video_title: str = "", max_tags: int = 5) -> list[str]:
    """
    Generate relevant hashtags from transcript text using Ollama.
    Returns a list of hashtag strings (without # prefix).
    Falls back to simple keyword extraction if Ollama is unavailable.
    """
    if not transcript_text or not transcript_text.strip():
        return []

    text_preview = transcript_text[:500].strip()

    system = (
        "You generate social media hashtags. Rules:\n"
        f"- Output exactly {max_tags} hashtags, comma-separated\n"
        f"- All lowercase, no spaces, no special characters\n"
        f"- Relevant to the content, not generic (#fyp #viral are banned)\n"
        "- Mix of broad and specific tags\n"
        "- Output ONLY the comma-separated hashtags, nothing else"
    )

    user_prompt = (
        f"Transcript: \"{text_preview}\"\n"
        f"{'Original title: ' + video_title if video_title else ''}\n\n"
        f"Generate {max_tags} relevant hashtags:"
    )

    try:
        response = chat(
            user_prompt,
            system=system,
            temperature=0.5,
            timeout=12,
            num_ctx=4096,
        ).strip()
        # Parse hashtags — model may return comma-separated OR space-separated
        tags = []
        # First try comma-separated
        raw_items = response.split(",") if "," in response else response.split()
        for t in raw_items:
            t = t.strip().lower()
            t = re.sub(r"[^a-z0-9]", "", t)  # keep only alphanumeric
            if t and len(t) >= 3 and t not in ("fyp", "viral", "foryou", "foryoupage", "trending"):
                tags.append(t)
            if len(tags) >= max_tags:
                break
        return tags
    except Exception as exc:
        logger.warning("Ollama hashtag generation failed: %s, using fallback", exc)
        # Fallback: extract frequent meaningful words
        words = re.findall(r"[a-z]{4,}", transcript_text.lower())
        stopwords = {"this", "that", "with", "from", "have", "will", "been", "were", "they", "them", "what", "when", "where", "which", "while", "about", "would", "could", "should", "just", "like", "over", "more", "some", "your", "make", "know", "think", "going", "really", "there"}
        word_counts = {}
        for w in words:
            if w not in stopwords:
                word_counts[w] = word_counts.get(w, 0) + 1
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:max_tags]]
