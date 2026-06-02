"""
AI Clipper - OpusClip-grade AI short-form content generator.
Runs a Flask server that serves the dashboard and API endpoints.

Usage:
    python main.py                  # Start on default port 7878
    AICLIPPER_PORT=9090 python main.py  # Custom port
"""

import os
import sys
import uuid
import json
import time
import signal
import shutil
import logging
import hashlib
import tempfile
import subprocess
import threading
import traceback
import webbrowser
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, request, jsonify, send_file,
    render_template, Response, stream_with_context,
)

# ── Project Setup ──────────────────────────────────
if os.name == "nt":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from core.config import (
    DOWNLOADS_DIR, CLIPS_DIR, TEMPLATES_DIR,
    STATIC_DIR, PLATFORMS,
)
from core import utils

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ai-clipper")

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)

# In-memory job store (job_id -> job dict)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

def _save_job(job_id: str, job: dict):
    """Persist job status to disk so it survives app crashes."""
    try:
        import json as _json
        _path = os.path.join(DOWNLOADS_DIR, job_id, "job.json")
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        with open(_path, "w", encoding="utf-8") as _f:
            _json.dump(job, _f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

# Ensure required directories exist on startup
for _p in (DOWNLOADS_DIR, CLIPS_DIR):
    os.makedirs(_p, exist_ok=True)


# ════════════════════════════════════════════════════════
# Dashboard
# ════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    return render_template("dashboard_v2.html")


@app.route("/v1")
def dashboard_v1():
    return render_template("dashboard.html")


@app.route("/api/version")
def api_version():
    return jsonify({"version": "2.1", "bulk_upload": True})


# ════════════════════════════════════════════════════════
# API: Load Video Info
# ════════════════════════════════════════════════════════

@app.route("/api/load", methods=["POST"])
def load_video():
    data = request.json or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"detail": "URL is required"}), 400

    # Build a deterministic-ish job id from URL + time so re-loads
    # of the same URL in the same minute share a download.
    ts = datetime.now().strftime("%Y%m%d%H%M")
    job_id = hashlib.md5(url.encode()).hexdigest()[:10]

    try:
        from core.downloader import get_video_info
        info = get_video_info(url)
    except Exception as e:
        log.error("Failed to fetch info for %s: %s", url, e)
        return jsonify({"detail": str(e)}), 500

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "url": url,
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader") or info.get("channel", "Unknown"),
            "thumbnail": info.get("thumbnail", ""),
            "status": "loaded",
            "created": time.time(),
            "steps": {
                "download": "pending",
                "transcribe": "pending",
                "analyze": "pending",
                "cut": "pending",
                "upload": "pending",
            },
            "progress": 0,
            "message": "Video loaded",
            "highlights": [],
            "clips": {},
            "options": {},
        }

    return jsonify({
        "job_id": job_id,
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader") or info.get("channel", "Unknown"),
        "thumbnail": info.get("thumbnail", ""),
    })


# ════════════════════════════════════════════════════════
# API: Run Full Pipeline (single video)
# ════════════════════════════════════════════════════════

@app.route("/api/pipeline/<job_id>", methods=["POST"])
def run_pipeline(job_id: str):
    if job_id not in JOBS:
        return jsonify({"detail": "Job not found"}), 404

    job = JOBS[job_id]
    opts = request.json or {}

    platform = opts.get("platform", "tiffftok")
    platform = "tiktok" if platform == "tiffftok" else platform
    if platform not in PLATFORMS:
        return jsonify({"detail": f"Unknown platform: {platform}"}), 400

    job["options"] = opts

    thread = threading.Thread(
        target=pipeline_worker, args=(job_id, opts), daemon=True,
    )
    thread.start()

    return jsonify({"status": "started", "job_id": job_id})


# ════════════════════════════════════════════════════════
# API: Server-Sent Events – live progress stream
# ════════════════════════════════════════════════════════

@app.route("/api/progress/<job_id>")
def progress_stream(job_id: str):
    if job_id not in JOBS:
        return jsonify({"detail": "Job not found"}), 404

    def event_stream():
        last_status = None
        while True:
            job = JOBS.get(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'error': 'Job removed'})}\n\n"
                break

            payload = {
                "status": job.get("status", "unknown"),
                "progress": job.get("progress", 0),
                "message": job.get("message", ""),
                "steps": job.get("steps", {}),
            }
            if job["status"] == "completed":
                payload["highlights"] = job.get("highlights", [])
                payload["chapters"] = job.get("chapters", [])

            data = json.dumps(payload)
            if data != last_status:
                yield f"event: update\ndata: {data}\n\n"
                last_status = data

            if job["status"] in ("completed", "error"):
                if job["status"] == "error":
                    yield f"event: error\ndata: {json.dumps({'error': job.get('error','')})}\n\n"
                break

            time.sleep(0.8)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ════════════════════════════════════════════════════════
# API: Poll Status (SSE fallback / backward compat)
# ════════════════════════════════════════════════════════

@app.route("/api/status/<job_id>")
def get_status(job_id: str):
    if job_id not in JOBS:
        return jsonify({"detail": "Job not found"}), 404

    job = JOBS[job_id]
    resp = {
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "steps": job["steps"],
    }
    if job["status"] == "completed":
        resp["highlights"] = job.get("highlights", [])
        resp["clips"] = job.get("clips", [])
    elif job["status"] == "error":
        resp["error"] = job.get("error", "Unknown error")

    return jsonify(resp)


# ════════════════════════════════════════════════════════
# API: Download Clip
# ════════════════════════════════════════════════════════

@app.route("/api/clip/<job_id>/<int:clip_index>")
def get_clip(job_id: str, clip_index: int):
    if job_id not in JOBS:
        return jsonify({"detail": "Job not found"}), 404

    clips = JOBS[job_id].get("clips", {})
    # clips is a dict[int, str] — clip_index must be a key
    if isinstance(clips, dict):
        if clip_index not in clips:
            return jsonify({"detail": "Clip not found"}), 404
        clip_path = clips[clip_index]
    else:
        # backwards compat with old list format
        if clip_index >= len(clips):
            return jsonify({"detail": "Clip not found"}), 404
        clip_path = clips[clip_index]

    if not clip_path or not os.path.exists(clip_path):
        return jsonify({"detail": "Clip file not found"}), 404

    download = request.args.get("download", "false").lower() in ("true", "1", "yes")
    return send_file(
        clip_path,
        mimetype="video/mp4",
        as_attachment=download,
        download_name=os.path.basename(clip_path),
    )


# ════════════════════════════════════════════════════════
# API: List Jobs
# ════════════════════════════════════════════════════════

@app.route("/api/jobs")
def list_jobs():
    results = []
    for jid, job in sorted(JOBS.items(), key=lambda x: x[1]["created"], reverse=True):
        results.append({
            "id": jid,
            "title": job["title"],
            "status": job["status"],
            "progress": job["progress"],
            "highlights_count": len(job.get("highlights", [])),
            "created": job["created"],
        })
    return jsonify(results)


# ════════════════════════════════════════════════════════
# API: Batch Processing – submit multiple URLs
# ════════════════════════════════════════════════════════

@app.route("/api/batch", methods=["POST"])
def batch_submit():
    data = request.json or {}
    urls = data.get("urls", [])
    opts = data.get("options", {})

    if not urls:
        return jsonify({"detail": "No URLs provided"}), 400
    if len(urls) > 20:
        return jsonify({"detail": "Max 20 videos per batch"}), 400

    batch_id = hashlib.md5("".join(urls).encode()).hexdigest()[:8]
    job_ids = []

    for url in urls:
        url = url.strip()
        if not url:
            continue

        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
        jid = hashlib.md5((url + ts).encode()).hexdigest()[:10]

        try:
            from core.downloader import get_video_info
            info = get_video_info(url)
        except Exception as e:
            log.warning("Skipping %s: %s", url, e)
            continue

        with JOBS_LOCK:
            JOBS[jid] = {
                "id": jid,
                "url": url,
                "title": info.get("title", "Unknown"),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader") or info.get("channel", "Unknown"),
                "thumbnail": info.get("thumbnail", ""),
                "status": "queued",
                "created": time.time(),
                "steps": {"download": "pending", "transcribe": "pending", "analyze": "pending", "cut": "pending", "upload": "pending"},
                "progress": 0,
                "message": "Queued for batch processing",
                "highlights": [],
                "clips": {},
                "options": opts,
                "batch_id": batch_id,
            }
        job_ids.append(jid)

        # Launch sequentially to avoid overwhelming the machine
        threading.Thread(
            target=_batch_worker, args=(jid, opts), daemon=True,
        ).start()

    return jsonify({"batch_id": batch_id, "job_ids": job_ids, "count": len(job_ids)})


@app.route("/api/batch/<batch_id>")
def batch_status(batch_id: str):
    jobs = [j for j in JOBS.values() if j.get("batch_id") == batch_id]
    if not jobs:
        return jsonify({"detail": "Batch not found"}), 404

    total = len(jobs)
    done = sum(1 for j in jobs if j["status"] == "completed")
    errors = sum(1 for j in jobs if j["status"] == "error")
    in_progress = total - done - errors

    return jsonify({
        "batch_id": batch_id,
        "total": total,
        "completed": done,
        "errors": errors,
        "in_progress": in_progress,
        "all_done": in_progress == 0,
    })


def _batch_worker(job_id: str, opts: dict):
    """Wait for the job to reach loaded state, then run pipeline."""
    # Small delay between batch starts to avoid resource contention
    time.sleep(2)
    pipeline_worker(job_id, opts)


# ════════════════════════════════════════════════════════
# API: Delete / Cleanup Job
# ════════════════════════════════════════════════════════

@app.route("/api/job/<job_id>", methods=["DELETE"])
def delete_job(job_id: str):
    if job_id not in JOBS:
        return jsonify({"detail": "Job not found"}), 404

    job = JOBS.pop(job_id)

    # Clean up files
    try:
        dl_dir = os.path.join(DOWNLOADS_DIR, job_id)
        cl_dir = os.path.join(CLIPS_DIR, job_id)
        shutil.rmtree(dl_dir, ignore_errors=True)
        shutil.rmtree(cl_dir, ignore_errors=True)
    except Exception:
        pass

    return jsonify({"deleted": job_id})


# ════════════════════════════════════════════════════════
# API: Health
# ════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "2.0.0"})


# ════════════════════════════════════════════════════════
# Ollama Proxy (avoids CORS issues from browser)
# ════════════════════════════════════════════════════════

@app.route("/api/ollama/status")
def ollama_status():
    import requests as _req
    url = request.args.get("url", "http://localhost:11434").rstrip("/")
    try:
        r = _req.get(url + "/api/tags", timeout=5)
        if r.status_code != 200:
            return jsonify({"ok": False, "error": "HTTP " + str(r.status_code)}), 502
        d = r.json()
        models = [m["name"] for m in d.get("models", [])]
        model = request.args.get("model", "llama3.1:8b")
        has_model = any(m.startswith(model.split(":")[0]) for m in models)
        return jsonify({"ok": True, "models": models, "model_found": has_model, "model": model})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/ollama/test", methods=["POST"])
def ollama_test():
    import requests as _req
    body = request.get_json(silent=True) or {}
    url = (body.get("url", "http://localhost:11434")).rstrip("/")
    model = body.get("model", "llama3.1:8b")
    try:
        r = _req.post(url + "/api/generate", json={"model": model, "prompt": "Reply with OK.", "stream": False, "options": {"num_predict": 5}}, timeout=15)
        if r.status_code == 200:
            return jsonify({"ok": True, "model": model})
        return jsonify({"ok": False, "error": "HTTP " + str(r.status_code)}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


# ════════════════════════════════════════════════════════
# Bulk Upload API
# ════════════════════════════════════════════════════════

@app.route("/api/bulk_upload", methods=["POST"])
def bulk_upload():
    """Upload a video file directly (no URL download needed)."""
    if "file" not in request.files:
        return jsonify({"detail": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"detail": "No file selected"}), 400

    # Create job
    job_id = f"bulk_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    save_dir = os.path.join(DOWNLOADS_DIR, job_id)
    os.makedirs(save_dir, exist_ok=True)

    # Save file
    safe_name = Path(file.filename).stem[:80] + Path(file.filename).suffix
    save_path = os.path.join(save_dir, "source" + Path(safe_name).suffix)
    file.save(save_path)

    # Get video info using ffprobe
    duration = 0
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", save_path],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace"
        )
        if r.returncode == 0:
            duration = float(r.stdout.strip())
    except Exception:
        pass

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "url": f"file://{safe_name}",
            "title": safe_name,
            "duration": duration,
            "uploader": "local",
            "thumbnail": "",
            "status": "loaded",
            "source_file": save_path,
            "created": time.time(),
            "message": "Video loaded — ready to process",
            "progress": 0,
            "highlights": [],
            "chapters": [],
            "clips": {},
            "options": {},
            "steps": {
                "download": "done",
                "transcribe": "pending",
                "analyze": "pending",
                "cut": "pending",
                "upload": "pending",
            },
        }

    log.info("Bulk upload: %s → job %s (%.0fs)", safe_name, job_id, duration)
    return jsonify({"job_id": job_id, "title": safe_name, "duration": duration})


@app.route("/api/start_job/<job_id>", methods=["POST"])
def start_job(job_id):
    """Start processing a job (transcribe → analyze → cut)."""
    if job_id not in JOBS:
        return jsonify({"detail": "Job not found"}), 404

    job = JOBS[job_id]
    if job["status"] not in ("loaded", "created"):
        return jsonify({"detail": f"Job already {job['status']}"}), 400

    job["status"] = "running"
    job["progress"] = 5
    job["message"] = "Starting pipeline…"

    # Get options from request or use defaults
    data = request.json or {}
    options = {
        "num_clips": data.get("num_clips", 5),
        "captions": data.get("captions", True),
        "caption_style": data.get("caption_style", "default"),
        "face_crop": data.get("face_crop", True),
        "whisper_model": data.get("whisper_model", "base"),
        "use_ollama": data.get("use_ollama", False),
        "ollama_url": data.get("ollama_url", "http://localhost:11434"),
        "ollama_model": data.get("ollama_model", "llama3.1:8b"),
        "brand_template": data.get("brand_template", "none"),
    }

    thread = threading.Thread(
        target=pipeline_worker,
        args=(job_id, options),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "started", "job_id": job_id})


# ════════════════════════════════════════════════════════
# Regenerate Caption API
# ════════════════════════════════════════════════════════

@app.route("/api/regenerate_caption/<job_id>/<int:clip_index>", methods=["POST"])
def regenerate_caption(job_id, clip_index):
    """Regenerate caption + hashtags for a specific clip."""
    if job_id not in JOBS:
        return jsonify({"detail": "Job not found"}), 404

    job = JOBS[job_id]
    highlights = job.get("highlights", [])
    if clip_index >= len(highlights):
        return jsonify({"detail": "Clip not found"}), 404

    h = highlights[clip_index]
    text = h.get("text", "")
    if not text:
        return jsonify({"detail": "No transcript for this clip"}), 400

    from core.ollama_client import generate_tiktok_caption, generate_hashtags

    caption = generate_tiktok_caption(text, job.get("title", ""))
    tags = generate_hashtags(text, job.get("title", ""))

    h["caption"] = caption
    h["hashtags"] = tags

    log.info("Regenerated caption %d: %s [%s]", clip_index, caption[:60], ", ".join(tags))
    return jsonify({"caption": caption, "hashtags": tags})


@app.route("/api/regenerate_hashtags/<job_id>/<int:clip_index>", methods=["POST"])
def regenerate_hashtags(job_id, clip_index):
    """Regenerate only hashtags for a specific clip (no caption)."""
    if job_id not in JOBS:
        return jsonify({"detail": "Job not found"}), 404

    job = JOBS[job_id]
    highlights = job.get("highlights", [])
    if clip_index >= len(highlights):
        return jsonify({"detail": "Clip not found"}), 404

    h = highlights[clip_index]
    text = h.get("text", "")
    if not text:
        return jsonify({"detail": "No transcript for this clip"}), 400

    from core.ollama_client import generate_hashtags

    tags = generate_hashtags(text, job.get("title", ""))
    h["hashtags"] = tags

    log.info("Regenerated hashtags %d: [%s]", clip_index, ", ".join(tags))
    return jsonify({"hashtags": tags})


# ════════════════════════════════════════════════════════
# Pipeline Worker (Background Thread)
# ════════════════════════════════════════════════════════

def _fallback_highlights(transcriptions, scenes, duration, num_clips):
    """Simple fallback: pick evenly-spaced clips from transcript segments."""
    highlights = []
    if transcriptions:
        # Pick top segments by length (longer = more content)
        sorted_segs = sorted(transcriptions, key=lambda s: len(s.get("text", "")), reverse=True)
        for i, seg in enumerate(sorted_segs[:num_clips]):
            st = max(0, seg["start"] - 2)
            et = min(duration, seg["end"] + 5)
            if et - st < 5:
                et = st + 5
            highlights.append({
                "id": i,
                "start": round(st, 2),
                "end": round(et, 2),
                "viral_score": max(20, 50 - i * 5),
                "hook_score": 0.0,
                "reason": f"Top segment #{i+1}",
                "text": seg.get("text", "")[:100],
                "audio_features": {},
                "speaker_changes": 0,
                "scene_changes": 0,
            })
    elif scenes:
        for i, sc in enumerate(scenes[:num_clips]):
            st = max(0, sc["time"] - 3)
            et = min(duration, sc["time"] + 47)
            highlights.append({
                "id": i,
                "start": round(st, 2),
                "end": round(et, 2),
                "viral_score": 30,
                "hook_score": 0.0,
                "reason": f"Scene change at {sc['time']:.1f}s",
                "text": "",
                "audio_features": {},
                "speaker_changes": 0,
                "scene_changes": 1,
            })
    return highlights


def pipeline_worker(job_id: str, options: dict):
    """Run the full clipping pipeline for a single job."""
    job = JOBS.get(job_id)
    if not job:
        return

    platform = options.get("platform", "tiktok")
    add_captions = options.get("captions", True)
    smart_crop = options.get("smart_crop", False)
    face_track = options.get("face_track", True)
    num_clips = min(max(options.get("num_clips", 5), 1), 20)
    caption_style = options.get("caption_style", "default")
    brand_template = options.get("brand_template", "none")
    whisper_model = options.get("whisper_model", "base")
    use_ollama = options.get("use_ollama", True)
    ollama_model = options.get("ollama_model", "llama3.1:8b")
    ollama_url = options.get("ollama_url", "http://localhost:11434")

    if use_ollama:
        os.environ["OLLAMA_MODEL"] = ollama_model
        os.environ["OLLAMA_URL"] = ollama_url
        # Keep Ollama model loaded — ping before pipeline starts to avoid
        # "model unloaded" issues between bulk jobs
        try:
            from core.ollama_client import keepalive, is_available
            if is_available():
                keepalive()
                log.info("Ollama keepalive ping sent — model kept loaded")
            else:
                log.warning("Ollama not reachable at pipeline start")
        except Exception as exc:
            log.warning("Ollama keepalive failed: %s", exc)

    source_path = ""

    try:
        # ── Step 1: Download / Locate Source ────────────
        job["status"] = "running"
        _save_job(job_id, job)

        # Bulk uploads already have the file — skip download
        is_bulk = job_id.startswith("bulk_") and job.get("source_file")
        if is_bulk:
            source_path = job["source_file"]
            job["steps"]["download"] = "done"
            job["progress"] = 30
            job["message"] = "File uploaded — extracting audio …"
        else:
            job["steps"]["download"] = "active"
            job["message"] = "Downloading video …"
            job["progress"] = 5

            from core.downloader import download_video, download_audio

            def on_dl_progress(pct):
                job["progress"] = 5 + int(pct * 0.25)  # 5-30

            source_path = download_video(job["url"], job_id, progress_callback=on_dl_progress)

            job["steps"]["download"] = "done"
            job["progress"] = 30
            job["message"] = "Download complete"

        # ── Step 2: Whisper Transcribe ────────────────
        job["steps"]["transcribe"] = "active"
        job["message"] = "Transcribing with Whisper … (may take a while)"
        job["progress"] = 32

        transcriptions = []
        audio_data = {}

        try:
            if is_bulk:
                # Extract audio from uploaded file via ffmpeg
                audio_path = os.path.join(DOWNLOADS_DIR, job_id, "audio.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", source_path, "-vn", "-acodec", "pcm_s16le",
                     "-ar", "16000", "-ac", "1", audio_path],
                    capture_output=True, text=True, timeout=300,
                    encoding="utf-8", errors="replace"
                )
            else:
                download_audio(job["url"], job_id)
        except Exception as exc:
            log.warning("Audio extraction failed: %s", exc)

        try:
            from core.analyzer import transcribe_audio, analyze_audio, get_duration
            transcriptions = transcribe_audio(job_id, model_size=whisper_model)
            job["message"] = f"Transcribed {len(transcriptions)} segments – analysing audio …"
        except Exception as exc:
            log.warning("Transcription failed: %s", exc)

        # Audio analysis (librosa)
        try:
            audio_data = analyze_audio(job_id)
            if audio_data:
                job["message"] = (
                    f"Transcribed {len(transcriptions)} segments – "
                    f"found {len(audio_data.get('excitement_moments', []))} exciting moments"
                )
        except Exception as exc:
            log.warning("Audio analysis failed: %s", exc)

        job["steps"]["transcribe"] = "done"
        job["progress"] = 55

        # ── Step 3: AI Analysis ───────────────────────
        job["steps"]["analyze"] = "active"
        job["message"] = "Running AI highlight detection …"
        job["progress"] = 60

        from core.analyzer import detect_scenes, find_highlights

        duration = get_duration(source_path)

        scenes = detect_scenes(source_path, threshold=0.35) if duration < 3600 else []

        log.info(f"Finding highlights: {len(transcriptions)} transcripts, {len(scenes)} scenes, duration={duration:.1f}s")

        try:
            highlights = find_highlights(
                transcriptions=transcriptions,
                scenes=scenes,
                audio_data=audio_data,
                duration=duration,
                num_clips=num_clips,
            )
        except Exception as e:
            log.error(f"find_highlights failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use simple segment-based highlights
            log.warning("Using fallback highlight detection")
            highlights = _fallback_highlights(transcriptions, scenes, duration, num_clips)

        highlights = highlights[:num_clips]

        # ── Step 3b: Chapter Detection & Hot Take Scoring ──
        chapters: list[dict] = []
        use_ollama = options.get("use_ollama", True)

        if transcriptions:
            try:
                from core.analyzer import detect_chapters, score_hot_takes

                job["message"] = "Detecting chapters…"
                chapters = detect_chapters(
                    transcript=transcriptions,
                    duration=duration,
                    use_ollama=use_ollama,
                )
                if chapters:
                    job["chapters"] = chapters
                    log.info("Detected %d chapters", len(chapters))
                    # Assign chapter title to each highlight
                    for h in highlights:
                        h_start = h.get("start", 0)
                        for ch in chapters:
                            if ch["start"] <= h_start < ch["end"]:
                                h["chapter_title"] = ch["title"]
                                h["chapter_summary"] = ch.get("summary", "")
                                break

                job["message"] = "Scoring hot takes…"
                highlights = score_hot_takes(
                    clips=highlights,
                    use_ollama=use_ollama,
                )
                # Re-sort by combined score (viral_score + hot_take bonus)
                for h in highlights:
                    h["combined_score"] = h.get("viral_score", 0) + h.get("hot_take_score", 0) * 5
                highlights.sort(key=lambda h: h.get("combined_score", 0), reverse=True)
                highlights = highlights[:num_clips]

            except Exception as e:
                log.warning("Ollama analysis failed: %s", e)

        job["highlights"] = highlights
        job["steps"]["analyze"] = "done"
        job["progress"] = 72
        job["message"] = f"Found {len(highlights)} highlights – generating captions …"

        # ── Step 3b: Generate AI captions for each clip ──
        caption_mode = options.get("caption_mode", "full")  # "full" = caption + hashtags, "hashtags_only" = skip caption
        try:
            from core.ollama_client import generate_tiktok_caption, generate_hashtags, is_available as ollama_available
            if use_ollama and ollama_available():
                for i, h in enumerate(highlights):
                    text = h.get("text", "")
                    if text:
                        if caption_mode == "hashtags_only":
                            h["caption"] = ""
                            tags = generate_hashtags(text, job.get("title", ""))
                            h["hashtags"] = tags
                            log.info("Clip %d: hashtags only [%s]", i+1, ", ".join(tags))
                        else:
                            caption = generate_tiktok_caption(text, job.get("title", ""))
                            h["caption"] = caption
                            tags = generate_hashtags(text, job.get("title", ""))
                            h["hashtags"] = tags
                            log.info("Clip %d: %s [%s]", i+1, caption[:60], ", ".join(tags))
                    else:
                        h["caption"] = ""
                        h["hashtags"] = []
                if caption_mode == "hashtags_only":
                    job["message"] = f"Generated hashtags for {len(highlights)} clips"
                else:
                    job["message"] = f"Generated captions + hashtags for {len(highlights)} clips"
            else:
                log.info("Ollama not available, using transcript text as caption")
                for h in highlights:
                    if caption_mode == "hashtags_only":
                        h["caption"] = ""
                    else:
                        h["caption"] = (h.get("text", "") or "")[:200]
                    h["hashtags"] = generate_hashtags(h.get("text", "") or "", job.get("title", ""))
        except Exception as exc:
            log.warning("Caption/hashtag generation failed: %s", exc)
            for h in highlights:
                if caption_mode == "hashtags_only":
                    h["caption"] = ""
                else:
                    h["caption"] = (h.get("text", "") or "")[:200]
                h["hashtags"] = []

        # Filter out highlights that are too short — minimum 3s for short videos, 5s for longer
        min_dur = 3.0 if duration < 30 else 5.0
        log.info("Filtering highlights: %d before, min_dur=%.1fs", len(highlights), min_dur)
        for h in highlights:
            dur = h.get("end", 0) - h.get("start", 0)
            log.info("  highlight %d: %.1fs - %.1fs (%.1fs)", h.get("id", -1), h.get("start", 0), h.get("end", 0), dur)
        highlights = [h for h in highlights if (h.get("end", 0) - h.get("start", 0)) >= min_dur]
        highlights = highlights[:num_clips]
        log.info("Filtering highlights: %d after", len(highlights))

        job["message"] = f"Found {len(highlights)} highlights – generating clips …"
        _save_job(job_id, job)

        # ── Step 4: Cut Clips ─────────────────────────
        job["steps"]["cut"] = "active"
        _save_job(job_id, job)
        job["progress"] = 73

        from core.cutter import cut_clip, build_caption_words, get_brand_template

        clips: dict[int, str] = {}
        templates = get_brand_template(brand_template) if brand_template != "none" else None

        for i, h in enumerate(highlights):
            caption_words = None
            if add_captions and transcriptions:
                caption_words = build_caption_words(transcriptions, h["start"], h["end"])

            def on_clip_prog(pct, idx=i, total=len(highlights)):
                overall = 73 + int((idx + pct / 100) / total * 26)
                job["progress"] = min(overall, 99)

            try:
                log.info("cut_clip: starting clip %d/%d (%.1fs @ %s)",
                         i + 1, len(highlights), h["end"] - h["start"],
                         options.get("crop_mode", "blur_bg"))
                clip_path = cut_clip(
                    source_path=source_path,
                    job_id=job_id,
                    clip_id=i,
                    start=h["start"],
                    end=h["end"],
                    platform=platform,
                    captions=caption_words if (add_captions and caption_words) else None,
                    caption_style=caption_style,
                    brand_template=templates,
                    face_track=face_track and smart_crop,
                    progress_callback=on_clip_prog if len(highlights) > 1 else None,
                    crop_mode=options.get("crop_mode", "blur_bg"),
                )
                clips[i] = clip_path
                job["message"] = f"Generated clip {i + 1} / {len(highlights)}"
                log.info("Clip %d/%d done: %s", i + 1, len(highlights), clip_path)
            except subprocess.TimeoutExpired:
                log.error("Clip %d timed out after 600s", i + 1)
                job["message"] = f"Clip {i + 1} timed out"
            except Exception as exc:
                log.error("Clip %d failed: %s", i + 1, exc, exc_info=True)
                job["message"] = f"Clip {i + 1} failed: {exc}"
                # Skip this clip — clips dict won't have this index, get_clip returns 404

        job["clips"] = clips

        # Build final highlights list: one entry per successfully-cut clip,
        # with video_url and clip_index. Skip highlights whose clip failed.
        final_highlights = []
        for h_idx, h in enumerate(highlights):
            if h_idx in clips:
                h["video_url"] = f"/api/clip/{job_id}/{h_idx}"
                h["clip_index"] = h_idx
                final_highlights.append(h)
            # else: clip failed — don't include it at all

        job["highlights"] = final_highlights  # replace with only valid clips

        job["steps"]["cut"] = "done"
        job["progress"] = 100
        job["status"] = "completed"
        job["message"] = f"Done – {len(final_highlights)} clips ready!"
        _save_job(job_id, job)

    except Exception as exc:
        import traceback as _tb
        log.error("Pipeline error for job %s: %s", job_id, exc)
        _tb.print_exc()
        job["status"] = "error"
        job["error"] = str(exc)
        job["message"] = f"Error: {exc}"
        for step_name, step_status in job.get("steps", {}).items():
            if step_status == "active":
                job["steps"][step_name] = "error"
                break
        _save_job(job_id, job)


# ════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════

def main():
    port = int(os.environ.get("AICLIPPER_PORT", 7878))
    host = os.environ.get("AICLIPPER_HOST", "127.0.0.1")

    # Check ffmpeg/ffprobe are available
    import shutil
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"\n❌ ERROR: '{tool}' not found in PATH.")
            print("AI Clipper requires ffmpeg and ffprobe to be installed.")
            print("Download from: https://ffmpeg.org/download.html")
            print("Or install via: winget install ffmpeg")
            print("Make sure ffmpeg.exe is in your system PATH.\n")
            sys.exit(1)

    utils.ensure_dirs(DOWNLOADS_DIR, CLIPS_DIR)

    # Check Ollama availability (non-blocking)
    from core.ollama_client import is_available as ollama_available, ensure_ollama
    if ollama_available():
        print("✅ Ollama is running — AI captions enabled")
    elif ensure_ollama():
        print("✅ Ollama started — AI captions enabled")
    else:
        print("⚠️  Ollama unavailable — captions will use transcript fallback")

    # Check if running on Windows
    if os.name == "nt":
        print("✅ Running on Windows")
    else:
        print("⚠️  Running on Linux/WSL")

    url = f"http://{host}:{port}"
    print()
    print("=" * 55)
    print(f"  ✂  AI Clipper  v2.0  –  {url}")
    print("=" * 55)
    print()

    # Graceful shutdown
    def _shutdown(signum, frame):
        log.info("Shutting down …")
        # Optionally purge downloads here
        os._exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Open browser after brief delay
    def _open():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()

    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
