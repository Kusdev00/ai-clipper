"""Utils - helper functions."""

import os
import shutil


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def format_duration(seconds: float) -> str:
    """Format seconds to MM:SS."""
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"


def format_size(mb: float) -> str:
    """Format size to human-readable."""
    if mb < 1024:
        return f"{mb:.1f} MB"
    return f"{mb/1024:.1f} GB"


def cleanup_old_jobs(dirs: list, max_age_hours: int = 24):
    """Remove directories older than max_age_hours."""
    import time
    now = time.time()
    for base_dir in dirs:
        if not os.path.exists(base_dir):
            continue
        for entry in os.listdir(base_dir):
            path = os.path.join(base_dir, entry)
            if os.path.isdir(path):
                age_hours = (now - os.path.getmtime(path)) / 3600
                if age_hours > max_age_hours:
                    shutil.rmtree(path, ignore_errors=True)
