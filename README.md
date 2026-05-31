# AI Clipper

Paste a video URL or upload files → AI finds the best moments → cuts clips with captions & hashtags → you upload them to TikTok/YouTube.

## What it does

Paste a URL (YouTube, TikTok, etc.) or upload video files directly. AI Clipper:

1. **Downloads** the video (or uses your uploaded file)
2. **Transcribes** speech with faster-whisper
3. **Analyzes** audio + video for exciting moments (scene changes, laughter, hooks, value, shareability)
4. **Scores** each segment for viral potential
5. **Cuts** the top moments into clips (vertical 9:16 for TikTok/Reels/Shorts)
6. **Generates** AI captions & hashtags via Ollama with llama3.1:8b

Then you download the clips, copy the captions, and upload them to TikTok Studio / YouTube Shorts.

## Requirements

- **Python 3.11+**
- **Ollama** with `llama3.1:8b` — for caption generation (set `OLLAMA_MODEL` env var to use a different model)
- **Google Chrome** — used for TikTok login (optional, only if you want auto-upload)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Kusdev00/ai-clipper.git
cd ai-clipper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

Then open **http://127.0.0.1:7878** in your browser.

## Bulk Upload

On the Upload tab, you can select multiple video files at once. They'll be processed one-by-one — get AI-scored clip suggestions for each, then download them all.

## Tech Stack

| Component | Tool |
|-----------|------|
| Backend | Flask |
| Transcription | faster-whisper |
| Audio analysis | librosa |
| AI captions | Ollama |
| Video cutting | FFmpeg |
| Dashboard | Vanilla JS + HTML |

## Config

Edit `config.yaml` to adjust:
- Whisper model size (tiny → large-v3)
- Viral score weights
- Output platform (TikTok/YouTube Shorts)
- Face tracking toggle

## How Viral Scoring Works

Each clip candidate is scored 0–100 on:

| Factor | Weight | How it's measured |
|--------|--------|-------------------|
| Hook quality | 15% | Opening words, questions, pattern interrupts |
| Excitement words | 20% | NLP keyword matching |
| Audio energy | 20% | RMS loudness, laughter detection |
| Speech density | 15% | Words per minute |
| Scene dynamics | 10% | Camera cuts, visual activity |
| Caps/shouting | 10% | ALL-CAPS detection |
| Questions/curiosity | 10% | "What if", "How to", cliffhangers |

## License

[MIT](LICENSE)
