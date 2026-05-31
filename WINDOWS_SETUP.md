# AI Clipper - Windows Setup

## Prerequisites
1. Install **Google Chrome** from https://www.google.com/chrome/
2. Install Python 3.12+ from python.org (check "Add to PATH")

## Setup (run in PowerShell or CMD)

```
cd C:\Users\ojala\Downloads\ai-clipper

# Create Windows virtual environment
python -m venv venv-win

# Activate it
venv-win\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

## TikTok Login (One Time)
1. Open http://localhost:7878 in your browser
2. Click "Connect TikTok"
3. A **Google Chrome** window opens with TikTok login
4. Log in with QR code or email/phone
5. Session is saved to `tiktok_state.json` — you never log in again

## Important Rules
- **NEVER delete `tiktok_state.json`** — contains your saved session
- **NEVER delete `C:\TikTokProfile`** — contains your Chrome profile data
- If login breaks, delete both and redo login once
- Close Chrome before clicking "Connect TikTok" if you get profile lock errors
