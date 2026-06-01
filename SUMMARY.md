# AI Clipper V2 UI — Summary of Changes

## What Was Added

### 1. `templates/dashboard_v2.html` — New Cyber-Minimal Dashboard UI

A complete, ground-up redesign of the AI Clipper dashboard with a modern cyber-minimal aesthetic.

**Design System:**
- **Dark-first color palette** with 4-layer surface depth (`--bg-base` → `--bg-surface` → `--bg-elevated` → `--bg-hover`)
- **CSS custom properties** for all colors, spacing, typography, shadows, and transitions
- **Google Fonts**: Inter (UI/headings) + JetBrains Mono (monospace/status/code elements)
- **8px grid spacing system** with consistent `--space-*` variables
- **Semantic color system**: accent (indigo), success (emerald), warning (amber), danger (red), info (sky), hot (orange)

**Layout:**
- Sticky header (52px) with brand mark, version badge, and status indicator
- Sticky sidebar (320px) with source input, bulk upload, video info, platform selector, options, and log
- Main content area with card-based layout
- Responsive breakpoints at 900px (tablet) and 600px (mobile)

**Key UI Components:**
- **Platform cards**: Grid of 4 platform options with hover glow effects and selected state with accent border + glow
- **Toggle switches**: Custom-styled with smooth transitions and accent glow when active
- **Progress section**: Large monospace percentage display, gradient progress bar with shimmer effect, pipeline step indicators
- **Clip cards**: 9:16 preview with hover zoom, viral score badges (color-coded), caption boxes with copy button, virality breakdown bars (hook/engage/value/share)
- **Timeline**: Color-coded clip segments on a horizontal track
- **Modals**: Preview modal (video player) and Settings modal (Ollama configuration)
- **Toast notifications**: Slide-in animations with color-coded borders

**Interactions & Animations:**
- Smooth CSS transitions on all interactive elements (150-300ms, custom easing)
- Hover states: border color changes, subtle transforms (translateY), shadow elevation
- Focus-visible outlines for accessibility
- Pulse animations on status dots
- Backdrop blur on header and modals

**Mobile Responsive:**
- Single-column layout below 900px
- Sidebar becomes scrollable horizontal section
- Clips grid adapts from 280px min to single column
- Reduced font sizes and padding

### 2. `RESEARCH.md` — Design & Trend Research

Comprehensive research document covering:
- TikTok/Shorts viral clip trends (2025-2026): hook formats, editing styles, sound trends, algorithmic strategies
- Modern creator tool UI design: dark mode as default, minimal/cyber-aesthetic, JetBrains Mono adoption, font pairing patterns
- Specific design recommendations table for creator tools

### 3. `main.py` — New Route

Added a single new route:
```python
@app.route("/v2")
def dashboard_v2():
    return render_template("dashboard_v2.html")
```

**No other changes to main.py** — all existing routes, functionality, config, and the Flask port remain unchanged.

## What Was NOT Changed

- `core/config.py` — untouched
- `core/analyzer.py` — untouched
- `core/cutter.py` — untouched
- `core/ollama_client.py` — untouched
- `templates/dashboard.html` — original v1 dashboard preserved
- Flask port (still defaults to 7878 via `AICLIPPER_PORT`)
- All API endpoints and backend logic

## How to Use

1. Start AI Clipper as usual: `python main.py`
2. Original dashboard: `http://localhost:7878/`
3. New V2 dashboard: `http://localhost:7878/v2`

## Design Philosophy

The V2 UI follows the "precision instrument" aesthetic — dark, minimal, information-dense, with monospace accents that signal technical capability. This aligns with 2025-2026 creator tool trends where builder tools adopt developer tool aesthetics (Linear, Vercel, Supabase). The design prioritizes:

- **Information density**: Creators tolerate dense UIs if well-structured
- **Semantic color**: Colors convey meaning (status, virality scores, platform selection)
- **Typography hierarchy**: JetBrains Mono for data/status, Inter for labels/UI
- **Progressive disclosure**: Advanced options hidden by default
- **Keyboard-first potential**: Structure supports future Cmd+K command palette
