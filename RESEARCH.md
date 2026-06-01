# AI Clipper Research Report — 2025/2026

## Part 1: TikTok / Shorts Viral Clip Trends

### Dominant Content Formats
- **"Hook-in-2" Format**: The first 2 seconds are now the critical window. Pattern interrupt (text overlay + unexpected visual) followed by immediate value proposition.
- **Photo Carousel Resurgence**: Slideshows on TikTok achieve 2–3× more saves than video because users revisit them as reference material.
- **Split-Screen / Reaction Stacks**: Layering 3–4 reactions in one frame for comedic density.
- **Funnel Clipping**: 60–90s "director's cut" + separate 15s hook clip driving traffic to the longer version.

### Editing & Visual Style Trends
- **Velocity Editing**: Speed-ramping on beat drops remains the most viral technique.
- **Word-by-Word Captions**: Alex Hormozi-style animated captions with color pops on key phrases — now mainstream.
- **Retro VHS / VCR Aesthetic**: Lo-fi tape effects, timestamp overlays, CRT scanlines as counter-aesthetic.
- **AI-Generated B-Roll**: Runway, Kling, Pika used for impossible/expensive b-roll. "AI b-roll + real talking head" is dominant hybrid.
- **Zero-Edit "Raw" Clips**: Over-edited content penalized; raw POV signals authenticity.

### Sound & Music Trends
- **Original Audio > Library Sounds**: Original voiceover now outranks lip-sync. ~1.2–1.5× reach boost from algorithm.
- **ASMR Layer**: Even non-ASMR content layers satisfying ambient sounds (15–30% higher completion).
- **Sped-Up / Slowed+Reverb**: "Nightcore-ification" of music is permanent.

### Strategic & Algorithmic Trends
- **Save Rate is Key Metric**: Algorithm weights saves/shares over likes/comments.
- **Search-Optimized Short Form**: ~40% of discovery for 18–24 via search. Optimize titles and on-screen text.
- **Multi-Platform Cross-Post**: One edit → TikTok, YT Shorts, IG Reels, LinkedIn with slightly different hooks.
- **Comment-Driven Content**: "Follow-up because you asked" is a proven engagement loop.

### 2026 Trajectories
- AI dubbing into 5–10 languages (ElevenLabs) for non-English markets.
- Interactive/clickable Shorts with "choose your path" mechanics.
- Longer-form within short apps (3–10 min "mini-podcast" clips).

---

## Part 2: Modern Creator Tool UI Design Trends

### Dark Mode — The New Default
- **Dark-First Design**: Default for Figma, VS Code, Linear, Arc, Notion. ~80% of devs, ~65% of creators prefer dark mode.
- **Elevated Surface Layering**: Base `#0D0D0F` → card `#1A1A1E` → hover `#252529`. Depth without shadows.
- **Warm Dark Palettes**: Shift from blue-tinted to neutral/warm dark (`#111113`, `#1a1a1c`).
- **OLED Black Option**: `#000000` for OLED displays to reduce battery burn.

### Minimal & Reductionist UI
- **"Boring UI" Movement**: Inspired by Linear/Vercel. Typography, spacing, functional layout over decoration.
- **Command-Palette-First (Cmd+K)**: Keyboard-driven navigation reduces visible menus/toolbars.
- **Contextual Toolbars**: Only relevant actions for current state, not 30-icon permanent sidebars.
- **Whitespace as Structure**: 16–24px minimum padding, hierarchy through spacing not lines.
- **Progressive Disclosure**: Advanced options behind toggles; default view shows 80/20.

### Cyber-Aesthetic / Neobrutalist Fusion
- **Neon Accents on Dark**: `#00FF87` (green), `#8B5CF6` (purple), `#06B6D4` (cyan).
- **Terminal/Console Aesthetics**: Monospace headers, `$>` prompts, CLI syntax in UI.
- **Glitch & Scanline Effects**: Sparingly — onboarding/loading only, not primary workspace.
- **Grid-Based Brutalist Layouts**: Strict 8px grid, visible structural alignment. "Engineered not designed."
- **Flat Solid Colors**: Mesh gradients fading; flat solids with occasional subtle gradient on accents only.

### JetBrains Mono & Monospace Font Trends
- **Adoption**: Now used beyond code editors — terminals, dashboards (Vercel, Supabase, Railway), docs (Tailwind, Astro).
- **Why JetBrains Mono**: Ligatures, distinguishable chars (`0` vs `O`), weight range 100–800, x-height optimized for 14px.
- **Mono-as-Branding**: Used for headings, navigation — signals "technical credibility."
- **Standard Font Pairing**: Inter/Geist (UI) + JetBrains Mono (code/terminal).
- **Variable Font**: Smooth weight transitions in animations.
- **OSS License (OFL)**: Free for commercial use.

### Design Recommendations Summary

| Element | Recommendation |
|---------|---------------|
| Color scheme | Dark-first, warm-neutral base, semantic accents |
| Typography | Inter/Geist for UI, JetBrains Mono for code/status |
| Font sizes | 13–14px mono, 14–15px body, 20–32px headings |
| Spacing | 8px grid, 16px min padding, 24px section gaps |
| Surfaces | 3-layer depth: base → elevated → hover (no shadows) |
| Navigation | Cmd+K palette + contextual toolbar |
| Borders | 1px subtle (`#ffffff12`) or none, use spacing |
| Animations | 150–200ms ease-out interactions, 300ms page transitions |
| Icons | 16–20px stroke (Lucide, Phosphor), 1.5px stroke |
| Status | Dot + color (●), information-dense |
| Border radius | 6–8px cards, 4px inputs, 12–16px modals |
| Data density | High — creators tolerate density if well-structured |

---

## Key Takeaways for AI Clipper

**From viral trends**: The winning formula is raw authenticity + strategic structure. 2-second hook, value-driven, save-worthy, search-optimized. Word-by-word animated captions are now the standard expectation (which AI Clipper already supports — this validates the feature direction).

**From UI trends**: Builder tools that feel like developer tools win. Dark-first, JetBrains Mono for status/monitoring elements, Inter for UI labels, command-driven interaction, and information-dense layouts. The "precision instrument" aesthetic signals technical capability to the creator audience.
