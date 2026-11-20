# Player UI polish: floating capsule, notices, sharp at 4K

Status: approved design, 2026-11-03. Follows `2026-11-03-player-ui-design.md` (the working Dear ImGui UI).

## Goal

Make the player window's UI look modern and feel responsive, and keep it sharp on a 4K screen. The user
asked for "a really nice ui": a modern look, better feedback, more features and sharpness at 4K. This spec
covers the look, the feedback and the sharpness. The features (seek thumbnails, audio/subtitle tracks,
folder playlist, chapters, in that order) each get their own spec later.

Success: the controls and notices match the approved mockups (floating capsule, centre notice card), text
and icons are crisp at 4K fullscreen, and the UI costs no more than the current one (see Performance).

## Look and behaviour

### Control capsule (bottom centre)

- A rounded capsule (radius = half its height), 78% of the window wide, at most 1100 px at scale 1,
  14 px above the bottom edge, background `rgba(20,22,30,0.72)`.
- Left to right: play/pause icon button; elapsed time; seek bar (takes the free width); total time
  (`--:--` when unknown); volume icon (hover opens a small horizontal slider above it, 0–200%); VSR badge;
  open (folder) icon; fullscreen/window icon.
- VSR badge: violet `#5b4fd6` "VSR 2x" when on, grey "VSR off" when off, not shown when the video has no
  upscaling. Clicking it toggles VSR.
- Seek bar: 4 px track `rgba(255,255,255,0.25)`, played part cyan `#3fd0e0`. On hover: 6 px, a white
  12 px knob at the play position, and a tooltip above the cursor with the time under it. Click or drag;
  the seek happens on release (as today). Disabled look and no tooltip when the duration is unknown.
- Visibility: shown while the mouse moved in the last 2 s, while paused, or while hovered (same rule as
  today). Fades in over 150 ms, out over 300 ms.
- In fullscreen the mouse cursor hides when the capsule is fully hidden, and shows again on movement.

### Notices (centre card)

- A rounded card in the middle of the window: an icon, one line of text and, for volume, a level bar.
- Shown 1 s, then fades out over 300 ms. A new notice replaces the current one at once.
- Events: volume ("Volume 80%" + bar), seek by key ("+5 s · 0:42:18" / "−5 s · …"), pause and play
  (icon only), VSR on/off ("VSR on, 2x" / "VSR off"), and the file name for 2 s when a video opens.
  Seeking with the seek bar shows no notice (the bar already shows it).

### Info overlay and menu

- The `I` info overlay becomes a rounded card in the top-left corner in the same style. Nothing else is
  drawn in the top-left corner.
- The right-click menu gets rounded corners, the theme colours and an icon before each item.

### Scale

- UI scale = window height in logical points (ImGui's DisplaySize.y) / 1080, clamped to 1–2.5; every size
  above is at scale 1. Fonts are rasterised at the scaled size and the framebuffer density (ImGui 1.93
  dynamic fonts), so text stays sharp: 16 px at 1080p, 32 px at 4K fullscreen, and a 200%-scaled 4K
  screen (1920x1080 points) is as sharp as an unscaled one.

## Fonts and icons

- `assets/fonts/Inter-Medium.ttf` (SIL Open Font License) and `assets/fonts/tabler-icons.ttf`
  (MIT), each with its licence file next to it, from pinned releases. Both may ship in the public repo.
- Loaded at start from `SDL_GetBasePath() + "../assets/fonts/"` and merged into one ImGui font (icons
  merged into the Inter font). Only the icon codepoints we use are listed, in `theme.h`.
- Missing or unreadable font files: one line on stderr, and the UI falls back to ImGui's default font
  with the text labels of today. The player never fails to start because of fonts.

## Code structure

- New `player/theme.{h,cpp}`: font loading and merging, colours, sizes, icon codepoints.
- `player/ui.{h,cpp}`:
  - `draw_ui` redrawn as the capsule, notice card, info card and menu, at the frame's scale.
  - `struct UiNotice { const char* icon; std::string text; float level = -1; double shown_at = -1e9; double hold = 1; };`
    in `UiState`; `level < 0` means no bar.
  - Pure helpers: `float fade_alpha(double now, double changed_at, bool visible, double in, double out)`
    and `float notice_alpha(double now, const UiNotice& n)`.
- `player/main.cpp`: posts a notice from the shared action helpers (so keys and buttons behave the same),
  hides the cursor as above, and keeps redrawing while a fade or a notice is running, also when paused.

## Testing

- `--uitest` adds: `fade_alpha` and `notice_alpha` timings (start, middle of fade, after); fonts load and
  merge and the icon glyphs exist (skipped with a note when the assets are not found); headless ImGui
  frames at scale 1 and 2.5 with `dur = 0`, a notice showing, and the capsule hidden. Also run with
  ImGui's asserts on (standalone debug build, as now).
- `tests/smoke.sh vsr.onnx` and `tests/codecs.sh` stay green.

## Performance

- `ui_ms` (from `--stats`) at most 0.3 ms per pass on average with the capsule and a notice showing.
- 4K fullscreen before/after check (1080p -> 3840x2160 with VSR, `--fullscreen`): display stages within
  ~1 ms of before, no more dropped frames.

## Out of scope

Seek thumbnails, audio/subtitle tracks, folder playlist, chapters (separate specs, in that order).
Themes or user-configurable colours.
