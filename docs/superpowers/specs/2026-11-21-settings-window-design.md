# Player settings window

Date: 2026-11-21. Status: design agreed in chat; awaiting the user's review of this document.
Follows `2026-11-03-player-ui-design.md` (the ImGui UI) and `2026-11-03-player-ui-polish-design.md` (its look).

## Intent

The user wants a settings window in the player that matches the existing UI design (the floating capsule,
centre-card notices, Inter + Tabler icons, custom drawing). Earlier they asked for switches between GPU and
CPU paths for decoding, upscaling and display "in the settings menu", without CPU upscaling. In this
conversation they added: **some settings should change on the fly**, and **the display section should report
the resolution of every connected monitor** (on their PC: 1 = PG27UCDM 3840x2160 @ 240 Hz, primary;
2 = LG ULTRAGEAR 2560x1440 @ 60 Hz).

Today the player has no settings: everything is a command-line flag (`--no-rtx`, `--no-vsr`, `--screen N`,
`--fullscreen`), and only recent files are saved.

**Success:** every setting below can be changed with the mouse from one card that looks like the rest of the
UI; the live ones take effect without interrupting playback; the others reopen the video at the same position;
choices survive a restart of the player.

## Goals, non-goals, anti-goals

- Goals: the settings in the table below; live where marked; saved between runs; monitors listed with real
  data and kept current when one is plugged in or out.
- Non-goals: CPU upscaling (the user ruled it out); rebuilding the decoder or display in-process (approach 3,
  later if the restart flash bothers the user); keyboard-shortcut editing; skins; per-video settings.
- Anti-goals: a window that looks like a stock ImGui tool panel (square checkboxes, default combos); any
  setting change that drops frames or stalls playback while the card is merely open; a settings file that can
  stop the player from starting.

## Approaches considered

1. **Custom-drawn card with the capsule's own drawing helpers** (chosen by the user). Matches the design
   exactly and stays sharp at 4K; more drawing code.
2. ImGui's stock widgets in a themed window: about half the code, but only roughly matches the look.
3. Everything live, rebuilding decoder/upscaler/display mid-playback: no restart flash, but a large risky
   refactor of `main.cpp`'s single setup block. Kept for later.

The user also asked whether to switch UI systems (RmlUi, Qt Quick); decided: stay on Dear ImGui.

## The settings

| Section | Setting | Values | Applies | Default |
|---|---|---|---|---|
| Upscaling | Upscaler | NVIDIA RTX / Ours / Off (segmented) | live | NVIDIA RTX where available, else Ours |
| Decoding | Decoder | GPU (NVDEC) / CPU (segmented) | reopens the video at the same position | GPU |
| Display | Screen | one row per monitor (radio rows) | reopens the video at the same position on that monitor | primary |
| Display | Start fullscreen | switch | next video opened | off |
| Playback & controls | Remember volume | switch | live (volume is saved as it changes) | on |
| Playback & controls | Hide controls after | 1 s / 2 s / 3 s / 5 s (segmented) | live | 2 s |
| Playback & controls | Show info overlay at start | switch | next video opened | off |

Builder's call recorded here: the sketch in chat said "Default volume" (a slider); this document makes it
**Remember volume** instead, because the capsule already has a volume slider and the README lists "volume
resets" when another video opens as a known limit. Remembering the last volume fixes that limit; a second
volume slider would duplicate the capsule's.

### Upscaler (live)

- Options shown only when they exist: "NVIDIA RTX" only in builds with `HAVE_RTX_VSR`; when the GPU/driver
  lacks RTX VSR it is shown disabled with a tooltip "not available on this GPU or driver". "Ours" disabled with
  a tooltip when no model was loaded (`have_vsr` false: no `vsr.onnx`, or the video already fills the screen).
  When neither upscaler exists the section shows one line saying why (the same reason the log prints). RTX VSR
  needs our model too (see call 1 below).
- Switching is a flag flip in the prepare stage: our model's engine is already open even when RTX VSR upscales
  (`main.cpp`, `vsr.open` runs first). RTX VSR is opened on first selection when the player started without it
  (`--no-rtx` or the setting); if opening fails, the option turns disabled and a notice says so.
- RTX VSR takes the full-size frame, ours the shrunk one when the screen plan shrinks (`shrink`); the switch
  must keep that split as the existing failure fallback does (`main.cpp` around the `rtx.run` call).
- "Off" is the same state as the `V` key and the capsule's VSR badge; all three stay in sync.
- The info overlay's upscaler line (`vsr_desc`, today a `const` computed once) is recomputed on a switch.
- `--collect` keeps collecting only while RTX VSR is the upscaler (as today).
- The next frame after a switch resets the recurrent model's history, as after a seek.

### Decoder (reopens)

- `Media::open` gains a way to skip NVDEC (`av_hwdevice_ctx_create`), so decoding happens on the CPU through
  the existing software path ("NVDEC unavailable, decoding on the CPU"). Upscaling stays on the GPU.
- GPU is the default; when NVDEC isn't available the GPU option is disabled with a tooltip.

### Screen (reopens)

- One row per connected monitor: number (the `--screen N` number), name from `SDL_GetDisplayName`,
  resolution in physical pixels (mode size x pixel density, as the upscaling plan uses), refresh rate, and tags
  "primary" and "playing here" (the monitor the window is on now, `SDL_GetDisplayForWindow`). Same data
  `--check` prints today.
- The list updates on SDL display added/removed/moved events while the card is open.
- Selecting a monitor reopens there. Dragging the window to another monitor only moves the "playing here" tag;
  it does not replan upscaling.
- A saved screen that no longer exists falls back to the primary (as `--screen` does today).

### Reopening

- Uses the existing restart path (`open_video` → `restart_args` → `_execv`/`execv`) with the same video and a
  new `--start T` flag: after opening, the player seeks to T before showing the first frame. Paused state is
  kept (`--paused`, also new, or folded into the same mechanism: builder's choice).
- A notice ("Reopening on screen 2…" / "Switching to CPU decoding…") shows before the restart.
- The saved settings are written before the restart, so the new process reads them.

## The card

- Centre card in the notices' style (`rgba(20,22,30,…)`, rounded, theme colours, Inter, Tabler icons), sized
  to its content, at most 80% of the window height, scrolling inside if needed (small windows, scale 1).
  Section headings with an icon; segmented buttons with the selected one in the accent colour (violet
  `#5b4fd6`, as the VSR badge); switches drawn like the capsule's controls; cyan `#3fd0e0` for focus/hover
  where the capsule uses it.
- Opens from: a gear icon in the capsule (between the VSR badge and the open icon), a "Settings" item in the
  right-click menu, and the `S` key. Closes with its ✕, `Esc` (before Esc leaves fullscreen or quits), `S`
  again, or a click outside the card.
- The video keeps playing behind it; the capsule stays visible while the card is open. Keys typed while the
  card has focus go to the card (the existing `ui_has_keyboard` rule).
- Scales with the UI scale like the rest (sharp at 4K fullscreen).

## Saving

- `settings.txt` next to `recent.txt` (`$XDG_CONFIG_HOME/open-player/`, `%APPDATA%\open-player\` on Windows,
  as `recent_file()` resolves), one `key=value` per line. Unknown keys and bad values are ignored and default;
  an unreadable or missing file means all defaults; write failures are ignored (one line on stderr). Written
  atomically (temp file + rename), like the pair files.
- Command-line flags win over saved settings for that run and are not saved (e.g. `--no-rtx` picks Ours for
  this run without changing the saved choice). A change made in the card is saved.
- Parsing and defaults live in a new `player/settings.{h,cpp}` as pure functions.

## Code structure (proposed; the builder may refine)

- `player/settings.{h,cpp}`: `struct Settings`, load/save, parse/format.
- `player/ui.{h,cpp}`: the card, drawn with the existing helpers (`icon_button`, `track`, `text_centred`,
  `col`); new `UiAction` kinds for each change; `UiState` gains the settings, the monitor list and which
  upscalers/decoders are available.
- `player/main.cpp`: applies live actions; restart actions go through `open_video` with `--start`; reads
  settings at start (after flags); recomputes `vsr_desc`; lazy `RtxVsr` open.
- `player/media.{h,cpp}`: the no-NVDEC option.
- `player/theme.h`: any new icon codepoints (gear, monitor, switch states).

## Testing

- `--uitest`: settings parse/format round trip, defaults, unknown keys and bad values, flag precedence;
  headless ImGui frames with the card open at scale 1 and 2.5, with 0, 1 and 3 monitors, and with each
  upscaler unavailable.
- `tests/smoke.sh` and `tests/codecs.sh` stay green (they must not be affected by a saved `settings.txt`:
  they set `XDG_CONFIG_HOME` to a temp folder, as they do for recent files).
- Speed: with the card open over 1080p -> 4K playback, `ui_ms` within the polish spec's budget (0.3 ms per
  pass average) and no extra dropped frames; switching upscaler live drops at most one frame.
- By hand, by the user: the look against the capsule, live switching, reopening on the other monitor and with
  the CPU decoder, plugging a monitor in and out with the card open. Automated GUI input on the user's desktop
  is not allowed (it once opened one of their files).

## Calls made after the builder check (the user may overrule)

1. **RTX VSR still needs our model.** Today RTX VSR is sized from our model's factor and screen plan
   (`main.cpp`, "shortcut: ... so it needs the model too"). That stays: with no `vsr.onnx`, or when the video
   already fills the screen, both upscalers are unavailable and the section shows one line saying why. RTX
   VSR without our model is a separate change.
2. **`V` and the capsule badge are a quick compare, not a setting.** They toggle upscaling for this run only
   and don't touch `settings.txt`; turning it back on restores the upscaler chosen in the card. Choosing "Off"
   in the card is saved. In the card, the Upscaler row shows "Off" whenever upscaling is off, whichever way.
3. **The saved screen is the monitor's identity**, `screen=<name> <width>x<height>`. At start it matches by
   name and resolution, then by name alone, then falls back to the primary, so a reordered or unplugged
   monitor never opens the video on the wrong screen silently. The card still shows the `--screen N` numbers.
4. **A reopen from the card uses the new choice.** The restart drops the flags the change overrides
   (`--screen`, `--no-rtx`, `--no-vsr`, and the decoder flag if one exists) from the arguments it passes on,
   since the choice is now saved. Other opens (Open, drag and drop) keep the original flags as today.
5. **Player keys keep working while the card is open** (space, arrows, `V`, `F`, `I`). The card is a plain
   window, not a popup, so `ui_has_keyboard()` stays false for it; `S` and `Esc` close it, and `Esc` closes
   the card before it leaves fullscreen or quits. Clicking outside the card closes it. The card is mouse-only:
   no Tab/arrow navigation between its controls (arrows stay seek/volume).

## Left to the builder

- Exact sizes and spacing within the card (following the polish spec's scale rules).
- Whether `--start` and the paused state are one flag or two.
- How the switch widget is drawn, as long as it matches the capsule.
