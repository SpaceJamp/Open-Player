# Settings Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A settings card in the player, drawn like the capsule, with a live upscaler switch, a decoder and monitor choice that reopen the video in place, and choices saved in `settings.txt`; our model named Pixel Flow (PXF).

**Architecture:** Pure settings logic (parse, format, monitor matching, restart arguments) in a new `player/settings.{h,cpp}`, checked by `--uitest`. The card is drawn in `player/ui.cpp` with the existing helpers and returns `UiAction`s; `player/main.cpp` applies live ones in place and reopens for the rest through the existing restart path plus a new `--start`.

**Tech Stack:** C++20, Dear ImGui 1.93 (vendored), SDL3, CUDA, FFmpeg; Windows build `build-windows.bat`, Linux build in WSL Arch `cmake --build build`.

**Spec:** `docs/superpowers/specs/2026-11-21-settings-window-design.md`

## Global Constraints

- Look: the polish spec's colours: card background `rgba(20,22,30,0.72)`-style dark, violet `#5b4fd6` for the selected segment, cyan `#3fd0e0` for hover/focus; Inter + Tabler icons; every size scaled by `ui_scale`.
- Name: "Pixel Flow" (two words, never "PixelFlow"), short "PXF"; sizes "Pixel Flow", "Pixel Flow Fast", "Pixel Flow Lean". File names `vsr.onnx`/`vsr.pt` and `--size` unchanged.
- Settings file: `settings.txt` in the same folder as `recent_file()`, `key=value` per line, written atomically (temp + rename); bad/unknown → defaults; never fails startup.
- Command-line flags win over saved settings for that run and are not saved; a reopen from the card drops the flags its change overrides.
- No CPU upscaling. RTX VSR still requires Pixel Flow's model and screen plan (spec call 1).
- No automated GUI input on the user's desktop. Windows test windows open on screen 2 (`--screen 2`), never driven by keystrokes.
- Both builds must stay green: non-RTX code paths compile on Linux (`#ifdef HAVE_RTX_VSR` around RTX-only code).

## Review Focus

1. A hand-edited or damaged `settings.txt` (CRLF line ends, spaces around `=`, `hide_after=999`, binary junk) → defaults for the bad lines, the rest kept, no crash. Test: Task 1 `settings_parse` cases.
2. Two monitors with the same name, or the saved one unplugged → the right one by resolution, else the primary, never a wrong one silently. Test: Task 1 `match_screen` cases.
3. A video path with spaces, brackets or non-ASCII, opened from the picker, then reopened from the card → same file, at the same position. Test: Task 1 `reopen_args` case with `/x/My Movie (2012) é.mkv`.
4. `--start` past the end, negative, or on a file with unknown duration → clamps to [0, dur) or plays from 0. Test: Task 3 `start_position` cases.
5. Selecting Pixel Flow or RTX while paused or after the video ended → the new upscale shows at once, like the `V` key. Covered by routing through the same `reset/step` path as `toggle_vsr` (Task 4); checked by hand.

---

### Task 1: Settings logic

**Files:**
- Create: `player/settings.h`, `player/settings.cpp`
- Modify: `player/CMakeLists.txt:43` (add `settings.cpp`), `player/ui.cpp` (`ui_selftest` calls `settings_selftest()`)

**Interfaces:**
- Produces:
  - `enum class Upscaler { Rtx, PixelFlow, Off };`
  - `struct Settings { Upscaler upscaler = Upscaler::Rtx; bool cpu_decode = false; std::string screen; bool start_fullscreen = false; bool remember_volume = true; float volume = 1; int hide_after = 2; bool info_at_start = false; };` (`screen` empty = primary; `hide_after` one of 1,2,3,5)
  - `Settings settings_parse(const std::string& text);` / `std::string settings_format(const Settings&);`
  - `Settings load_settings(const std::string& file);` / `bool save_settings(const std::string& file, const Settings&);`
  - `std::string settings_file();` (same folder as `recent_file()`)
  - `struct ScreenInfo { std::string name; int w, h; float hz; bool primary; };`
  - `std::string screen_key(const ScreenInfo&);` → `"PG27UCDM 3840x2160"`
  - `int match_screen(const std::vector<ScreenInfo>&, const std::string& key);` → index, or the primary's index, or 0
  - `std::vector<std::string> reopen_args(const std::vector<std::string>& argv, const std::string& video, double start, bool paused);` — `restart_args` with `--screen N`, `--no-rtx`, `--no-vsr`, `--cpu-decode`, `--start T`, `--paused` removed, then `--start <start>` and, when paused, `--paused` appended
  - `int settings_selftest();` (0 on pass)
- Keys in the file: `upscaler=rtx|pixelflow|off`, `decoder=gpu|cpu`, `screen=<key>`, `start_fullscreen=0|1`, `remember_volume=0|1`, `volume=<0..2>`, `hide_after=1|2|3|5`, `info_at_start=0|1`.

- [ ] **Step 1: Write the failing tests** in `settings_selftest()` (same `check` macro style as `ui_selftest`):

```cpp
Settings d;
check(settings_parse("") .upscaler == Upscaler::Rtx && d.hide_after == 2 && d.remember_volume);
Settings s = settings_parse("upscaler=pixelflow\r\ndecoder = cpu\nhide_after=999\nvolume=7\nbogus=1\n\x01\x02\nscreen=LG ULTRAGEAR 2560x1440\n");
check(s.upscaler == Upscaler::PixelFlow && s.cpu_decode && s.hide_after == 2 && s.volume == 1 && s.screen == "LG ULTRAGEAR 2560x1440");
check(settings_parse(settings_format(s)).screen == s.screen && settings_parse(settings_format(s)).upscaler == s.upscaler);
// save/load round trip in a temp folder; a missing file loads defaults; save leaves no .tmp behind
std::vector<ScreenInfo> mon = {{"PG27UCDM", 3840, 2160, 240, true}, {"LG ULTRAGEAR", 2560, 1440, 60, false},
                               {"LG ULTRAGEAR", 1920, 1080, 60, false}};
check(screen_key(mon[1]) == "LG ULTRAGEAR 2560x1440");
check(match_screen(mon, "LG ULTRAGEAR 1920x1080") == 2 && match_screen(mon, "LG ULTRAGEAR 3840x2160") == 1);
check(match_screen(mon, "Gone 1x1") == 0 && match_screen(mon, "") == 0 && match_screen({}, "x") == 0);
check(reopen_args({"open-player", "--screen", "2", "--no-rtx", "a.mkv", "--start", "3", "--paused"}, "/x/My Movie (2012) é.mkv", 61.5, false) ==
      (std::vector<std::string>{"open-player", "/x/My Movie (2012) é.mkv", "--start", "61.5"}));
check(reopen_args({"open-player", "--fullscreen", "a.mkv"}, "a.mkv", 0, true) ==
      (std::vector<std::string>{"open-player", "--fullscreen", "a.mkv", "--start", "0", "--paused"}));
```

(`match_screen` with a name match but no resolution match picks the first monitor of that name.)

- [ ] **Step 2: Run, expect a compile failure** (`settings.h` missing): `cmd /c build-windows.bat`.
- [ ] **Step 3: Implement `player/settings.{h,cpp}`**; `save_settings` writes `file + ".tmp"`, closes, checks, renames (as `write_pair_file` does); trims spaces and `\r`; ignores unknown keys and out-of-range values.
- [ ] **Step 4: Run** `build-win\open-player.exe --uitest` → `ui checks passed`; Linux: `wsl -d Arch -- bash -lc 'cd "/mnt/c/Users/Yeti/Documents/Super Resoulution" && cmake --build build -j && SDL_VIDEO_DRIVER=offscreen ./build/open-player --uitest'` → `ui checks passed`.
- [ ] **Step 5: Commit** `feat(player): settings file, monitor matching and reopen arguments`.

### Task 2: CPU decoding option

**Files:** Modify `player/media.h:26`, `player/media.cpp:59-76`, `player/main.cpp` (flag parsing, USAGE)

**Interfaces:**
- Produces: `bool Media::open(const char* path, int cuda_device, bool nvdec = true);`, `bool Media::nvdec() const;` (NVDEC actually in use; false after a fallback or with the option), and the flag `--cpu-decode`.

- [ ] **Step 1: Check today's behaviour** (the failing case): on Linux, `ffmpeg -f lavfi -i testsrc2=size=320x240:rate=25 -t 2 /tmp/c.mp4`, then `SDL_VIDEO_DRIVER=offscreen SDL_AUDIO_DRIVER=dummy timeout 5 ./build/open-player --cpu-decode /tmp/c.mp4` → prints usage (unknown flag).
- [ ] **Step 2: Implement**: `nvdec == false` skips `av_hwdevice_ctx_create` and prints `decoding on the CPU (setting)`; the info overlay's Decode line already reports the software path.
- [ ] **Step 3: Run** the same command → stderr has `decoding on the CPU`, exit 0 after the timeout's kill or the clip's end; run without the flag → no such line.
- [ ] **Step 4: Commit** `feat(player): --cpu-decode`.

### Task 3: Startup from settings, `--start`, remembered volume, screen by identity

**Files:** Modify `player/main.cpp` (argument parsing ~132-160, screen choice ~186-196, after `media.open`, `set_volume`), `player/settings.{h,cpp}` (pure helper)

**Interfaces:**
- Consumes: Task 1 (`load_settings`, `settings_file`, `match_screen`, `screen_key`, `ScreenInfo`), Task 2 (`Media::open(..., nvdec)`).
- Produces: `double start_position(double start, double dur);` in `settings.cpp` (clamps to `[0, dur - 1]` when `dur > 1`, `max(0, start)` when dur unknown); flag `--start T`; a `Settings cfg` in `main` with the flags applied over it; `std::vector<ScreenInfo> list_screens(std::vector<SDL_DisplayID>* ids)` in `main.cpp` (also used by `--check`'s screen lines).

- [ ] **Step 1: Failing tests** in `settings_selftest`: `start_position(-3, 100) == 0`, `start_position(500, 100) == 99`, `start_position(42, 100) == 42`, `start_position(42, 0) == 42`.
- [ ] **Step 2: Run `--uitest`** → fails to compile.
- [ ] **Step 3: Implement**: two values in `main`: `Settings saved = load_settings(settings_file())` (what the file holds; only the card and the volume change it, then `save_settings`) and `Settings cfg = saved` with the flags applied for this run (`--no-rtx` → PixelFlow, `--no-vsr` → Off, `--cpu-decode`, `--screen N`, `--fullscreen`), so a flag is never saved. With no `--screen`, `match_screen` on `cfg.screen`; volume starts at `cfg.volume` when `remember_volume`, and `set_volume` saves it into `saved` (at most once per second, and on quit); `show_info` starts from `info_at_start`; `upscaler=rtx` where RTX VSR can't open starts on Pixel Flow without changing `saved`; `--start T` seeks to `start_position(T, dur)` before the first frame, and `--paused` starts paused on that frame.
- [ ] **Step 4: Run** `--uitest` (both builds) → pass; `bash tests/smoke.sh vsr.onnx` on Linux → pass (it sets `XDG_CONFIG_HOME`, so no saved settings leak in); on Windows `build-win\open-player.exe --screen 2 --stats --start 10 <clip>` killed after 5 s → first `--stats` line's position ≥ 10.
- [ ] **Step 5: Commit** `feat(player): start from saved settings; --start; remember volume`.

### Task 4: Live upscaler switch and the Pixel Flow name

**Files:** Modify `player/main.cpp` (RTX block ~270-285, prepare stage ~495-515, `toggle_vsr`, `vsr_desc`, log lines, USAGE, `--check`), `player/ui.{h,cpp}` (`vsr_badge_text`)

**Interfaces:**
- Consumes: Task 1 `Upscaler`, Task 3 `cfg`.
- Produces:
  - `std::string vsr_badge_text(bool available, bool on, bool rtx, int factor);` → `"PXF 2x"`, `"RTX 2x"`, `"VSR off"`, `""`.
  - In `main`: `auto set_upscaler = [&](Upscaler u) -> bool` (false + notice when the choice can't be used), `bool rtx_available` (RTX opened or not yet tried), `std::string vsr_desc` recomputed by a `describe_vsr()` lambda.

- [ ] **Step 1: Failing test** in `ui_selftest`: `vsr_badge_text(true, true, false, 2) == "PXF 2x" && vsr_badge_text(true, true, true, 2) == "RTX 2x" && vsr_badge_text(true, false, true, 2) == "VSR off" && vsr_badge_text(false, false, false, 2).empty()`.
- [ ] **Step 2: Run `--uitest`** → compile failure (signature).
- [ ] **Step 3: Implement**: `set_upscaler` sets `vsr_on`/`use_rtx`, opens `RtxVsr` on first RTX choice (`rtx.open(w, h, nw*s, nh*s, st)`; failure → `rtx_available = false`, notice "NVIDIA RTX isn't available on this GPU or driver", stays on Pixel Flow), then the same `reset = true, step |= paused, drop_ahead()` as `toggle_vsr`, and `describe_vsr()`. `toggle_vsr` (V key, badge) turns off/on and restores the last chosen upscaler; it doesn't save. Replace "our model/network" with "Pixel Flow" in logs, USAGE, `--check`, notices ("Pixel Flow, 2x" / "NVIDIA RTX, 2x").
- [ ] **Step 4: Run** both builds + `--uitest` → pass; Windows `build-win\open-player.exe --check vsr.onnx` → RTX line mentions Pixel Flow; Linux `bash tests/smoke.sh vsr.onnx` → pass.
- [ ] **Step 5: Commit** `feat(player): switch upscaler live; name our model Pixel Flow`.

### Task 5: The settings card

**Files:** Modify `player/ui.{h,cpp}`, `player/theme.h` (icon codepoints)

**Interfaces:**
- Consumes: Task 1 `Settings`, `ScreenInfo`; Task 4 badge.
- Produces:
  - `UiState` gains: `bool settings_open; Settings settings; std::vector<ScreenInfo> screens; int screen_index; int playing_on; bool rtx_available, pxf_available, nvdec_available; std::string vsr_unavailable_reason;` and `controls_visible(now, last_mouse, paused, hovered, hide_after)`.
  - `UiAction::Kind` gains `ToggleSettings, SetUpscaler, SetDecoder, SetScreen, SetStartFullscreen, SetRememberVolume, SetHideAfter, SetInfoAtStart` (`value` carries the choice: `Upscaler` as int, screen index, seconds, 0/1).
  - `bool ui_settings_hovered();` (for click-outside closing).
- Icons (Tabler, add the codepoints to `theme.h` and the font's glyph list; verify each in the pinned font, as the existing icon check does): `settings` (gear), `device-desktop`, `cpu`, `sparkles` (exists), `adjustments`. 

- [ ] **Step 1: Failing tests** in `ui_selftest`: `controls_visible(10, 7.5, false, false, 3)` true and `controls_visible(10, 7.5, false, false, 2)` false; headless ImGui frames (as the existing ones) with `settings_open = true` at scale 1 and 2.5, with 0, 1 and 3 screens, with `rtx_available = pxf_available = false` → no ImGui assert, and the draw list is non-empty.
- [ ] **Step 2: Run `--uitest`** → compile failure.
- [ ] **Step 3: Implement** the card per the spec's "The card" and the chat sketch: sections Upscaling / Decoding / Display / Playback & controls; segmented buttons (selected = violet), monitor radio rows `1  PG27UCDM  3840x2160 @ 240 Hz  primary` with "playing here", switches; disabled options greyed with a tooltip; ✕ in the corner; a plain ImGui window (not a popup) so `ui_has_keyboard()` stays false; scrolls when taller than 80% of the window. Gear icon in the capsule between the VSR badge and the open icon; "Settings" in the right-click menu.
- [ ] **Step 4: Run** `--uitest` both builds → pass.
- [ ] **Step 5: Commit** `feat(player): settings card`.

### Task 6: Wire the card into the player

**Files:** Modify `player/main.cpp` (event loop, `draw_ui` call ~551, action switch ~558, key handling ~457-475)

**Interfaces:**
- Consumes: Tasks 1-5.

- [ ] **Step 1: Implement**: the capsule stays shown while the card is open (pass `hovered = true` to `controls_visible` then); `S` toggles the card; `Esc` closes it first; a click outside (mouse down while `!ui_settings_hovered()` and the card open) closes it. Actions: `SetUpscaler` → `set_upscaler`, save; `SetHideAfter`/`SetRememberVolume`/`SetStartFullscreen`/`SetInfoAtStart` → set, save; `SetDecoder`/`SetScreen` → set, save, notice ("Switching to CPU decoding…" / "Reopening on screen 2…"), then restart with `reopen_args(argv, path, pos, paused)` through the existing `_execv`/`execv` code (factor out of `open_video` into one `restart_with(args)` lambda). Monitor list rebuilt on `SDL_EVENT_DISPLAY_ADDED/REMOVED/MOVED` and when the card opens; `playing_on` from `SDL_GetDisplayForWindow`. A test hook `OPEN_PLAYER_SHOW_SETTINGS=1` opens the card at start (for the timing check; no input needed).
- [ ] **Step 2: Run** both builds, `--uitest`, Linux `bash tests/smoke.sh vsr.onnx` and `bash tests/codecs.sh` → pass.
- [ ] **Step 3: Timing**: Windows, `OPEN_PLAYER_SHOW_SETTINGS=1 build-win\open-player.exe --screen 2 --stats <1080p clip>` for 15 s, then without the hook → average `ui_ms` with the card ≤ 0.3 ms per pass and dropped frames no higher than without.
- [ ] **Step 4: Commit** `feat(player): settings card applies live, saves, reopens in place`.

### Task 7: Docs

**Files:** Modify `README.md`

- [ ] **Step 1**: rename "our network/model" to Pixel Flow (PXF) throughout; a "Settings" paragraph under Controls (what's live, what reopens, where `settings.txt` lives, flags win); `S` in the key table; `--cpu-decode` and `--start` in Options; the badge text change.
- [ ] **Step 2**: `grep -n "our network\|our model\|Ours" README.md player/*.cpp` → only intentional uses left.
- [ ] **Step 3: Commit** `docs: settings window and the Pixel Flow name`.

### Finish

- [ ] Dispatch 4 bug-check agents (read-only, no GUI input, no reverse engineering of NVIDIA binaries): settings parsing/saving; live switching in `main.cpp`; the card's drawing and input; reopen/restart and flags.
- [ ] Fix their real findings, rebuild both, `--uitest`, smoke, codecs; commit; push when the user asks.
- [ ] Hand the by-hand checks to the user: look against the capsule, live switching, reopening on screen 2 and with CPU decoding, plugging a monitor in and out.
