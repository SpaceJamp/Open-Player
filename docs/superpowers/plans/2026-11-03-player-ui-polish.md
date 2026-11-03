# Player UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default-looking ImGui control bar with the approved floating capsule, centre notice card and restyled info card and menu, sharp at 4K.

**Architecture:** A new `theme` unit loads Inter + Tabler icon fonts (merged) and holds colours and icon codepoints. `ui.cpp` draws everything with ImGui widgets plus the window draw list, at a per-frame scale (window height / 1080). Small pure helpers drive fades and notice timing; `main.cpp` posts notices from its shared action helpers and keeps redrawing while something animates.

**Tech Stack:** C++17, Dear ImGui 1.93.0 (dynamic fonts), SDL3, Vulkan; CMake in Arch WSL.

**Spec:** `docs/superpowers/specs/2026-11-03-player-ui-polish-design.md`

## Global Constraints

- Capsule: 78% of window width, max 1100 px at scale 1, 14 px above bottom, radius = half height, background `rgba(20,22,30,0.72)`.
- Colours: played seek part cyan `#3fd0e0`; track `rgba(255,255,255,0.25)`; VSR badge violet `#5b4fd6` (on) / grey (off) / hidden (no upscaling).
- Seek bar 4 px, 6 px on hover with a white 12 px knob and a time tooltip; seek on release; disabled and no tooltip when duration unknown.
- Capsule visibility rule unchanged (2 s after mouse moves, always while paused or hovered); fade in 150 ms, out 300 ms; cursor hidden in fullscreen while the capsule is fully hidden.
- Notices: centre card; shown 1 s (file name 2 s), fade out 300 ms, new replaces old; events: volume (text + bar), key seek ("+5 s · 0:42:18" / "−5 s · …"), pause/play (icon only), VSR on/off ("VSR on, 2x" / "VSR off"), file name on open. Seek-bar seeks: no notice.
- Info card top-left, nothing else in the top-left corner. Menu: rounded, theme colours, icon per item.
- Scale = window pixel height / 1080, clamped 1–2.5; base font 16 px at scale 1.
- Fonts: `assets/fonts/Inter-Medium.ttf` (OFL) and `assets/fonts/tabler-icons.ttf` (MIT) with licence files, pinned releases; missing fonts → one stderr line and today's text labels, never a failed start.
- Performance: `ui_ms` ≤ 0.3 ms average with capsule + notice showing; 4K fullscreen before/after within ~1 ms on display stages, no more dropped frames.
- Every existing key shortcut and UI action keeps working. Build/test in Arch WSL; tests offscreen.

## Review Focus

- Fonts missing or unreadable: the player starts and shows text-label controls (test in Task 1 and Task 3).
- A tiny window (320x180): the capsule still draws without asserts or negative sizes (test in Task 3).
- Volume above 100% (up to 200%): the notice bar and slider never overflow their track (test in Task 2: `notice_level(2.0f) == 1.0f`).
- Notices fired rapidly (holding an arrow key): each replaces the last; no stacking, alpha resets (test in Task 2).
- A non-ASCII file name in the open notice ("Vidéos", "Amélie"): drawn without asserts; Inter covers Latin-1 (test in Task 3).

---

### Task 1: Fonts and theme

**Files:**
- Create: `assets/fonts/Inter-Medium.ttf`, `assets/fonts/Inter-LICENSE.txt`, `assets/fonts/tabler-icons.ttf`, `assets/fonts/tabler-icons-LICENSE.txt`
- Create: `player/theme.h`, `player/theme.cpp`
- Modify: `player/CMakeLists.txt` (add `theme.cpp`), `player/ui.cpp` (`ui_selftest`), `.superpowers/sdd/<plan>/uidebug.sh` (build the debug test binary at `build/uitest-dbg` so `../assets` resolves; add `theme.cpp`)

**Interfaces:**
- Produces (`theme.h`):
  - `std::string assets_dir();` — `<directory of /proc/self/exe>/../assets`.
  - `bool load_theme(const std::string& font_dir);` — call after `ImGui::CreateContext()`, before the first frame. Loads Inter at 16 px, merges the icon font (`MergeMode = true`, `GlyphMinAdvanceX` = 16, only the listed codepoints), sets style colours/rounding. Returns false (prints one stderr line, leaves ImGui's default font) if a file is missing.
  - `bool theme_loaded();`
  - Icon constants as UTF-8 strings named `ICON_PLAY, ICON_PAUSE, ICON_VOLUME, ICON_VOLUME_OFF, ICON_FOLDER, ICON_MAXIMIZE, ICON_MINIMIZE, ICON_SPARKLES, ICON_FORWARD_5, ICON_BACKWARD_5, ICON_INFO, ICON_X, ICON_CLOCK` plus matching `ImWchar` codepoints `CP_PLAY` … (values taken from the pinned `tabler-icons.css`).
  - Colours `COL_CAPSULE, COL_TRACK, COL_PLAYED, COL_VSR_ON, COL_VSR_OFF, COL_TEXT` as `ImU32` (values from Global Constraints).

- [ ] **Step 1: Download the fonts.** Inter from the `rsms/inter` release `v4.1` zip (`extras/ttf/Inter-Medium.ttf` + `LICENSE.txt`); Tabler from npm `@tabler/icons-webfont@3.31.0` via unpkg (`dist/fonts/tabler-icons.ttf`, `dist/tabler-icons.css` for codepoints, `LICENSE`). If a pinned version is missing, take the newest existing one and record a ruling.
- [ ] **Step 2: Write the failing test** in `ui_selftest()`:

```cpp
ImGui::CreateContext();
check(!load_theme("/definitely/missing") && !theme_loaded());  // fallback: default font stays usable
ImGui::DestroyContext();
ImGui::CreateContext();
if (std::filesystem::exists(assets_dir() + "/fonts/Inter-Medium.ttf")) {
    check(load_theme(assets_dir() + "/fonts") && theme_loaded());
    ImGuiIO& io = ImGui::GetIO(); io.DisplaySize = {1920, 1080};
    unsigned char* px; int fw, fh; io.Fonts->GetTexDataAsRGBA32(&px, &fw, &fh);
    for (ImWchar c : {CP_PLAY, CP_PAUSE, CP_VOLUME, CP_FOLDER, CP_MAXIMIZE, CP_SPARKLES, (ImWchar)'A', (ImWchar)0xE9})
        check(io.Fonts->Fonts[0]->IsGlyphInFont(c));
} else printf("note: assets not found next to the binary, font checks skipped\n");
ImGui::DestroyContext();
```

- [ ] **Step 3: Run** `--uitest`. Expected: build fails (`load_theme`, `assets_dir`, `CP_*` undefined).
- [ ] **Step 4: Implement `theme.{h,cpp}`**, add to CMake and `uidebug.sh`.
- [ ] **Step 5: Run** `--uitest` and `uidebug.sh`. Expected: `ui checks passed` in both, no "skipped" note from `build/`.
- [ ] **Step 6: Commit** `feat(player): bundle Inter and Tabler icon fonts, theme colours`

### Task 2: Fade and notice timing

**Files:**
- Modify: `player/ui.h`, `player/ui.cpp`

**Interfaces:**
- Produces (`ui.h`):
  - `struct UiNotice { const char* icon = nullptr; std::string text; float level = -1; double shown_at = -1e9; double hold = 1; };` (`level` 0..1, < 0 = no bar)
  - `float fade_alpha(double now, double changed_at, bool visible, double in, double out);`
  - `float notice_alpha(double now, const UiNotice& n);` — 0 before `shown_at`, 1 until `shown_at + hold`, linear to 0 over 0.3 s.
  - `float notice_level(float volume);` — `volume / 2` clamped to 0..1 (volume range 0..2).
  - `UiState` gains `float scale = 1; UiNotice notice;` (appended after `recent`, so existing aggregate initialisers keep compiling).

- [ ] **Step 1: Write the failing test:**

```cpp
check(fade_alpha(10, 10, true, .15, .3) == 0 && std::fabs(fade_alpha(10.075, 10, true, .15, .3) - .5f) < 1e-3 && fade_alpha(10.2, 10, true, .15, .3) == 1);
check(fade_alpha(10, 10, false, .15, .3) == 1 && std::fabs(fade_alpha(10.15, 10, false, .15, .3) - .5f) < 1e-3 && fade_alpha(10.4, 10, false, .15, .3) == 0);
UiNotice n{nullptr, "Volume 80%", .4f, 5, 1};
check(notice_alpha(4, n) == 0 && notice_alpha(5.5, n) == 1 && std::fabs(notice_alpha(6.15, n) - .5f) < 1e-3 && notice_alpha(6.4, n) == 0);
n.shown_at = 6.2;  // a new notice replaces the fading one: full again
check(notice_alpha(6.25, n) == 1);
check(notice_level(0.8f) == 0.4f && notice_level(2.0f) == 1.0f && notice_level(-1) == 0);
```

- [ ] **Step 2: Run** `--uitest`. Expected: build fails (`fade_alpha` undefined).
- [ ] **Step 3: Implement** the three helpers and the struct changes.
- [ ] **Step 4: Run** `--uitest` and `uidebug.sh`. Expected: `ui checks passed`.
- [ ] **Step 5: Commit** `feat(player): fade and notice timing`

### Task 3: Draw the capsule, notice card, info card and menu

**Files:**
- Modify: `player/ui.h`, `player/ui.cpp`

**Interfaces:**
- Consumes: Task 1 theme API, Task 2 helpers and `UiState.scale/notice`.
- Produces:
  - `draw_ui(const UiState&)` keeps its signature and `UiAction` kinds; VSR badge click → `ToggleVsr`; folder → `Open`; volume popup slider → `SetVolume` (0..2).
  - `bool ui_animating(double now);` — true while the capsule fade or a notice is in progress (main keeps redrawing).
  - `bool ui_controls_hidden();` — capsule alpha is 0 after the last `draw_ui` (main hides the cursor in fullscreen).
  - Without a loaded theme, icons fall back to today's text labels ("Play", "Pause", "VSR on", "Open", "Fullscreen").

- [ ] **Step 1: Extend the headless test** (replace the current headless loop): for both `load_theme(assets_dir()+"/fonts")` loaded and not loaded, for `DisplaySize` in `{1920,1080}`, `{3840,2160}` (scale 2.5 clamp check: `2160/1080 = 2`; also pass `scale = 2.5f`), `{320,180}`, for `dur` in `{600.0, 0.0}`, draw 3 frames with: a notice `{ICON_VOLUME, "Volume 80%", .4f, now, 1}`, a non-ASCII file notice `{nullptr, "Am\xc3\xa9lie (2001).mkv", -1, now, 2}`, and with the capsule hidden (`last_mouse = now - 10`, not paused). Also `check(ui_animating(now))` right after a notice is set and `!ui_animating(now + 5)` with the capsule hidden and no notice; `check(ui_controls_hidden())` after frames with the capsule hidden for > 0.3 s.
- [ ] **Step 2: Run** `--uitest`. Expected: build fails (`ui_animating`, `ui_controls_hidden` undefined).
- [ ] **Step 3: Implement.** Draw the capsule as one borderless, background-less ImGui window sized to the capsule, painting the rounded rect with `GetWindowDrawList()->AddRectFilled` and the seek track/played part/knob with the draw list over an `InvisibleButton` (hover = `IsItemHovered`, drag = `IsItemActive`, seek on `IsItemDeactivated` when it was dragged or clicked). Global alpha via `ImGui::PushStyleVar(ImGuiStyleVar_Alpha, a)` and alpha-multiplied draw-list colours. Fonts: `PushFont(nullptr, 16 * scale)` at the start of `draw_ui`. The visibility change time is a function-static, as `drag`/`dragging` are today. Info card: same rounded style at `{10*scale, 10*scale}`.
- [ ] **Step 4: Run** `--uitest` and `uidebug.sh`. Expected: `ui checks passed` in both (asserts on).
- [ ] **Step 5: Commit** `feat(player): floating capsule, notice card, restyled info and menu`

### Task 4: Wire into the player

**Files:**
- Modify: `player/main.cpp`, `README.md` (controls paragraph), `.gitignore` (none expected)

**Interfaces:**
- Consumes: `load_theme`, `assets_dir`, `UiNotice`, `notice_level`, `ui_animating`, `ui_controls_hidden`, `ICON_*`.

- [ ] **Step 1: Startup:** after `ImGui::CreateContext()`, `load_theme(assets_dir() + "/fonts")`. Use `assets_dir()` for the window icon too (one way to find assets). Post the file-name notice (`hold = 2`) once the first frame is shown.
- [ ] **Step 2: Notices from the shared helpers:** `toggle_pause` (icon `ICON_PLAY`/`ICON_PAUSE`, no text), `set_volume` ("Volume N%", level `notice_level(volume)`), key seeks only (`ICON_FORWARD_5`/`ICON_BACKWARD_5`, "+5 s · " + `clock_text(target)`), `toggle_vsr` ("VSR on, 2x" with `s`, or "VSR off"). Seek-bar `Seek` actions post nothing.
- [ ] **Step 3: Per pass:** `UiState.scale = clamp(window pixel height / 1080, 1, 2.5)`; mark dirty while `ui_animating(now())` (also when paused, it is time-bounded); in fullscreen `SDL_HideCursor()` when `ui_controls_hidden()`, `SDL_ShowCursor()` on mouse motion or when leaving fullscreen.
- [ ] **Step 4: Run** `uitest.sh`, `bash tests/smoke.sh vsr.onnx`, `bash tests/codecs.sh`. Expected: all PASS.
- [ ] **Step 5: Speed:** offscreen `speed.sh` (before = the commit before Task 1) and on the 4K screen fullscreen `screen4k.sh`. Expected: `ui_ms` ≤ 0.3 average; display stages within ~1 ms of before; dropped not higher.
- [ ] **Step 6: Manual (user):** play a video windowed and fullscreen at 4K; check capsule look, fades, hover tooltip, volume popup, VSR badge, notices for each key, info card, menu icons, cursor hiding.
- [ ] **Step 7: Commit** `feat(player): notices, cursor hiding and 4K scaling wired in`
