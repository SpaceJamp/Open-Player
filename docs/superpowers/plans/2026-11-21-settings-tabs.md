# Settings Tabs, Resume Position, Training Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Side tabs (General / Training / About) in the settings card, per-video resume, and a live Training tab fed by `train.py`.

**Architecture:** Pure helpers in `player/settings.{h,cpp}` (positions list, training status parsing) checked by `--uitest`; `train.py` gains a small atomic `training.json` writer checked by `tests/train_check.py`; `player/ui.cpp` draws the side tabs; `player/main.cpp` wires resume and the status reader.

**Tech Stack:** C++20, Dear ImGui (custom-drawn), Python 3 (train.py).

**Spec:** `docs/superpowers/specs/2026-11-21-settings-tabs-design.md`

## Global Constraints

- `positions.txt` and `training.json` live next to `settings.txt` (`settings_file()`'s folder; train.py: `XDG_CONFIG_HOME`, else `%APPDATA%\open-player`, else `~/.config/open-player`).
- `positions.txt`: `<seconds>\t<absolute path>` per line, newest first, at most 200; resume only if `t >= 10` and `t < dur - 30` (dur unknown: `t >= 10`); `--start` wins; default on (`remember_position=1`).
- `training.json` keys: `state` ("training"|"finished"|"stopped"), `step`, `steps`, `loss`, `val_db`, `best_db`, `s_per_step`, `eta_s`, `suggest`, `out`, `updated` (unix time); written atomically every 100 steps and after each validation; "not running" when `updated` > 5 min old and state "training".
- Tabs on the left side; icons only when the card is too narrow; last tab remembered (`settings_tab=0|1|2`).
- Untrusted input: both files can be hand-edited or junk: bad lines/values are ignored, files read up to 64 KB / 1 MB.

## Review Focus

1. A video whose path has a tab or a newline in it → never breaks `positions.txt` (refused, not stored). Test: Task 1.
2. A damaged or huge `training.json` (half-written, junk, 10 MB) → "No training yet"/ignored, no crash. Test: Task 2 parse cases.
3. Seeking to a resume point in a video shorter than 40 s → plays from 0. Test: Task 1 `resume_point` cases.
4. The same video opened from two players → the last one to save wins, no corruption (atomic save). Test: Task 1 save leaves no `.tmp`.
5. Training tab with the clock set back (updated in the future) → shown as running, not "not running". Test: Task 2.

---

### Task 1: Positions (resume)

**Files:** Modify `player/settings.{h,cpp}` (tests in `settings_selftest`), `player/main.cpp`.

**Interfaces (Produces):** `using Positions = std::vector<std::pair<double, std::string>>;` `Positions load_positions(const std::string& file);` `bool save_positions(const std::string& file, const Positions&);` `Positions positions_add(Positions, const std::string& path, double t);` (newest first, no duplicates, cap 200, paths with `\t`/`\n`/`\r` refused) `double resume_point(const Positions&, const std::string& path, double dur);` (-1 = start from 0) `std::string positions_file();` `Settings` gains `bool remember_position = true; int settings_tab = 0;`.

- [x] Failing tests: add/dedupe/cap 200; refuse `"a\tb"`; round trip through a temp file, no `.tmp` left; `resume_point` → 120 for (120, dur 3600), -1 for t 5, -1 for t 3590 of 3600, -1 for any t in a 35 s clip, 15 for dur 0 and t 15; junk lines ignored; settings round trip of the two keys.
- [x] `--uitest` fails to compile; implement; `--uitest` passes (Windows and Linux).
- [x] main.cpp: after the start seek, if no `--start` and `cfg.remember_position`: `t = resume_point(...)`; `t >= 0` → `seek(t)` + notice "Resumed at <clock_text(t)>". Save (`positions_add` + `save_positions`) every 10 s while playing, in `restart_with`, and at exit. Test run on screen 2: play 15 s of a clip, kill it, reopen → log/stats show it starts near 15 s.
- [x] Commit.

### Task 2: Training status

**Files:** Modify `train.py`, `tests/train_check.py`, `player/settings.{h,cpp}`.

**Interfaces (Produces):** train.py `status_file() -> Path`, `write_status(**fields)` (atomic: temp + `os.replace`, never raises). C++: `struct TrainingStatus { std::string state, suggest, out; long long step = 0, steps = 0, updated = 0; double loss = -1, val_db = NAN, best_db = NAN, s_per_step = 0, eta_s = 0; };` `bool parse_training_status(const std::string& json, TrainingStatus&);` `std::string training_file();` `bool training_stale(const TrainingStatus&, long long now);` (state "training" and `now - updated > 300`).

- [x] Failing tests: train_check writes a status into a temp `XDG_CONFIG_HOME` and reads back every key, no temp file left; C++: parse a full JSON, reject junk/half JSON, ignore unknown keys, `training_stale` false when `updated` is in the future.
- [x] Implement: train.py calls `write_status(state="training", ...)` every 100 steps and after `check()`, `state="finished"` at the end, `"stopped"` on Ctrl+C. The C++ reader is a small flat-JSON reader (strings, numbers), like `update.cpp`'s.
- [x] `py tests/train_check.py` and `--uitest` pass; commit.

### Task 3: Side tabs and the Training tab

**Files:** Modify `player/ui.{h,cpp}`, `player/main.cpp`.

**Interfaces:** `UiState` gains `int settings_tab;` and `TrainingStatus training; bool training_known; long long now_unix;`; `UiAction::SetSettingsTab` (value 0..2); `SetRememberPosition`.

- [x] Failing tests: headless frames with each tab at scale 1 and 2.5 and at 640x360 (icons-only column), Training tab with no file / running / stale / finished; clicking the second tab row emits `SetSettingsTab` 1; the card still never covers the capsule.
- [x] Implement: left column (icon + label rows; icons only below a width), General = today's sections + the Remember switch, Training = status lines + progress bar (main reads `training.json` once a second while the tab is open), About = today's About.
- [x] `--uitest` both builds; commit.

### Task 4: Docs

- [x] README: tabs, resume, Training tab (and that `train.py` writes `training.json`). Commit.

### Finish

- [x] 4 bug checks (one security); fix findings.
- [ ] Ask before merging into main.
