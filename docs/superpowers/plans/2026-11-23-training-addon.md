# Training Add-on Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** People who download Open Player can choose a videos folder, train Pixel Flow, and end with it working,
all from Settings > Training.

**Architecture:** train.py learns four things (a stop file, `--more`, a `failed` state, batch size from VRAM). A new
C++ unit `training_addon` downloads and unpacks a pinned Python + wheels + FFmpeg into
`%LOCALAPPDATA%\Open Player\training\` (TensorRT add-on pattern). A new unit `training_run` starts/stops train.py,
exports and installs the model. The Training tab gets the buttons; the installer ships the scripts.

**Tech Stack:** C++20 player (SDL3, ImGui, WinHTTP), Python 3.13 embeddable + PyTorch, Inno Setup.

**Spec:** `docs/superpowers/specs/2026-11-23-training-addon-design.md`

## Global Constraints

- Windows only; keep `#ifdef _WIN32` guards so the Linux build still compiles (don't build or test Linux).
- Every download: HTTPS, host allow-list, exact size cap, SHA-256 fixed in the source. No pip at install or run time.
- Install through `training.new\` + rename; any failure or cancel leaves no `training.new\` and the old add-on intact.
- No admin rights; writes only under `%LOCALAPPDATA%\Open Player\` and the settings folder (`%APPDATA%\open-player\`).
- Folders: add-on `user_data_dir()\training`, work `user_data_dir()\training-work`, scripts `{app}\training\`
  (installed) or the repo root (developer build: `exe_dir()\..`).
- Defaults only in the UI: 100000 steps, size `standard`. `--more` adds 100000.
- The model is installed as `user_data_dir()\vsr.onnx` + `vsr.ref` via temp + rename.
- Our model is "Pixel Flow" (two words) in all UI text.
- Commit and push after every task (branch `training-addon`).
- End: 4 bug-check agents in parallel, one a security review; fix real findings with tests.

## Review Focus

- A videos folder path with spaces, non-ASCII letters or a trailing backslash: train.py must get it intact
  (test in Task 4: `train_command` round-trips through `CommandLineToArgvW`).
- Player closed and reopened mid-training: tab shows the run, Stop still works (Task 4: stop file test needs no
  process handle; Task 5 UI state from `training.json` alone).
- Disk full or cancel during the 3.5 GB download: no `training.new\` left, old add-on untouched (Task 3 test with a
  failing extract).
- A run started from the command line: tab shows progress but no Stop/Use (Task 5 UI check with `out` elsewhere).
- Out of GPU memory at batch 1, or no usable videos: tab shows `failed` and the reason (Task 1 test).

---

### Task 1: train.py: stop file, `--more`, `failed` state

**Files:**
- Modify: `train.py` (args ~330-349, schedule 355-379, loop ~446-496, `__main__`)
- Test: `tests/train_check.py`

**Interfaces:**
- Produces: stop file `status_file().with_name("training.stop")`; `training.json` `state` adds `"failed"` with
  `"reason": str`; checkpoint key `"base"` (int, default 0); `lr_at(step, steps, peak, base=0)`.

- [ ] **Step 1: Write failing tests in `tests/train_check.py`**

```python
# schedule restarts over the new N with --more: base = start step
assert train.lr_at(100_000, 200_000, 2e-4, base=100_000) == 2e-4
assert abs(train.lr_at(200_000, 200_000, 2e-4, base=100_000) - 2e-4 / 20) < 1e-12
assert train.lr_at(500, 1000, 2e-4) == train.lr_at(500, 1000, 2e-4, base=0)
```
In the existing CUDA+ffmpeg block add three runs on the 3 test videos (`--steps 100 --workers 0 --crop 32
--frames 2 --batch 2 --size fast`):
  1. create `training.stop` (with `XDG_CONFIG_HOME` set to the temp dir) before start → exits 0, `training.json`
     `state == "stopped"`, stop file gone, `s.pt` saved.
  2. `--resume s.pt --more 100` after a finished 100-step run → exits 0, checkpoint `step == 200`, `base == 100`.
  3. an empty videos folder → nonzero exit, `training.json` `state == "failed"`, `"no usable videos" in reason`.

- [ ] **Step 2: Run `py tests/train_check.py`** — Expected: FAIL on the `lr_at` base assert.

- [ ] **Step 3: Implement in `train.py`**
  - `lr_at(step, steps, peak, base=0)`: cosine over `(step - base) / (steps - base)`.
  - `--more N` (int; requires `--resume`, else SystemExit): `base = start`, `a.steps = start + N`; saved as `base`
    in every `save(...)`; a plain resume reads `base` from the checkpoint (default 0). The "already has N steps"
    error message adds "or use --more N".
  - Stop file: in the loop, `if stop or stop_file.exists()`: treat as Ctrl+C; delete the file after saving.
    Delete a stale stop file at start.
  - `failed`: wrap `main()` in `__main__`: on `SystemExit` with a message or any `Exception`, `write_status(
    state="failed", reason=str(e)[:300])` then re-raise. `KeyboardInterrupt` keeps today's behaviour.

- [ ] **Step 4: Run `py tests/train_check.py`** — Expected: `train checks passed (...)`.

- [ ] **Step 5: Commit** `feat(train): stop file, --more, failed state in training.json`

### Task 2: train.py: batch size from VRAM, halve on out-of-memory

**Files:**
- Modify: `train.py`
- Test: `tests/train_check.py`

**Interfaces:**
- Produces: `auto_batch(total_bytes: int, crop: int, frames: int) -> int`; `--batch` default `None` (auto).

- [ ] **Step 1: Measure on this RTX 5080** (16 GB): peak `torch.cuda.max_memory_allocated()` for 20 steps at
  batch 8, crop 256, frames 8, `standard`. Record the bytes-per-sample in a comment next to `auto_batch`.

- [ ] **Step 2: Write failing tests**

```python
assert train.auto_batch(16 * 2**30, 256, 8) == 8
assert train.auto_batch(8 * 2**30, 256, 8) in (3, 4)
assert train.auto_batch(2 * 2**30, 256, 8) == 1
assert train.auto_batch(64 * 2**30, 256, 8) == 8   # never above today's default
```

- [ ] **Step 3: Run `py tests/train_check.py`** — Expected: FAIL, `auto_batch` missing.

- [ ] **Step 4: Implement**
  - `auto_batch`: `min(8, max(1, floor(0.75 * total / per_sample)))`, `per_sample` scaled by
    `(crop/256)^2 * frames/8` from the measured value (0.75: room for a video playing beside it).
  - `a.batch = a.batch or auto_batch(torch.cuda.get_device_properties(0).total_memory, a.crop, a.frames)`; print
    the batch chosen.
  - Out of memory (`torch.OutOfMemoryError`) inside a step: `torch.cuda.empty_cache()`, halve the batch (rebuild
    the DataLoader with the new `batch_size`), print `out of GPU memory: batch now N`, put `batch` in
    `training.json`, redo the step. At batch 1: `SystemExit("not enough GPU memory")` (→ `failed` from Task 1).

- [ ] **Step 5: Run `py tests/train_check.py`** — Expected: pass. While the user's `D:\Torrent` run is training,
  only these short test-video runs share the GPU; Step 1's measurement waits until that run is stopped or done.

- [ ] **Step 6: Commit** `feat(train): batch size from the GPU's memory, halved on out-of-memory`

### Task 3: The Training add-on (download, verify, unpack)

**Files:**
- Create: `player/training_addon.h`, `player/training_addon.cpp`
- Modify: `player/CMakeLists.txt:43` (add the .cpp), `player/ui.cpp` (call `training_addon_selftest()` from
  `ui_selftest`, like `addon_selftest`), `player/trt_loader.{h,cpp}` (add `training_addon_dir()`,
  `training_work_dir()`)
- Test: `training_addon_selftest()` in `player/training_addon.cpp`

**Interfaces:**
- Consumes: `https_download`, `verify_file_sha256` (update.h); `run_tar` pattern and `finish_pending_addon_removal`
  pattern (addon.cpp); `user_data_dir()`.
- Produces:
  - `struct TrainingPin { const char* url; const char* sha256; uint64_t size; const char* kind; };` kind:
    `"python"` (embeddable zip → `python\`), `"wheel"` (→ `Lib\site-packages\`), `"ffmpeg"` (zip → `ffmpeg\`, only
    `ffmpeg.exe`, `ffprobe.exe`, licence files kept).
  - `inline constexpr int TRAINING_ADDON_VERSION = 1;` and `extern const std::vector<TrainingPin> TRAINING_PINS;`
  - `uint64_t training_addon_size();` (sum of sizes, for "3.5 GB" text)
  - `int training_addon_installed();` 0 none, 1 current, 2 outdated (`version.txt` != `TRAINING_ADDON_VERSION`).
  - `bool install_training_addon(const std::function<void(uint64_t,uint64_t)>& progress, const std::atomic<bool>&
    cancel, std::string& error);`
  - `bool unpack_training_file(const TrainingPin& pin, const std::string& file, const std::string& into,
    std::string& error);` (testable without downloads)
  - `void remove_training_addon(bool in_use); void finish_pending_training_removal();`
  - `std::string training_python();` (`training_addon_dir()\python\python.exe`) and
    `std::string training_ffmpeg_dir();`
  - `int training_addon_selftest();`

- [ ] **Step 1: Choose the pins.** Python 3.13.x embeddable amd64 zip from `www.python.org`; torch 2.15 cu132
  `cp313` win wheel from `download.pytorch.org` (fall back to the newest torch with a `cp313` CUDA wheel that passes
  Step 2); its dependency wheels (`filelock typing_extensions sympy mpmath networkx jinja2 MarkupSafe fsspec
  setuptools`) plus `numpy onnx onnxscript onnx_ir ml_dtypes protobuf packaging` from `files.pythonhosted.org`; an
  FFmpeg release zip from a versioned, immutable URL (BtbN's GitHub release builds; host
  `github.com` + `objects.githubusercontent.com`). Get sizes and SHA-256 by downloading each once.

- [ ] **Step 2: Prove the pins work (scratchpad, not the repo):** unpack them by hand the way the code will
  (zips only, `._pth` with `Lib\site-packages`, `..\` scripts folder path, `import site` on), then run
  `python.exe tests\train_check.py` with `PATH` = `ffmpeg\;%PATH%`. Expected: `train checks passed` and exit code 0
  (no exit crash). Fix pins until it does. Then write `TRAINING_PINS`.

- [ ] **Step 3: Write `training_addon_selftest()`** (test zips built with the existing `write_zip` helper):
  - a "wheel" zip unpacks into `Lib\site-packages\` with its folders;
  - an entry with `..\` or an absolute path is refused and nothing lands outside;
  - the "python" kind rewrites `python313._pth` to contain `Lib\site-packages`, the scripts folder and
    `import site`;
  - the "ffmpeg" kind keeps only `ffmpeg.exe`, `ffprobe.exe`, `LICENSE*`;
  - an install whose unpack fails (blocker file at `training.new`) returns false, leaves no `training.new\` and an
    existing `training\` untouched;
  - `version.txt` "1" → `training_addon_installed() == 1`; "0" → 2; missing → 0.

- [ ] **Step 4: Run `build-win\open-player.exe --uitest`** — Expected: FAIL (not implemented).

- [ ] **Step 5: Implement** per the Interfaces; downloads go to `user_data_dir()\training-download\` one file at a
  time (deleted after unpack); one progress over `training_addon_size()`; hosts allow-list per pin's host.

- [ ] **Step 6: Run `--uitest`** — Expected: `ui checks passed`.

- [ ] **Step 7: Commit** `feat(player): the Training add-on: pinned Python, PyTorch and FFmpeg`

### Task 4: Running training from the player

**Files:**
- Create: `player/training_run.h`, `player/training_run.cpp`
- Modify: `player/CMakeLists.txt`, `player/ui.cpp` (selftest call), `player/settings.{h,cpp}`
  (`train_videos` key; `TrainingStatus` gets `reason`, `batch`; `parse_training_status` reads them and accepts
  `"failed"`)
- Test: `training_run_selftest()`; settings round-trip in `trt_loader.cpp`'s settings checks

**Interfaces:**
- Consumes: `training_python()`, `training_ffmpeg_dir()`, `training_work_dir()`, `quote_arg` (ui.h).
- Produces:
  - `std::string training_scripts_dir();` (`exe_dir()\training` if it has train.py, else `exe_dir()\..`)
  - `enum class StartMode { New, Resume, More };`
  - `std::wstring train_command(const std::string& python, const std::string& scripts, const std::string& videos,
    const std::string& work, StartMode m);` → `"<python>" -u "<scripts>\train.py" "<videos>" --out
    "<work>\vsr.pt"` + `--resume "<work>\vsr.pt"` (Resume) or + `--resume ... --more 100000` (More).
  - `bool start_training(const std::string& videos, StartMode m, std::string& error);` (CreateProcessW,
    `CREATE_NO_WINDOW`, env with `training_ffmpeg_dir()` first in `PATH`, stdout+stderr to `<work>\train.log`, the
    process handle kept in the unit for `training_child_exited()`)
  - `bool training_child_exited(int& code);` (false while running or never started here)
  - `void request_stop();` (writes `training.stop` next to `training.json`)
  - `bool own_run(const TrainingStatus& t);` (`t.out` names `<work>\vsr.pt`, compared case-insensitively as paths)
  - `bool export_and_install(std::string& error);` (runs `export.py <work>\vsr_best.pt`, waits, copies
    `vsr_best.onnx`/`.ref` → `user_data_dir()\vsr.onnx`/`vsr.ref` via `.tmp` + rename)
  - `bool start_over(std::string& error);` (deletes `<work>\`; refuses while `own_run` is training and not stale)
  - `int training_run_selftest();`

- [ ] **Step 1: Write `training_run_selftest()`:**
  - `train_command` with videos `D:\My Vids\Zoë\` (space, non-ASCII, trailing backslash) parsed back with
    `CommandLineToArgvW` gives exactly `[python, -u, scripts\train.py, D:\My Vids\Zoë\, --out, work\vsr.pt]`;
    More adds `--resume work\vsr.pt --more 100000`.
  - `request_stop()` with `XDG_CONFIG_HOME` = temp dir creates `open-player\training.stop`.
  - `own_run`: `out` = work's vsr.pt (other case, `/` vs `\`) → true; `D:\X\vsr.pt` → false.
  - `export_and_install` install half: given fake `vsr_best.onnx`/`.ref` in a temp work dir and a temp
    `user_data_dir` (`LOCALAPPDATA` override), the copies appear, no `.tmp` left; an existing `vsr.onnx` is replaced.
  - settings: `train_videos=D:/Zoë vids` round-trips; `training.json` with `"state":"failed","reason":"x"` parses.

- [ ] **Step 2: Run `--uitest`** — Expected: FAIL.

- [ ] **Step 3: Implement** per the Interfaces. Export step's Python run uses the same env and `CREATE_NO_WINDOW`;
  its output appends to `train.log`.

- [ ] **Step 4: Run `--uitest`** — Expected: pass.

- [ ] **Step 5: Commit** `feat(player): start, stop and export training from the player`

### Task 5: The Training tab

**Files:**
- Modify: `player/ui.h` (UiState fields, UiAction kinds), `player/ui.cpp` (`draw_training`, its UI checks
  ~1031-1046), `player/main.cpp` (actions, folder dialog code 2, add-on thread, child-exit watch, auto-export)

**Interfaces:**
- Consumes: Tasks 3 and 4.
- Produces: UiAction kinds `ChooseTrainVideos, StartTraining, StopTraining, UseModel, StartOver,
  DownloadTrainingAddon, CancelTrainingAddon, RemoveTrainingAddon`; UiState fields
  `std::string train_videos; int train_video_count = -1; int training_addon = 0; float training_addon_progress =
  -1; std::string training_addon_error; bool has_checkpoint, has_best, finished_own, gpu_ok = true; bool
  exporting;`

- [ ] **Step 1: Extend the tab's UI checks** (existing loop at ui.cpp ~1031) with these states, each drawn at both
  scales; assert the visible buttons by label:
  - no add-on, folder set → `Start` (and the "3.5 GB" download note);
  - add-on downloading → progress + `Cancel`, no `Start`;
  - own run training → `Stop`, `Use this model` (has_best), no `Start`;
  - own run finished → `Train more`, `Use this model`, `Start over`;
  - command-line run (`out` elsewhere) training → text "started from the command line", no `Stop`/`Use this model`;
  - failed → the reason text and "train.log" path, `Start`;
  - `gpu_ok == false` → "Training needs an NVIDIA graphics card", no `Start`.
  A click on `Start` returns `StartTraining`.

- [ ] **Step 2: Run `--uitest`** — Expected: FAIL.

- [ ] **Step 3: Implement `draw_training`** per the spec's sketch (layout as in the design doc's text sketch;
  sizes from `training_addon_size()`).

- [ ] **Step 4: Wire `main.cpp`:**
  - `ChooseTrainVideos`: `SDL_ShowOpenFolderDialog` posting `SDL_EVENT_USER` code 2; saves `train_videos`, counts
    videos on a thread (extensions as train.py's `VIDEO_EXT`).
  - `StartTraining`: add-on missing/outdated → start its download (thread, like TensorRT's), then start; else
    `start_training` with `More` if `finished_own`, `Resume` if a checkpoint exists, else `New`.
  - `StopTraining` → `request_stop()`. `UseModel` → `export_and_install` on a thread, then
    `reopen("Loading your Pixel Flow...", true)`; TensorRT missing → post "Pixel Flow also needs TensorRT: Settings
    > Upscaling > Download".
  - Once a second (where training.json is read today): if `training_child_exited(code)` and the status isn't
    finished/stopped/failed → show failed "training stopped unexpectedly, see train.log"; when an own run turns
    `finished`, run `UseModel` once.
  - `gpu_ok`: from the CUDA device check the player already makes at start.
  - At start: `finish_pending_training_removal()` next to `finish_pending_addon_removal()`.

- [ ] **Step 5: Build (`build-windows.bat`) and run `--uitest`** — Expected: `ui checks passed`.

- [ ] **Step 6: Commit** `feat(player): the Training tab chooses videos, starts, stops and installs Pixel Flow`

### Task 6: Installer, docs, end-to-end

**Files:**
- Modify: `installer/open-player.iss` (ship `train.py model.py train_diag.py export.py` to `{app}\training\`; the
  uninstall step's `Data` cleanup also offers `training\` and `training-work\`), `installer/THIRD-PARTY-NOTICES.txt`
  (Python, PyTorch, FFmpeg: downloaded, not shipped), `README.md` (Install step 3: training from the player)

- [ ] **Step 1: Build the installer** (`installer\build-installer.bat`); install it to a test folder. Expected:
  `{app}\training\train.py` exists.

- [ ] **Step 2: End-to-end on the installed copy, with the user's OK first** (3.5 GB download; the user's own
  training run must be stopped or finished, it uses the GPU): Download → choose a small folder (the 3 test videos
  from train_check) → Start → Stop → Start (resumes) → Use this model → About shows Pixel Flow usable. Expected:
  each step behaves as in the spec.

- [ ] **Step 3: Commit** `feat(installer): ship the training scripts; docs for training in the player`

- [ ] **Step 4: Four bug-check agents in parallel** (correctness; Windows process/filesystem; UI states and
  concurrency; security: downloads and hashes, command line from a picked path, stop file, temp + rename, nothing
  sent off the machine). Black-box only for NVIDIA binaries. Fix real findings with tests, commit, push.
