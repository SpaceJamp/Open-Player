# Training add-on: train Pixel Flow from the player

Date: 2026-11-23. Status: draft, waiting for the user's approval. Reviewed by a builder check; its open questions
are answered below (two by the user, the rest marked as my calls).

## Intent

The user wants people who **download Open Player** to train their own Pixel Flow from inside the player: pick the
folder of videos, start and stop training, and end with Pixel Flow usable, with no command line, no Python install
and no repository checkout. Today that takes setup-windows.ps1, `py train.py <folder>`, `py export.py`, and copying
`vsr.onnx` + `vsr.ref` by hand, which only a developer does. No model ever ships (standing rule), so without this a
downloaded player can never use Pixel Flow.

## Goals

- Settings > Training: **Choose videos folder...**, **Start**, **Stop**, and the live progress the tab already shows.
- Everything training needs comes as an optional **Training add-on**, downloaded on first Start, like the TensorRT
  add-on: pinned URLs and SHA-256s fixed in the source, progress, cancel, remove.
- When training finishes (or on **Use this model** while it runs), the player exports the best model and installs
  it as `%LOCALAPPDATA%\Open Player\vsr.onnx` + `vsr.ref`; Pixel Flow then works (with the TensorRT add-on).

## Non-goals

- Training options in the UI (steps, size, crop, teacher, pairs): defaults only (100000 steps, `standard`). The
  command line keeps them for developers.
- Linux (Windows-only focus); don't break the Linux build (guards stay).
- Non-NVIDIA GPUs: training needs CUDA, as today. The tab says so when no NVIDIA GPU is found.
- Changing how training works (losses, data, model).

## Anti-goals (failure even if it "works")

- Running anything not pinned: no `pip install` resolving versions at run time, no unpinned URL, no hash skipped.
- Needing admin rights, or writing outside `%LOCALAPPDATA%\Open Player\` and the settings folder.
- A half-installed add-on after a failed or cancelled download (same `.new` + rename rule as TensorRT).
- Losing training progress: Stop, closing the player, or a crash of the player must not lose more than the last
  2000 steps (train.py's own save interval) — Stop saves first.
- Shipping a model, or putting the user's videos or paths anywhere off the machine.

## Constraints (from the project as it is)

- `train.py`, `model.py`, `train_diag.py`, `export.py` need Python + torch (CUDA) + numpy; `export.py` also onnx +
  onnxscript. `train.py` runs the **`ffmpeg` and `ffprobe` executables** (subprocess); the player only has FFmpeg's
  DLLs, not these programs.
- PyTorch on Python 3.15 needs torch 2.15 (2.14 crashes at exit, found 2026-11-23). Pin a Python and torch pair
  that is known to train and exit cleanly; prefer Python 3.13 embeddable (wheels exist for every dependency).
- train.py stops cleanly on SIGINT (saves, exits). A windowed player can't send Ctrl+C to a child reliably on
  Windows.
- train.py already writes `training.json` to the player's settings folder (`%APPDATA%\open-player\`); the tab reads
  it once a second. It also stays correct when the player is closed and reopened.
- The installer (installer/open-player.iss) ships the player, its DLLs, assets and notices; no Python files today.
- GitHub release assets are capped at 2 GB per file (reason B was dropped).

## Approaches

- **A. Training add-on (chosen by the user).** Download python.org's embeddable Python, the pinned wheels from
  PyTorch's and PyPI's servers, and an FFmpeg build; unpack into `%LOCALAPPDATA%\Open Player\training\`. The
  installer ships the training scripts. Reuses the TensorRT add-on pattern. Trade-off: ~3.5 GB one-time download;
  pins need refreshing now and then.
- **B. One frozen training program** (PyInstaller) attached to each release. One download, but over GitHub's 2 GB
  per-file limit (split files), a huge build and signing on every release. Dropped.
- **C. Training rewritten in C++.** No extra download, months of work. Dropped.

## Design (decided)

### The add-on

- Folder: `%LOCALAPPDATA%\Open Player\training\` (`training_addon_dir()`), next to `tensorrt\`.
- Contents: `python\` (python.org's embeddable zip, unpacked; its `._pth` file edited to add `Lib\site-packages`
  and the scripts folder), `Lib\site-packages\` (each wheel **unzipped**, not pip-installed: wheels are zip files,
  so no pip and no code runs at install), `ffmpeg\` (`ffmpeg.exe`, `ffprobe.exe` and their licence).
- A fixed list in the source: for each file, its URL, size and SHA-256 (the TensorRT add-on's `https_download` +
  `verify_file_sha256` + host allow-list: `www.python.org`, `download.pytorch.org`, `files.pythonhosted.org`, and
  the FFmpeg build's host). Install goes to `training.new\` and is renamed into place only when everything verified
  and unpacked; any failure or cancel removes `training.new\`.
- Download shows one progress bar over the total size, with Cancel, in the Training tab (same look as TensorRT's).
- **Remove** in the Training tab (and the uninstaller removes it, like TensorRT's folder).

### The scripts

- The installer ships `train.py`, `model.py`, `train_diag.py`, `export.py` into `{app}\training\`. They are part of
  the player's version, so the signed updater keeps them current with it. A developer build uses the repo's copies.

### Running training

- **Choose videos folder...**: the same folder dialog as TensorRT's; saved in settings.txt as `train_videos=`.
- **Start**: if the add-on is missing, offer to download it first. Then start `python.exe -u train.py <folder>
  --out <work>\vsr.pt` (CreateProcessW, no console window, command line built with `quote_arg`), where `<work>` is
  `%LOCALAPPDATA%\Open Player\training-work\`. If `<work>\vsr.pt` exists, add `--resume <work>\vsr.pt` (continue
  where it stopped). `PATH` for the child gets the add-on's `ffmpeg\` first. The child's output goes to
  `<work>\train.log` (for diagnosing; the tab doesn't show it).
- **Stop**: the player creates a stop file, `<settings folder>\training.stop`. train.py checks for it each step
  (next to its SIGINT check), saves, deletes the file and exits. Works even when the player was restarted after
  Start, because it doesn't need the child's handle.
- **Closing the player** leaves training running (it's a separate process). On the next start the tab picks it up
  again from `training.json`, and Stop still works.
- One training at a time: Start is disabled while `training.json` says training and isn't stale.
- **New folder, old checkpoint:** choosing a different folder and pressing Start continues the same model on the
  new videos (`--resume`), which is what you want when adding videos. A **Start over** button (shown only when a
  checkpoint exists) deletes `<work>\` after a confirm, then starts fresh.
- **Train more (the user's choice, 2026-11-23):** after a run finished, Start reads **Train more**: another 100000
  steps on the same model, on the folder chosen now, with the learning-rate schedule started fresh for those
  steps (warm-up and cosine over the new 100000) while the weights carry on. train.py gets a flag for this,
  `--more N` with `--resume` (my call on the name): total steps become done + N, and the schedule runs over the new
  N only. Without the flag, resume behaves as today.
- **Batch size from the GPU's memory (the user's choice, 2026-11-23):** train.py picks its batch size from the
  card's total VRAM (`torch.cuda.get_device_properties(0).total_memory`) when `--batch` isn't given: 8 at 16 GB
  (today's default, tuned on an RTX 5080), scaled down with less memory, at least 1. The table is calibrated by the
  builder on this RTX 5080 (measure peak memory at batch 8, size the rest from it with headroom for a video playing
  next to it). If a step still runs out of memory, train.py halves the batch, says so in its output and
  `training.json`, and goes on from that step (no restart); at batch 1 it fails with the reason "not enough GPU
  memory". The command line's `--batch` still overrides all of this.

### Which runs the tab controls (my call)

- The tab **shows** any run's progress (also a developer's `py train.py --out D:\X\vsr.pt`), as today.
- It **controls** only its own run: Stop, Use this model and the automatic export act only when
  `training.json`'s `out` is `<work>\vsr.pt`. A command-line run shows "started from the command line" and no
  buttons, so a developer's own `vsr.onnx` is never overwritten and their run never stopped from the tab.

### Failures (my call)

- train.py writes `state: "failed"` with a short `reason` to `training.json` when it exits on an error (out of GPU
  memory, no usable videos, a crash), from a top-level `try/except` around `main()`. The tab shows the reason and
  where `train.log` is. Without that, a failed run looks like "training" until it goes stale.
- If the child process exits without writing `finished`/`stopped`/`failed`, the player marks it failed itself
  ("training stopped unexpectedly, see train.log").

### Add-on updates (my call)

- The add-on folder holds `version.txt` (one number, raised in the source whenever any pin changes). When a player
  update needs a newer add-on, the tab says "the Training add-on needs an update (3.5 GB)" and Start offers the
  download; the old folder is replaced the same `.new` + rename way, or removed at the next start when its
  `python.exe` is running (TensorRT's pending-removal pattern). No per-file patching.

### Using the model

- **Use this model** (enabled once `vsr_best.pt` exists), and automatically when `training.json` says `finished`:
  run `python.exe export.py <work>\vsr_best.pt` with output into `<work>\`, then copy `vsr_best.onnx` → `vsr.onnx`
  and `vsr_best.ref` → `vsr.ref` into `%LOCALAPPDATA%\Open Player\` through temp + rename, then reload the upscalers
  (the same path as choosing a TensorRT folder: `reopen(...)`). If the TensorRT add-on is missing, the tab says
  Pixel Flow also needs it and offers its Download.
- Exporting while training runs is fine (it reads the saved `vsr_best.pt`, not the running model).

### Tab states (text sketch)

```
TRAINING
Videos   D:\Torrent                         [Choose videos folder...]
         159 videos
[Start]  [Use this model]  [Start over]          Add-on: 3.5 GB  [Remove]
-- while training --
Training   step 42,000 / 100,000   loss 0.031   +1.8 dB vs bilinear   about 5 h left
suggest: ok
[Stop]  [Use this model]
```

## Left to the builder

- The exact pins (Python 3.13.x embeddable, torch 2.15 cu132 or the newest pair that trains and exits cleanly on
  3.13, numpy, onnx, onnxscript and each dependency wheel), and the FFmpeg build to download (an LGPL or GPL
  build from a stable, versioned URL; its licence goes into the add-on folder and THIRD-PARTY-NOTICES).
- Counting videos for the "159 videos" line (cheap: extensions only, on a background thread).
- How the tab detects "no NVIDIA GPU" (the player already knows from CUDA at start).
- Tests: the add-on's unpack/verify/refuse-outside-folder and `._pth` edit in `--uitest` (like `addon_selftest`),
  train.py's stop-file check in `tests/train_check.py`, the tab's states in the UI checks.
- Four bug-check agents at the end, one a security review (standing rule): downloads and hashes, the command line
  built from a user-picked path, the stop file, temp + rename for the model files.
