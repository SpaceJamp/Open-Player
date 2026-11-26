# TensorRT for RTX Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pixel Flow runs on TensorRT for RTX (a 106 MB add-on downloaded on first use), with the full TensorRT
add-on kept as an optional faster engine and an automatic fall-back to NVIDIA RTX VSR when Pixel Flow can't keep up.

**Architecture:** `addon.cpp` gains a second pinned add-on (the TensorRT for RTX DLLs, unpacked from NVIDIA's
wheel). `trt_loader` delay-loads either engine and a pure `choose_engine` picks one. `vsr.cpp` gets a second
backend on TensorRT for RTX's C++ API behind the existing `Vsr` interface. A pure `should_fall_back` decides the
switch to RTX VSR from the stats the player already keeps. Settings > Upscaling offers the downloads.

**Tech Stack:** C++20, TensorRT 11.5 (full), TensorRT for RTX 1.7 C++ SDK (headers + import libs), CUDA, ImGui.

**Spec:** `docs/superpowers/specs/2026-11-26-trt-rtx-engine-design.md` (parent: `2026-11-26-pixel-flow-ready-made-design.md`)

## Global Constraints

- Windows only now; keep `#ifdef`s so the Linux build still compiles (Linux comes later).
- Add-on pin: `https://pypi.nvidia.com/onnxruntime-ep-nv-tensorrt-rtx-cu13/onnxruntime_ep_nv_tensorrt_rtx_cu13-0.5.0-py3-none-win_amd64.whl`,
  SHA-256 `609b19f7236c8522ef005a9f763003a78f492e5d871afad39fe1b247760eb3c7`, size 110875664, host
  `pypi.nvidia.com`. Kept: `tensorrt_rtx_1_7.dll`, `tensorrt_onnxparser_rtx_1_7.dll`, `TRTRTX_README.txt`,
  `TRTRTX_Acknowledgements.txt` (+ `cudart64_13.dll`/`tensorrt_plugins.dll` only if Task 4 proves them needed).
  Folder `%LOCALAPPDATA%\Open Player\tensorrt-rtx\`. Same rules as the TensorRT add-on (`.new` + rename, hash
  before unpack, `tar.exe` exit code checked).
- Engine order: full TensorRT (user folder, add-on, next to exe) → TensorRT for RTX add-on → none.
- Fall-back: 10 s window of playback, >5% frames dropped or Pixel Flow time/frame > frame interval; ignored in
  the first 5 s after start/seek/engine build; switches this video to RTX VSR (saved setting unchanged), notice
  "Pixel Flow can't keep up here: using NVIDIA RTX VSR"; without RTX VSR a one-time notice "Pixel Flow can't keep
  up here".
- The player must still build and run with neither SDK installed (Pixel Flow then unavailable, as today without
  TensorRT).
- Commit and push after every task on branch `trt-rtx`; 4 bug-check agents at the end (one security).

## Review Focus

- The RTX add-on present but the GPU isn't an RTX card (GTX 16): Pixel Flow says why, no crash (Task 3 test of
  `choose_engine` with `rtx_ok=false`).
- Both engines present: the full one wins, About says so (Task 3 test).
- A seek or a window resize right after start: no false fall-back (Task 6 test with the 5 s grace).
- The add-on download cancelled or the disk full: no half folder (Task 2 test, same as the Training add-on's).
- An engine cache built by one engine read by the other: never (Task 4: separate `.rtx.engine` names, test).

---

### Task 1: Build setup for the TensorRT for RTX SDK

**Precondition (the user):** download the TensorRT for RTX 1.7 SDK for Windows (zip, NVIDIA login) from
https://developer.nvidia.com/tensorrt-rtx and unzip it to `C:\TensorRT-RTX` (the folder holding `include\` and
`lib\`).

**Files:** Modify `player/CMakeLists.txt`, `setup-windows.ps1`, `build-windows.bat` (comment only)

- [ ] **Step 1:** CMake: `TENSORRT_RTX_ROOT` cache path (default the folder under `C:/TensorRT-RTX` that has
  `include/NvInfer.h` or the SDK's own header name); if found: `HAVE_TRT_RTX`, include dir (SYSTEM), import libs,
  `/DELAYLOAD` of its DLLs, `delayimp`; else a STATUS line "building without TensorRT for RTX".
- [ ] **Step 2:** `setup-windows.ps1`: an optional check line for the SDK with the download hint (like the RTX
  Video SDK's).
- [ ] **Step 3:** `build-windows.bat` and `--uitest` pass with and without the SDK folder present (rename it to
  test). Commit `build: optional TensorRT for RTX SDK`.

### Task 2: The TensorRT for RTX add-on (download, verify, unpack)

**Files:** Modify `player/addon.{h,cpp}`, `player/trt_loader.{h,cpp}` (`rtx_addon_dir()`)

**Interfaces (Produces):** `bool extract_trt_rtx(const std::string& wheel, const std::string& dir, std::string& error);`
`bool install_trt_rtx_addon(progress, cancel, error);` `void remove_trt_rtx_addon(bool in_use);`
`void finish_pending_rtx_removal();` `std::string rtx_addon_dir();` constants `RTX_WHEEL_URL/SHA256/SIZE`.

- [ ] **Step 1: Tests in `addon_selftest`:** a test wheel with `onnxruntime_ep_nv_tensorrt_rtx/tensorrt_rtx_1_7.dll`,
  `.../tensorrt_onnxparser_rtx_1_7.dll`, `.../TRTRTX_README.txt`, `.../onnxruntime_providers_nv_tensorrt_rtx.dll`,
  and `../evil.dll` → the two engine DLLs and the README kept, the ORT plugin DLL dropped, nothing outside; a wheel
  without `tensorrt_rtx_1_7.dll` refused with the old add-on kept; a failure leaves no `tensorrt-rtx.new`.
- [ ] **Step 2:** `--uitest` → FAIL. **Step 3:** implement (pattern of `extract_tensorrt`/`install_tensorrt_addon`;
  4.5 GB free-space check → 0.5 GB). **Step 4:** `--uitest` → pass. **Step 5:** commit.

### Task 3: Loading and choosing the engine

**Files:** Modify `player/trt_loader.{h,cpp}`, `player/main.cpp` (About/`--check` lines)

**Interfaces (Produces):** `enum class Engine { None, Full, Rtx };`
`Engine choose_engine(bool full_ok, bool rtx_ok);` `bool load_trt_rtx(std::string& version, std::string& why_not);`
(loads from `rtx_addon_dir()` with `SetDllDirectoryW`, checks the major version is 1) and
`std::string engine_line(Engine e, TrtFrom full_from, const std::string& full_ver, const std::string& rtx_ver)`
→ "Pixel Flow engine: TensorRT 11.5.0 (add-on)" / "TensorRT for RTX 1.7.0 (add-on)" / "none".

- [ ] **Step 1: Tests:** `choose_engine(true,true)==Full`, `(false,true)==Rtx`, `(true,false)==Full`,
  `(false,false)==None`; `engine_line` texts; `load_trt_rtx` with an empty `rtx_addon_dir` (XDG_DATA_HOME temp) →
  false, `why_not` "not installed".
- [ ] **Step 2–4:** RED, implement, GREEN. **Step 5:** commit.

### Task 4: Pixel Flow on TensorRT for RTX (`vsr.cpp`)

**Files:** Modify `player/vsr.{h,cpp}`, `player/main.cpp` (pass the chosen engine)

- [ ] **Step 1:** Split today's builder/runtime calls behind two small backends (`FullTrt`, `RtxTrt`) chosen by
  `Engine`; RTX: same strongly typed fp16 network from the ONNX through its ONNX parser, plan cached as
  `<model>.<w>x<h>.rtx.engine` (full keeps `.engine`/`.win.engine`), its runtime (JIT) cache file next to it.
- [ ] **Step 2: Tests:** the cache-name function `engine_cache_name(onnx, w, h, Engine)` gives distinct names for
  the two engines (unit, in `--uitest`); `open-player --selftest vsr.onnx` passes on both engines (force with
  `--engine full|rtx`, a new developer flag), PyTorch reference within today's tolerance.
- [ ] **Step 3:** measure with `--stats`: standard 1080p → 4K on the 5080 within 10% of the spike (28.5 ms);
  confirm which DLLs are really needed (remove `cudart64_13.dll`/`tensorrt_plugins.dll` from the add-on folder and
  run `--selftest`), update the Task 2 keep-list. Commit.

### Task 5: Settings > Upscaling: first use and the optional full add-on

**Files:** Modify `player/ui.{h,cpp}`, `player/main.cpp`

- [ ] **Step 1: UI checks:** with no engine and Pixel Flow chosen, the Upscaling section shows "Download (106 MB)"
  and "Faster: full TensorRT (1.9 GB)"; RTX add-on present: "Remove" + "Faster: full TensorRT (1.9 GB)" + status
  line from `engine_line`; downloading: progress + Cancel; full present: today's buttons.
- [ ] **Step 2–4:** RED, implement (new `UiAction`s `DownloadTrtRtx`, `CancelTrtRtx`, `RemoveTrtRtx`; download on a
  thread like TensorRT's; reopen after install), GREEN. **Step 5:** commit.

### Task 6: Fall back to RTX VSR when Pixel Flow can't keep up

**Files:** Modify `player/main.cpp`, `player/ui.cpp` (only the notice text), a pure helper in `player/diag.h` or a
new small header

**Interfaces (Produces):** `struct KeepUp { double since_start, window_s; int frames, dropped; double pxf_ms,
frame_ms; }; bool should_fall_back(const KeepUp& k);`

- [ ] **Step 1: Tests:** `since_start` 4 → false; 10 s window with 6% dropped → true; 4% dropped and pxf 20 ms at
  33 ms frames → false; 0% dropped but pxf 40 ms at 33 ms → true; `window_s` 5 → false.
- [ ] **Step 2–4:** RED, implement + wire (counters reset at start/seek/engine build; switch via the existing
  upscaler-change path, not saved), GREEN. **Step 5:** commit.

### Task 7: README rewrite, end-to-end, bug checks

**Files:** `README.md` (complete rewrite, the user's request), `installer/THIRD-PARTY-NOTICES.txt`

- [ ] **Step 1:** Rewrite README.md from scratch: users first (install, playing, captions, Pixel Flow with the
  TensorRT for RTX download and the optional full add-on, the fall-back, training in the player, privacy and
  updates, limits), then developers (build incl. both SDKs, command line, training from the command line, tests,
  releasing, license incl. the TensorRT for RTX exception). Every fact checked against the code.
- [ ] **Step 2:** Notices: the TensorRT for RTX add-on (downloaded from NVIDIA, its licence in its folder).
- [ ] **Step 3:** Installer built; the user installs it; live check with desktop control: Pixel Flow chosen with
  no engine → Download → plays at about 28 ms/frame; full add-on present → used; fall-back seen by forcing a slow
  case (developer flag or a 4K → 8K file).
- [ ] **Step 4:** Commit, push; 4 bug-check agents (correctness; Windows/DLL loading; UI/threads; security:
  download pins, DLL search order); fix real findings with tests.
