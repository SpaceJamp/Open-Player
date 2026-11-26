# Pixel Flow on TensorRT for RTX (project 1 of the ready-made Pixel Flow plan)

Date: 2026-11-26. Status: draft, reviewed by a builder check, waiting for the user's approval. Windows only for now; Linux later (must not be
ruled out). Parent draft: `2026-11-26-pixel-flow-ready-made-design.md` (engine spike results there).

## Intent

Pixel Flow today needs NVIDIA's full TensorRT, a 1.9 GB add-on, and builds its engine for up to a minute per new
resolution. The user wants Pixel Flow to work for people who download the player without that weight. Project 2
(a shipped ready-made model) builds on this.

## Decisions (the user, 2026-11-26)

- **TensorRT for RTX is the default engine**; the **full TensorRT add-on stays as an optional speed boost**.
  When both are there, the full one is used (22.9 vs 28.5 ms/frame for standard 1080p -> 4K on an RTX 5080).
- **TensorRT for RTX is downloaded on first use** (about 106 MB), not bundled in the installer.

## Goals

- A **TensorRT for RTX add-on**, like the TensorRT add-on: pinned file, SHA-256 in the source, download with
  progress and Cancel, unpacked into `%LOCALAPPDATA%\Open Player\tensorrt-rtx\`, Remove.
- **First use:** choosing Pixel Flow (Settings > General > Upscaler) with neither engine present offers the
  download (one click, size shown); the RTX one is offered first, the full add-on stays under "Faster (1.9 GB)".
- **Pixel Flow runs on TensorRT for RTX**: engine built on the user's GPU (about 2 s), cached per resolution like
  today's `.engine` files, playback held while it builds (existing behaviour).
- **Engine choice:** the full add-on (or a user's own TensorRT folder) when present, else TensorRT for RTX, else
  Pixel Flow unavailable with the reason. About and `--check` say which engine runs.

## Non-goals

- The shipped model (project 2), 8-bit inference, Linux (later), ONNX Runtime (not shipped).
- Non-RTX NVIDIA cards (GTX 16 and older): TensorRT for RTX doesn't run there; they keep the full add-on path.

## Anti-goals

- Pixel Flow output visibly different between the two engines (the spike: mean 0.004 on fp16).
- Breaking today's users who have the full add-on (it keeps working, still preferred).
- Unpinned downloads, or anything loaded before its SHA-256 matched (same rules as the TensorRT add-on).

## Approaches

- **A. Own C++ API of TensorRT for RTX (my call).** Its interfaces follow TensorRT's (`createInferBuilder`,
  `buildSerializedNetwork`, `IRuntime`, `enqueueV3`), so `vsr.cpp`'s builder/runtime code gets a second backend
  with little new code. Needs its headers (from NVIDIA's TensorRT-RTX SDK or its open headers) at build time.
- B. Through ONNX Runtime's plugin (what the spike used): no headers needed, but ships onnxruntime.dll and a second
  runtime model. Dropped.

## Design

- **Download:** `onnxruntime_ep_nv_tensorrt_rtx_cu13-0.5.0-py3-none-win_amd64.whl` from pypi.nvidia.com (the
  package that carries `tensorrt_rtx_1_7.dll`, `tensorrt_onnxparser_rtx_1_7.dll`); only those DLLs and NVIDIA's
  licence/readme are kept (the ORT plugin DLL is dropped). Reuses `addon.cpp`'s download/verify/`tar_extract`
  pattern (`.new` + rename, refuse `..`).
- **Loading:** delay-loaded like TensorRT today (`trt_loader`): a second loader for the RTX DLLs; `TrtFrom` gains
  `RtxAddOn`.
- **vsr.cpp:** an engine abstraction with two implementations (full TensorRT, TensorRT for RTX) behind the same
  `Vsr` interface; the RTX one builds the strongly typed fp16 network from the same ONNX and caches its plan as
  `<model>.<w>x<h>.rtx.engine`; a runtime cache file for TensorRT for RTX's just-in-time kernels next to it.
- **UI:** Upscaling section shows "Pixel Flow engine: TensorRT for RTX 1.7 (add-on)" / "TensorRT 11.5 (add-on)";
  buttons Download (106 MB) / Faster: full TensorRT (1.9 GB) / Remove / Choose folder...
- **Installer/uninstaller:** nothing shipped; the uninstall prompt already covers `%LOCALAPPDATA%\Open Player`.

## Left to the builder

- Getting TensorRT for RTX's C++ headers (SDK download or open headers) and how the build finds them
  (`-DTENSORRT_RTX_ROOT`), like `TENSORRT_ROOT`.
- Whether `enqueueV3`/IO tensor names match exactly; JIT runtime-cache API names.
- Tests: add-on unpack/verify/refuse in `--uitest`; engine choice order; `--selftest` against the PyTorch
  reference on both engines; speed check in `--stats`.
- Four bug-check agents at the end, one a security review.

## Builder check (answered)

- **The wheel is self-contained:** the spike re-run with PATH = Windows folders only (no CUDA toolkit) built the
  engine in 2.8 s and ran at 30.0 ms/frame with identical output. Kept files: `tensorrt_rtx_1_7.dll` (229 MB
  unpacked), `tensorrt_onnxparser_rtx_1_7.dll`, `cudart64_13.dll`, `tensorrt_plugins.dll` only if the build needs
  it (builder checks), NVIDIA's README/licence. The library names cuDNN/NCCL/nvrtc-builtins but didn't need them.

## Decided (the user, 2026-11-26): licence

- "NVIDIA TensorRT for RTX" added to LICENSE-EXCEPTION (the AGPL section 7 linking permission), so Approach A
  (its own C++ API) stands.

## Decided (the user, 2026-11-26): when Pixel Flow can't keep up

- **Fall back to NVIDIA RTX VSR automatically.** Details (my calls):
  - Trigger: over a 10-second window of playback (not paused, not seeking), more than 5% of frames dropped or
    Pixel Flow's own time per frame above the video's frame interval. Measured with the numbers `--stats` already
    collects. Not during the first 5 s after start, a seek or an engine build.
  - Action: switch this video to RTX VSR (the same path as choosing it in Settings), notice "Pixel Flow can't keep
    up here: using NVIDIA RTX VSR". The saved upscaler setting is not changed; the next video tries Pixel Flow again.
  - No RTX VSR on this PC: keep Pixel Flow and show the notice "Pixel Flow can't keep up here" once.
  - Applies to either engine (TensorRT for RTX or the full add-on).
