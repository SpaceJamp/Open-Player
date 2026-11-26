# Ready-made Pixel Flow, faster engine, no TensorRT add-on

Date: 2026-11-26. Status: DRAFT, not approved. Brainstorming paused mid-way (the user had to leave).

## Intent (from the user)

Training Pixel Flow is a hassle. Keep Pixel Flow, but make it faster to learn and more efficient, for people who
download the player, not only the maintainer. Also drop the 1.9 GB TensorRT add-on download on Windows.

## Decided so far

- Approach 1 (user's pick): ship a ready-made Pixel Flow, trained once by the maintainer on openly licensed footage
  (candidates: Blender open films such as Sintel and Tears of Steel, CC-BY; Xiph.org test sequences). The model is
  under 1 MB and goes in the installer. Training on your own videos becomes an optional short fine-tune from that
  model instead of a 9-hour run from scratch.
- Faster playback with 8-bit inference (INT8, RTX 20 and newer; FP8 on RTX 40/50), checked against fp16 quality
  before it is used. The user asked whether 8-bit wears the card more: no. Same frames per second, less work per
  frame, so less power and heat (inference; to be measured with --stats).
- Drop the TensorRT add-on.

## Rejected

- Replacing Pixel Flow with shader upscalers (FSR 1, NIS) or open neural models: the user chose to keep Pixel Flow.
- Only speeding up training, without a shipped model: every downloader would still have to train.
- Distilling from an open teacher such as Real-ESRGAN (approach 2): kept as a follow-up if the open-footage
  model looks soft.

## Open: how to run Pixel Flow without the TensorRT add-on

Not decided yet. Options under study:
1. TensorRT's small "lean" runtime shipped in the installer, with engines built once as version- and
   hardware-compatible. Smallest change. What NVIDIA's Engine Compatibility page (TensorRT 11.5, updated
   2026-11-18) says: a version-compatible engine carries its own copy of the lean runtime; both modes "can be
   slower"; hardware compatibility (AMPERE_PLUS, compute capability 8.0+) is stated only for Linux ARM SBSA, not
   x86_64 Linux or Windows; FP8 is named only for the same-GPU-generation mode. Inference: RTX 20 (Turing, 7.5)
   would be left out. Unknown: runtime size (PyPI page unreadable), dynamic shapes and INT8 in these modes. Risky
   until tested on x86_64.
2. Pixel Flow's layers written as the player's own CUDA kernels (Tensor Cores, fp16/int8): no library to ship,
   full control, but a lot of work and must match TensorRT's speed.
3. Vulkan compute (for example ncnn): small and could later run on AMD/Intel, but likely slower on NVIDIA and
   needs CUDA/Vulkan interop.

## Still to settle

- The pick above, then whether this is one project or two (shipped model; engine change) and their order.
- Which open footage, and its licence terms for redistributing a model trained on it.
- Fine-tune length and defaults in train.py and the in-player training.

## Update 2026-11-26 (continued in a second session)

- **Linux too (the user):** everything must work on Linux as well, and Linux users get a download with updates
  like Windows (not only build-from-source).
- **Measured download sizes** (pypi.nvidia.com, 11.5.0.63 / TRT-RTX EP 0.5.0):

  | | Windows | Linux x86_64 |
  |---|---|---|
  | full TensorRT (today's add-on) | 1.9 GB | 3.6 GB |
  | TensorRT lean runtime | 13 MB | 44 MB |
  | TensorRT for RTX (via onnxruntime-ep-nv-tensorrt-rtx) | 106 MB | 140 MB |

- **New option 4: TensorRT for RTX** — NVIDIA's TensorRT for apps shipped to consumer RTX PCs: builds the engine
  on the user's GPU in seconds (no prebuilt engines, no compatibility flags), all RTX cards, Windows and Linux.
  Proposed pick (not yet decided), pending a throwaway spike on the RTX 5080: Pixel Flow standard 1080p -> 4K
  through TensorRT for RTX and the lean runtime vs today's TensorRT (speed, output difference).
- **Three projects, in this order (proposed):** 1. the engine (TensorRT for RTX, Windows + Linux); 2. the shipped
  ready-made Pixel Flow (footage, licence, fine-tune); 3. a Linux download with updates (likely an AppImage),
  after 1 removes the 3.6 GB add-on.
- **Windows first (the user, 2026-11-26):** optimise for Windows now; Linux (support + download) stays a
  requirement for later, so choices must not rule it out, but the spike and project 1 are Windows-only.

### Engine spike, RTX 5080, Windows (2026-11-26; throwaway, scratch only)

Pixel Flow standard (vsr_best.pt from the in-player run, fp16 ONNX, strongly typed), 1080p -> 4K, median per frame:

| engine | download | ms/frame | first build on the user's PC | catch |
|---|---|---|---|---|
| full TensorRT 11.5 (today) | 1.9 GB | 22.9 | 26 s | the add-on |
| lean runtime, version-compatible engine | 13 MB | 22.8 | none | engine must be prebuilt per GPU generation, on that GPU |
| lean, + hardware-compatible (ampere+) | 13 MB | 25.9 | none | works on Windows here; unproven on other GPUs; no RTX 20 |
| TensorRT for RTX 1.7 (ORT EP 0.5.0) | 106 MB | 28.5 (CUDA graphs: same) | 1.7-2.8 s | 25% slower than full TensorRT |

TensorRT for RTX output vs ONNX Runtime CPU (fp16, 96x128): mean abs diff 0.004, max 0.058.
Reading: the lean route needs hardware the maintainer doesn't have; TensorRT for RTX is practical but standard
drops under 30 fps at 1080p -> 4K; fast (9-13 ms on full TensorRT) or 8-bit would restore it (untested on TRT-RTX).

### Decided (the user, 2026-11-26): engine

- **TensorRT for RTX by default; the full TensorRT add-on stays as an optional speed boost** (25% faster here).
  When the add-on is present the player uses it, else TensorRT for RTX.
- My call (builder detail): drive TensorRT for RTX through its own C++ API (its interfaces follow TensorRT's, so
  vsr.cpp's engine code ports closely), not through ONNX Runtime (one runtime fewer to ship).
