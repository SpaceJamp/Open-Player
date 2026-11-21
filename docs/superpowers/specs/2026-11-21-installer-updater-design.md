# Windows installer, updater and TensorRT add-on

Date: 2026-11-21. Status: approved by the user, 2026-11-21.

## Intent

Open Player is for consumers who download it from GitHub (`SpaceJamp/Open-Player`, public, AGPL-3.0). They get a
normal Windows installer (Inno Setup) and the player keeps itself up to date (built-in updater, approach A).
**No Pixel Flow model ships, ever**: every user trains their own (README says so prominently). People who train
one get TensorRT through an optional add-on download from NVIDIA.

The user's standing rule applies throughout: no vulnerabilities; every finished piece gets 4 bug checks, one of
them a security review, and its gaps are closed.

**Success:** a consumer on any NVIDIA RTX PC downloads one small installer, installs without admin rights, plays
video with NVIDIA RTX VSR at once, gets later versions through a verified one-click update, and can add TensorRT
from Settings if they train their own Pixel Flow.

## Goals, non-goals, anti-goals

- Goals: Inno Setup installer and uninstaller; RTX VSR without a model; the player starting without TensorRT;
  the TensorRT add-on; the in-player updater with signature checks; a release script; the update key generated
  and backed up encrypted before the user's PC reset.
- Non-goals: Linux packages and a Linux updater (Linux users build from source); Authenticode code signing
  (SmartScreen will warn "unknown publisher" once; can be added later); delta updates; any model download.
- Anti-goals: anything that lets a third party run code on users' PCs (an unsigned or unverified update, a
  downloaded DLL not checked against a pinned hash, a private key in a repo unencrypted); silent network traffic the
  user can't turn off; NVIDIA's SDK files in the public repo; an installer that needs admin rights.

## Approaches considered (updater)

A. **Built into the player** (chosen): checks GitHub Releases, verifies an ECDSA signature with the user's own key,
   runs the new installer. One program, no background task.
B. A separate updater program: a second binary to build and secure.
C. WinGet only: no in-app updates.

## 1. Player changes

### RTX VSR without a model

Today RTX VSR is sized from Pixel Flow's plan and opened only when a model loaded (`main.cpp`: `rtx_tried =
have_vsr && want_rtx`). New: the screen plan uses Pixel Flow's factor when a model loads, else **2x**, with the same
"only upscale video smaller than the screen; shrink first when 2x overshoots" rules. RTX VSR opens whenever the
plan upscales, model or not. With no model and no RTX VSR, the video plays without upscaling (as today). The
settings card offers RTX/Off without a model; Pixel Flow is disabled with "no Pixel Flow model (vsr.onnx); train
your own" or "needs the TensorRT add-on". README's notice is updated to say RTX VSR works without a model.

### Starting without TensorRT

TensorRT's DLLs are delay-loaded (MSVC `/DELAYLOAD:nvinfer_11.dll /DELAYLOAD:nvonnxparser_11.dll`). Before the
first TensorRT call the player looks for TensorRT, in order: the folder set in Settings ("use my own TensorRT
11.x"), then the add-on folder `%LOCALAPPDATA%\Open Player\tensorrt\`, then next to the exe (developer builds). It
adds that folder with `AddDllDirectory` and checks `LoadLibraryExW(..., LOAD_LIBRARY_SEARCH_USER_DIRS)` succeeds
before touching TensorRT, so a missing DLL is a clean "Pixel Flow needs the TensorRT add-on", never a crash. The log
records which TensorRT version loaded (`getInferLibVersion`). Major version must be 11; other majors are refused
with a message. Cached `.engine` files already rebuild when TensorRT changes.

### TensorRT add-on

- Settings > Upscaling: "Pixel Flow needs TensorRT (1.9 GB download from NVIDIA)" with Download / Use my own
  folder / Remove.
- Download: `tensorrt_cu13_libs-11.5.0.63-py3-none-win_amd64.whl` from `https://pypi.nvidia.com/...` (NVIDIA's
  own package index, no login), over HTTPS (WinHTTP), into `%LOCALAPPDATA%\Open Player\tensorrt\download.whl`, with
  progress in the card and Cancel. The SHA-256 is **pinned in the player's source**
  (`d656f2e523e822993915ef8d7b70557afab69182f19315ddc8c326eacef2d645`); a mismatch deletes the file and reports it.
  The URL comes from the player's source too, never from a server response.
- Unpack: Windows' own `%SystemRoot%\System32\tar.exe` (by absolute path) extracts only `tensorrt_libs/*.dll` and
  the wheel's `LICENSE.txt` into the add-on folder; the wheel is then deleted. About 2.3 GB on disk. Files needed:
  `nvinfer_11.dll`, `nvonnxparser_11.dll`, `nvinfer_plugin_11.dll`, `nvinfer_builder_resource_*_11.dll` (sm75 to
  sm120 and ptx).
- User-supplied TensorRT folders are the user's choice and are not hash-checked; the log notes the version.
- Uninstalling Open Player offers to remove the add-on folder.

### Updater

- Setting "Check for updates" (Settings > About), default on. With it on, at most once a day at start, a
  background thread asks `https://api.github.com/repos/SpaceJamp/Open-Player/releases/latest` (generic User-Agent
  `Open-Player-Updater`, nothing about the user sent). Only installed copies check (the installer writes
  `installed.txt` next to the exe); builds from source never do.
- A release carries three assets: `OpenPlayer-Setup-<build>.exe`, `update.json` and `update.json.sig`.
  `update.json` = `{"build": <number>, "file": "OpenPlayer-Setup-<build>.exe", "sha256": "<hex>", "notes": "..."}`.
  The signature is ECDSA P-256 over SHA-256 of `update.json`'s exact bytes, raw r||s (64 bytes), base64.
- The player verifies the signature with the **public key compiled into it** (Windows CNG: `BCryptImportKeyPair`
  with `BCRYPT_ECCPUBLIC_BLOB`, `BCryptVerifySignature`), then requires `build` > its own build number (no
  downgrades, no replays), then shows "Update available: build N" in About and once as a notice.
- "Update": downloads the installer to `%TEMP%\open-player-update\`, checks its SHA-256 against the signed
  `update.json`, saves settings, starts it with `/SILENT /SUPPRESSMSGBOXES /NORESTART` and exits; the installer
  relaunches the player. Any failed check deletes the download, logs why and shows "Update failed: ...".
- Download hosts are limited to the list in call 2 below (redirects
  elsewhere are refused); sizes are capped (`update.json` 64 KB, installer 500 MB).

### Versions in Settings > About (added by the user, 2026-11-21)

About lists, under the build number:
- **TensorRT:** the version actually loaded (`getInferLibVersion()`, e.g. `11.5.0`) and where it came from
  (add-on / your folder / next to the program), or "not installed (Pixel Flow needs the TensorRT add-on)".
- **NVIDIA RTX VSR:** the version of the `nvngx_vsr.dll` in use, read with Windows' standard file-version API
  (`GetFileVersionInfoW`, as Explorer's Properties > Details shows it; nothing inside NVIDIA's binary is
  inspected), plus the RTX Video SDK API version the player was built against (`NVSDK_NGX_VERSION_API_MACRO`,
  currently 1.5.0). "not available on this GPU or driver" when RTX VSR can't open; "not in this build" on Linux.
- **NVIDIA driver:** the CUDA version the driver supports (`cuDriverGetVersion`, e.g. `13.5`), since RTX VSR and
  TensorRT both depend on it.
The same lines go to the log at start and to `--check`.

## 2. Installer (Inno Setup)

- `installer/open-player.iss`: per-user (`PrivilegesRequired=lowest`), into `{localappdata}\Programs\Open Player`,
  fixed `AppId`, version = build number; Start-menu shortcut; optional desktop shortcut; optional "Open with"
  registration for common video types (not taking over as default); uninstaller; `installed.txt` marker.
- Contents: `open-player.exe`, the FFmpeg/SDL/dav1d DLLs, the VC++ runtime DLLs (app-local), `nvngx_vsr.dll`,
  `assets/`, `LICENSE`, `LICENSE-EXCEPTION`, a third-party notices file (NVIDIA credit and terms reference, FFmpeg
  LGPL, SDL zlib, Dear ImGui, fonts), and a link to the source. No TensorRT, no model. Roughly 60-80 MB.
- `build-windows.bat` stays the developer build; `installer/build-installer.bat` builds the release
  (`iscc`, installed by `setup-windows.ps1` via `winget install JRSoftware.InnoSetup`).

## 3. The update key

- `tools/make-update-key.ps1` (run once, now, before the PC reset): generates an ECDSA P-256 key with the
  OpenSSL that ships with Git for Windows, writes the **encrypted** private key `open-player-update-key.pem`
  (AES-256, passphrase typed by the user in their own terminal; never on the command line, never seen by Claude)
  and the public key as C source (`player/update_pubkey.h`, committed to the public repo).
- Backup: the encrypted key is pushed to a new **private** repo `SpaceJamp/Open-Player-Keys`; the passphrase goes
  in the user's password manager, never on GitHub. `*.pem` is git-ignored in the public repo.
- Losing key and passphrase means installed copies can't verify future updates; users would have to reinstall.
- `tools/sign-release.ps1`: hashes the installer, writes `update.json`, signs it (OpenSSL, asks for the passphrase),
  converts the DER signature to r||s base64, and verifies it with the public key before finishing.
- `tools/release.ps1`: build, installer, sign, then `gh release create v<build>` with the three assets (run by the
  user; publishing is theirs to confirm).

## Security review focus

The 4 bug checks include a security review of: signature verification (bad/missing/old signatures, a valid
`update.json` with a different installer, a replayed older release), redirects and hosts, size caps, the add-on's
pinned hash and extraction (path traversal in the zip via `tar.exe`: only `tensorrt_libs/*.dll` names extracted,
into a folder we create), DLL search order (no DLL loaded from the current folder or `PATH` before ours), the
installer's per-user permissions, and the key handling.

## Testing

- `--uitest`: `update.json` parsing; signature verification with a test key pair (good, tampered byte, wrong key,
  older build, mismatched installer hash); host allow-list; version compare; TensorRT folder search order logic.
- On this PC: the installer builds, installs into a temp folder (`/DIR=`), the installed player starts without
  TensorRT and plays with RTX VSR without a model, uninstalls cleanly; the add-on downloads, verifies and unpacks
  once (1.9 GB, run once); a signed test release served from a local folder updates an installed copy.
- By hand, by the user: the real GitHub release flow, the first SmartScreen prompt.

## Calls made after the builder check (the user may overrule)

1. **Local update test without weakening the shipped player:** a CMake option `OPEN_PLAYER_UPDATE_TEST` (off by
   default) compiles in a *test* public key and also allows `http://127.0.0.1` as a host. Such builds say "TEST
   BUILD" in About, and `release.ps1` refuses to package one. There is no runtime override (flag or environment
   variable) in any build.
2. **Download hosts** (checked 2026-11-21: GitHub now redirects release downloads to
   `release-assets.githubusercontent.com`): `api.github.com`, `github.com`, `release-assets.githubusercontent.com`
   and `objects.githubusercontent.com`, HTTPS only. The plan re-checks this against a real release before shipping.
3. **TensorRT add-on needs nothing else:** tested on this PC with every CUDA/TensorRT folder removed from `PATH`:
   `--selftest` built a new engine and passed (51.2 dB) with only TensorRT's own DLLs and the NVIDIA driver. The
   wheel's metadata lists no dependencies.
4. **Where an installed copy finds `vsr.onnx`:** `%LOCALAPPDATA%\Open Player\` (engines cached there too), so
   upgrades and uninstalls never touch a user's model, and "Open with" (current folder = the video's) doesn't
   matter. Lookup order: `--model`, then that folder, then next to the exe (developer builds); installed copies
   skip the current folder. Settings and the README say where to put the file. The uninstaller asks before
   removing that folder.
5. **The key comes first** (the PC is reset today). The plan's first task writes `make-update-key.ps1`; the user
   runs it in their own terminal and types the passphrase; Claude then commits the *encrypted* key to the private
   `SpaceJamp/Open-Player-Keys` (the user asked for this backup) and the public key header to the public repo.
6. **Build number = commits on `main`:** `release.ps1` refuses a modified tree, `build unknown`, or any branch but
   `main`; `main`'s history is never rewritten (the README's release notes say so). A player whose build is
   `unknown` never offers updates.

## Left to the builder

- The exact Inno Setup wizard pages and wording; icons.
- How the add-on's progress is drawn (matching the card).
- Whether `release.ps1` and `sign-release.ps1` are one script.
