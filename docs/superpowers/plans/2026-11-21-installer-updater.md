# Installer, Updater and TensorRT Add-on Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A consumer Windows installer (Inno Setup), a signed in-player updater, a TensorRT add-on downloaded from NVIDIA, RTX VSR without a model, and version lines in About.

**Architecture:** The player gains `player/update.{h,cpp}` (manifest parsing, signature and hash checks, host allow-list, download), `player/trt_loader.{h,cpp}` (finding and delay-loading TensorRT, version strings) and `player/addon.{h,cpp}` (the TensorRT add-on). Releases are built and signed by PowerShell scripts in `tools/`; the installer lives in `installer/`.

**Tech Stack:** C++20, Windows CNG (BCrypt: SHA-256, ECDSA P-256), WinHTTP, `%SystemRoot%\System32\tar.exe`, OpenSSL from Git for Windows (key and signing scripts only), Inno Setup 6, GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-11-21-installer-updater-design.md`

## Global Constraints

- No Pixel Flow model ships anywhere (installer, releases, add-on). README states it prominently.
- The update private key exists only encrypted (AES-256, passphrase typed by the user); never in the public repo; backup in private `SpaceJamp/Open-Player-Keys`. `*.pem` git-ignored.
- Signatures: ECDSA P-256 over SHA-256 of `update.json`'s exact bytes, raw r||s 64 bytes, base64, in `update.json.sig`.
- Hosts: `api.github.com`, `github.com`, `release-assets.githubusercontent.com`, `objects.githubusercontent.com`, HTTPS only; `http://127.0.0.1` only in `OPEN_PLAYER_UPDATE_TEST` builds. Caps: `update.json` 64 KB, installer 500 MB.
- TensorRT add-on: `tensorrt_cu13_libs-11.5.0.63-py3-none-win_amd64.whl`, SHA-256 `d656f2e523e822993915ef8d7b70557afab69182f19315ddc8c326eacef2d645`, from `https://pypi.nvidia.com/tensorrt-cu13-libs/`, URL and hash in source; only `tensorrt_libs/*.dll` and `LICENSE.txt` extracted to `%LOCALAPPDATA%\Open Player\tensorrt\`.
- TensorRT major must be 11. Installed copies find `vsr.onnx` in `%LOCALAPPDATA%\Open Player\`.
- Build number = commits on `main`; `build unknown` never offers updates; releases refuse modified trees, non-`main` branches and test builds.
- Per-user install, no admin. Linux build keeps compiling (updater, add-on and delay-load are Windows-only).
- No automated GUI input on the user's desktop; GUI runs go to screen 2 without input. Downloads of the 1.9 GB add-on and pushes to GitHub only with the user's OK in chat.

## Review Focus

1. A download cut off or corrupted (network drop, full disk) → the partial file is deleted, nothing is run or unpacked, a clear message. Test: Task 2 `verify_file_sha256` on a truncated file; Task 5 extraction into a full/readonly folder leaves no half add-on.
2. GitHub answers 403 (rate limit), 404 (no release yet) or junk JSON → no notice, one log line, next check tomorrow. Test: Task 2 `parse_update_json` on junk and on a release without the three assets.
3. A signed but older or replayed release, or a valid `update.json` paired with another installer → refused. Test: Task 2 cases.
4. A user profile path with non-ASCII characters (`C:\Users\Zoë`) → add-on, model lookup and update download all work. Test: Task 3 and Task 5 run their folder tests under a temp folder named `op-zoë`.
5. A zip entry with `../` or an absolute path in the add-on → never written outside the add-on folder. Test: Task 5 extraction of a test zip with `tensorrt_libs/../evil.dll`.

---

### Task 1: Update key and its encrypted backup (first: the PC is reset today)

**Files:** Create `tools/make-update-key.ps1`, `player/update_pubkey.h` (generated); Modify `.gitignore` (`*.pem`); in `C:\Users\Yeti\Documents\Open-Player-Keys`: add `open-player-update-key.pem`.

**Interfaces:** Produces `static const unsigned char UPDATE_PUBKEY[64]` (X||Y of the P-256 public key) in `player/update_pubkey.h`.

- [ ] **Step 1:** Write `tools/make-update-key.ps1 -Out <keys repo folder>`: uses `C:\Program Files\Git\usr\bin\openssl.exe`; refuses to overwrite an existing key; `genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -aes-256-cbc -out <Out>\open-player-update-key.pem` (OpenSSL prompts for the passphrase twice on the console, nothing on the command line); exports the public key (`pkey -pubout -outform DER`, last 64 bytes after the `04` prefix) into `player/update_pubkey.h`; then signs a random 32-byte test message and verifies it with the public key, printing `key OK`.
- [ ] **Step 2:** Open a terminal tab for the user with `powershell -ExecutionPolicy Bypass -File tools\make-update-key.ps1 -Out C:\Users\Yeti\Documents\Open-Player-Keys`; the user types the passphrase (twice, then once for the self-test). Expected: `key OK`.
- [ ] **Step 3:** Check the key file is encrypted: first line `-----BEGIN ENCRYPTED PRIVATE KEY-----`. Expected: yes; otherwise delete it and stop.
- [ ] **Step 4:** Commit the encrypted key in the keys repo and push (user approved the backup); commit `tools/make-update-key.ps1`, `player/update_pubkey.h`, `.gitignore` on `installer`. Remind the user to put the passphrase in their password manager.

### Task 2: Update logic (manifest, signature, hash, hosts)

**Files:** Create `player/update.h`, `player/update.cpp`; Modify `player/CMakeLists.txt` (add `update.cpp`, link `bcrypt` on WIN32), `player/ui.cpp` (`ui_selftest` calls `update_selftest()`).

**Interfaces (Produces):**
- `struct UpdateInfo { int build = -1; std::string file, sha256, notes; };`
- `bool parse_update_json(const std::string& text, UpdateInfo& out);` — exactly these keys; `file` must match `OpenPlayer-Setup-<build>.exe`; `sha256` 64 hex chars.
- `int build_number(const std::string& build_text);` — `"build 214 (4a30cf8)"` → 214; `-modified` or `unknown` → -1.
- `bool should_offer(int current, const UpdateInfo& u);` — `current > 0 && u.build > current`.
- `bool host_allowed(const std::string& url);` — Global Constraints list.
- `bool verify_signature(const unsigned char pubkey[64], const std::string& msg, const std::string& sig_b64);` (WIN32: BCrypt ECDSA P-256; elsewhere returns false)
- `std::string sha256_hex(const std::string& data);` `bool verify_file_sha256(const std::string& path, const std::string& hex);`
- `int update_selftest();`

- [ ] **Step 1: Failing tests** in `update_selftest()` with a test key pair generated once by `openssl` and pasted as constants (test public key, a signed `update.json`, its signature): good signature → true; one byte of `update.json` changed → false; signature by another key → false; `build_number` cases above; `should_offer(210, {214})` true, `(214, {214})` false, `(-1, {999})` false; `parse_update_json` rejects junk, missing keys, `file` with `..\` or `/`, short `sha256`; `host_allowed("https://release-assets.githubusercontent.com/x")` true, `("http://github.com/x")` false, `("https://github.com.evil.io/x")` false, `("https://evil.io/?github.com")` false; `sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"`; `verify_file_sha256` on a truncated copy → false.
- [ ] **Step 2:** Build (`build-dev`), run `--uitest`. Expected: compile/link failure.
- [ ] **Step 3:** Implement `player/update.{h,cpp}`. Host check parses scheme and host exactly (no substring match).
- [ ] **Step 4:** `--uitest` on Windows → `ui checks passed`; Linux build compiles and its `--uitest` passes (signature tests skipped there with a note).
- [ ] **Step 5:** Commit `feat(player): update manifest, signature and hash checks`.

### Task 3: TensorRT loading, RTX VSR without a model, versions in About

**Files:** Create `player/trt_loader.h`, `player/trt_loader.cpp`; Modify `player/CMakeLists.txt` (WIN32: `/DELAYLOAD:nvinfer_11.dll /DELAYLOAD:nvonnxparser_11.dll`, link `delayimp`; developer builds keep TensorRT's DLLs copied next to the exe), `player/main.cpp` (plan factor, RTX without model, model lookup, About/log/--check lines), `player/vsr.cpp` (called only when TensorRT loaded), `player/ui.{h,cpp}` (About lines, Pixel Flow unavailable reasons), `player/settings.{h,cpp}` (`trt_folder` key).

**Interfaces (Produces):**
- `enum class TrtFrom { None, UserFolder, AddOn, NextToExe };`
- `std::vector<std::string> tensorrt_candidates(const std::string& user_folder, const std::string& addon, const std::string& exe_dir);` — in that order, empty entries skipped.
- `TrtFrom load_tensorrt(const Settings&, std::string& version, std::string& why_not);` — WIN32: `AddDllDirectory` + `LoadLibraryExW(L"nvinfer_11.dll", nullptr, LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_SYSTEM32)`; checks major == 11 via `getInferLibVersion`; Linux: dynamic linking as today, returns `NextToExe`.
- `std::string addon_dir();` (`%LOCALAPPDATA%\Open Player\tensorrt`), `std::string user_data_dir();` (`%LOCALAPPDATA%\Open Player`), `bool installed_copy();` (`installed.txt` next to the exe).
- `std::string rtx_vsr_version();` (file version of the loaded `nvngx_vsr.dll` via `GetFileVersionInfoW`, `"not available on this GPU or driver"`, or `"not in this build"`), `std::string driver_cuda_version();` (`cuDriverGetVersion` → `"13.5"`).
- `std::string about_text(const std::string& build, const std::string& log, const std::vector<std::string>& versions);` (replaces the 2-argument one).
- `Settings` gains `std::string trt_folder;` (key `trt_folder=`).

- [ ] **Step 1: Failing tests:** `tensorrt_candidates("D:/my trt", "C:/a", "C:/b")` order; empty user folder skipped; `about_text` contains `TensorRT: 11.5.0 (add-on)` and `NVIDIA RTX VSR: ...` lines when given; settings round-trip `trt_folder` with a path under `op-zoë`.
- [ ] **Step 2:** Build, `--uitest`. Expected: compile failure.
- [ ] **Step 3: Implement.** Plan: `ms = have_model ? Vsr::model_scale(model) : 2`; RTX opens whenever the plan upscales (`worth_it`), independent of `have_vsr`; `us.vsr_unavailable` only when neither upscaler can run; Pixel Flow's segment disabled with "no Pixel Flow model (vsr.onnx): train your own" or "needs the TensorRT add-on". Model lookup: `--model`, `user_data_dir()`, next to exe; installed copies skip the current folder. Version lines go to the log at start, to `--check`, and to About.
- [ ] **Step 4: Runs** (screen 2, no input, temp `XDG_CONFIG_HOME`): copy `build-dev\open-player.exe` and its non-TensorRT DLLs to a temp folder without TensorRT and without `vsr.onnx` → log has `VSR: NVIDIA RTX Video Super Resolution` and `TensorRT: not installed`; `--check` lists the three version lines; normal `build-dev` run → `TensorRT: 11.5.0 (next to the program)`. Linux smoke and codecs green.
- [ ] **Step 5:** Commit `feat(player): RTX VSR without a model, TensorRT loaded on demand, versions in About`.

### Task 4: TensorRT add-on

**Files:** Create `player/addon.h`, `player/addon.cpp`; Modify `player/update.cpp` (shared `https_download(url, path, cap, progress, cancel)` with WinHTTP, redirects checked with `host_allowed` plus `pypi.nvidia.com` for the add-on only), `player/ui.cpp` (card: Download / Use my own folder / Remove, progress, Cancel), `player/main.cpp` (actions; background thread).

**Interfaces (Produces):**
- `bool https_download(const std::string& url, const std::string& to, size_t cap, const std::vector<std::string>& hosts, std::function<void(size_t, size_t)> progress, const std::atomic<bool>& cancel, std::string& error);`
- `bool extract_tensorrt(const std::string& wheel, const std::string& dir, std::string& error);` — runs `%SystemRoot%\System32\tar.exe -xf <wheel> -C <tmp> tensorrt_libs/*.dll <dist-info>/LICENSE.txt` into `<dir>.new`, rejects any extracted path outside it, then renames `<dir>.new` → `<dir>`; on any failure removes `<dir>.new`.
- `void remove_addon();`
- UiAction kinds: `DownloadTensorRT`, `CancelDownload`, `ChooseTensorRTFolder`, `RemoveTensorRT`; UiState gains `addon_progress` (0..1, <0 none) and `addon_status`.

- [ ] **Step 1: Failing tests:** `extract_tensorrt` on a small test zip made at test time (with `tar.exe -a -cf`) containing `tensorrt_libs/a.dll`, `tensorrt_libs/../evil.dll`, `other/b.dll` → only `a.dll` lands in the folder, `evil.dll` nowhere; extraction into a folder under `op-zoë`; a failing extraction (destination is a file) leaves no `<dir>.new`.
- [ ] **Step 2:** `--uitest` fails to compile.
- [ ] **Step 3: Implement.** The add-on download goes to `<addon dir>\download.whl`, is checked against the pinned hash, extracted, then deleted; a mismatch or cancel deletes everything.
- [ ] **Step 4:** `--uitest` green both builds. **Ask the user in chat** before the one real 1.9 GB download test; with their OK: run it via the card on screen 2 started with `OPEN_PLAYER_SHOW_SETTINGS=1` and a test hook `OPEN_PLAYER_ADDON_AUTOSTART=1` honoured only in `OPEN_PLAYER_UPDATE_TEST` builds; expected: folder holds `nvinfer_11.dll` etc., About shows `TensorRT: 11.5.0 (add-on)`.
- [ ] **Step 5:** Commit `feat(player): TensorRT add-on from NVIDIA, hash-checked`.

### Task 5: Updater

**Files:** Modify `player/update.cpp` (check thread, `latest` JSON, asset URLs), `player/main.cpp` (start-up check once a day, notice, About action, launching the installer), `player/ui.cpp` (About: "Check for updates" switch, "Update available: build N" + Update button, status), `player/settings.{h,cpp}` (`check_updates=1`, `last_update_check=<unix time>`), `player/CMakeLists.txt` (`option(OPEN_PLAYER_UPDATE_TEST ...)`: test public key, `http://127.0.0.1` allowed, About says TEST BUILD).

**Interfaces (Produces):** `struct UpdateCheck { bool done; bool available; UpdateInfo info; std::string installer_url, error; };` `void start_update_check(UpdateCheck&);` `bool download_and_verify_update(const UpdateCheck&, std::string& installer_path, std::string& error);` `bool run_installer(const std::string& path);` (`CreateProcessW` with `/SILENT /SUPPRESSMSGBOXES /NORESTART`, then the player exits).

- [ ] **Step 1: Failing tests:** `parse_release_json` (GitHub `releases/latest` shape) picks the three asset URLs by exact name; a release missing one → no update, no notice; `due_for_check(now, last, enabled)` once a day and never when disabled.
- [ ] **Step 2:** `--uitest` fails to compile.
- [ ] **Step 3: Implement.** Only `installed_copy()` builds check; a failed check logs one line and shows nothing.
- [ ] **Step 4: End-to-end in a TEST build:** sign a test `update.json` with the test key, serve a folder with `py -m http.server` on 127.0.0.1 shaped like the GitHub API, install a test copy (Task 6 installer, `/DIR=` temp) and let it update itself; tamper one byte of the installer and confirm it is refused. Expected: updated build number in `--version`; tampered → "Update failed: the download doesn't match".
- [ ] **Step 5:** Commit `feat(player): signed in-player updater`.

### Task 6: Inno Setup installer

**Files:** Create `installer/open-player.iss`, `installer/build-installer.bat`, `installer/THIRD-PARTY-NOTICES.txt`; Modify `setup-windows.ps1` (`winget install JRSoftware.InnoSetup` step).

- [ ] **Step 1:** Write the `.iss` per the spec (per user, `{localappdata}\Programs\Open Player`, fixed AppId GUID, version from the build number, Start menu, optional desktop icon and "Open with" for mp4/mkv/webm/mov/avi/m4v/ts, `installed.txt`, uninstall asks before removing `%LOCALAPPDATA%\Open Player`). Files: exe, FFmpeg/SDL/dav1d DLLs, `vcruntime140*.dll`/`msvcp140*.dll` from the VS redist folder, `nvngx_vsr.dll`, `assets\`, licences and notices; never `nvinfer*`, `*.onnx`, `*.engine`.
- [ ] **Step 2:** `installer\build-installer.bat` builds `build-win` then runs `iscc`, refusing a TEST build. Expected: `installer\out\OpenPlayer-Setup-<build>.exe`, 60–80 MB.
- [ ] **Step 3:** Install silently into a temp `/DIR=`; run the installed `--version` and `--check`; check the folder has no `nvinfer*`/`.onnx`; uninstall silently; folder gone. Expected as stated.
- [ ] **Step 4:** Commit `feat: Inno Setup installer`.

### Task 7: Release scripts

**Files:** Create `tools/sign-release.ps1`, `tools/release.ps1`.

- [ ] **Step 1:** `sign-release.ps1 -Installer <exe> -Key <pem>`: writes `update.json` (build from the file name, sha256), signs with OpenSSL (`dgst -sha256 -sign`, passphrase prompted), converts DER → r||s base64, verifies with `player/update_pubkey.h`'s key via OpenSSL before finishing.
- [ ] **Step 2:** `release.ps1`: refuses a modified tree, a branch other than `main`, a TEST build, or an unknown build; builds the installer, signs, then prints the `gh release create v<build> ...` command and runs it only after the user types `yes`.
- [ ] **Step 3:** Dry run with the test key on a temp copy: produces three files; the player's `verify_signature` accepts them (checked by `--uitest`-style test vector regenerated from the script's output).
- [ ] **Step 4:** Commit `feat(tools): release signing and publishing scripts`.

### Task 8: README and memory

- [ ] **Step 1:** README: "Install (Windows)" for consumers first (download from Releases, SmartScreen note, no model, RTX VSR works at once, Pixel Flow needs your own model + the TensorRT add-on, where to put `vsr.onnx`), updates and how to turn the check off, About's version lines, the release process for maintainers (key, `release.ps1`, never rewrite `main`). Fix the notice: RTX VSR no longer needs a model.
- [ ] **Step 2:** Update the project memory and its backup in `Open-Player-Keys`.
- [ ] **Step 3:** Commit `docs: installer, updates, TensorRT add-on`.

### Finish

- [ ] Four read-only bug checks, one a security review (signature/host/size logic, extraction, DLL search order, installer permissions, key handling); fix real findings test-first; both builds, `--uitest`, smoke, codecs green; ask before merging and pushing.
