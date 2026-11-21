#pragma once
// The updater's checks. A release carries OpenPlayer-Setup-<build>.exe, update.json and update.json.sig;
// update.json = {"build": N, "file": "OpenPlayer-Setup-N.exe", "sha256": "<64 hex>", "notes": "..."}, signed with
// the maintainer's key (ECDSA P-256 over SHA-256 of update.json's exact bytes, raw r||s, base64). Nothing is
// downloaded from or run off anything that fails these checks.
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct UpdateInfo {
    int build = -1;
    std::string file, sha256, notes;
};
// Exactly these keys, file == "OpenPlayer-Setup-<build>.exe", sha256 64 lowercase hex; false otherwise.
bool parse_update_json(const std::string& text, UpdateInfo& out);
int build_number(const std::string& build_text);  // "build 214 (4a30cf8)" -> 214; modified or unknown -> -1
bool should_offer(int current, const UpdateInfo& u);  // only newer, and never from a build without a number
bool host_allowed(const std::string& url);           // HTTPS to GitHub's API, site and release-asset hosts only
bool host_in(const std::string& url, const std::vector<std::string>& hosts);  // HTTPS to exactly one of hosts
// pubkey: X || Y of a P-256 key. Windows only (CNG); elsewhere always false.
bool verify_signature(const unsigned char pubkey[64], const std::string& msg, const std::string& sig_b64);
std::string sha256_hex(const std::string& data);  // lowercase hex; "" where unavailable (not Windows)
bool verify_file_sha256(const std::string& path, const std::string& hex);
// HTTPS GET (Windows: WinHTTP). Every hop, redirects included, must be HTTPS to one of hosts; more than cap bytes
// fails. https_download writes to `to` via `to`.part (deleted on failure); https_get returns the body.
bool https_download(const std::string& url, const std::string& to, uint64_t cap, const std::vector<std::string>& hosts,
                    const std::function<void(uint64_t, uint64_t)>& progress, const std::atomic<bool>& cancel,
                    std::string& error);
bool https_get(const std::string& url, uint64_t cap, const std::vector<std::string>& hosts, std::string& body,
               std::string& error);
// From GitHub's releases/latest JSON: the download URLs whose file name is exactly name (empty if none or several).
std::string release_asset_url(const std::string& release_json, const std::string& name);
bool due_for_check(int64_t now, int64_t last, bool enabled);  // at most once a day

// The whole check (on a background thread): latest release -> update.json + signature -> verified -> newer?
struct UpdateCheck {
    std::atomic<bool> done{false};
    bool available = false;
    UpdateInfo info;
    std::string installer_url, error;
};
void check_for_update(UpdateCheck& c, int current_build);
// Downloads the installer into %TEMP%\open-player-update and checks it against the signed hash; false + error.
bool download_update(const UpdateCheck& c, const std::function<void(uint64_t, uint64_t)>& progress,
                     const std::atomic<bool>& cancel, std::string& installer_path, std::string& error);
bool run_installer(const std::string& path);  // starts it silently; the player then exits
int update_selftest();  // part of --uitest; 0 on pass
