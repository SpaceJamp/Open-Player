#pragma once
// The updater's checks. A release carries OpenPlayer-Setup-<build>.exe, update.json and update.json.sig;
// update.json = {"build": N, "file": "OpenPlayer-Setup-N.exe", "sha256": "<64 hex>", "notes": "..."}, signed with
// the maintainer's key (ECDSA P-256 over SHA-256 of update.json's exact bytes, raw r||s, base64). Nothing is
// downloaded from or run off anything that fails these checks.
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
int update_selftest();  // part of --uitest; 0 on pass
