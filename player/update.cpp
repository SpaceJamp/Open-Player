#include "update.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#endif

// A strict reader for update.json's one flat shape: string keys, string or integer values, nothing else.
static bool parse_flat_json(const std::string& t, std::map<std::string, std::string>& out) {
    size_t i = 0;
    auto ws = [&] { while (i < t.size() && isspace((unsigned char)t[i])) i++; };
    auto str = [&](std::string& s) {
        if (i >= t.size() || t[i] != '"') return false;
        for (i++; i < t.size() && t[i] != '"'; i++) {
            if ((unsigned char)t[i] < 0x20) return false;
            if (t[i] == '\\') {
                if (++i >= t.size()) return false;
                char c = t[i];
                if (c == 'n') s += '\n';
                else if (c == 't') s += '\t';
                else if (c == '"' || c == '\\' || c == '/') s += c;
                else return false;  // \u and the rest: not needed, not accepted
            } else s += t[i];
        }
        return i++ < t.size();
    };
    ws();
    if (i >= t.size() || t[i++] != '{') return false;
    for (;;) {
        ws();
        std::string k, v;
        if (!str(k) || out.count(k)) return false;
        ws();
        if (i >= t.size() || t[i++] != ':') return false;
        ws();
        if (i < t.size() && t[i] == '"') {
            if (!str(v)) return false;
            v = "s" + v;
        } else {
            size_t b = i;
            if (i < t.size() && t[i] == '-') i++;
            while (i < t.size() && isdigit((unsigned char)t[i])) i++;
            if (i == b || i - b > 9) return false;
            v = "n" + t.substr(b, i - b);
        }
        out[k] = v;
        ws();
        if (i < t.size() && t[i] == ',') { i++; continue; }
        if (i < t.size() && t[i] == '}') { i++; break; }
        return false;
    }
    ws();
    return i == t.size();
}

bool parse_update_json(const std::string& text, UpdateInfo& out) {
    if (text.size() > 64 * 1024) return false;
    std::map<std::string, std::string> m;
    if (!parse_flat_json(text, m) || m.size() != 4) return false;
    for (const char* k : {"build", "file", "sha256", "notes"})
        if (!m.count(k)) return false;
    if (m["build"][0] != 'n' || m["file"][0] != 's' || m["sha256"][0] != 's' || m["notes"][0] != 's') return false;
    int build = atoi(m["build"].c_str() + 1);
    std::string file = m["file"].substr(1), sha = m["sha256"].substr(1);
    if (build <= 0 || file != "OpenPlayer-Setup-" + std::to_string(build) + ".exe") return false;
    if (sha.size() != 64 || !std::all_of(sha.begin(), sha.end(), [](char c) { return isdigit((unsigned char)c) || (c >= 'a' && c <= 'f'); }))
        return false;
    out = {build, file, sha, m["notes"].substr(1)};
    return true;
}

int build_number(const std::string& b) {
    int n = 0;
    char hash[64] = {}, end = 0;
    // "build 214 (4a30cf8)": a number and a plain hash; "-modified" builds and "build unknown" don't count
    if (sscanf(b.c_str(), "build %d (%63[0-9a-f]%c", &n, hash, &end) != 3 || end != ')' || n <= 0) return -1;
    return n;
}

bool should_offer(int current, const UpdateInfo& u) { return current > 0 && u.build > current; }

bool host_allowed(const std::string& url) {
    const std::string p = "https://";
    if (url.compare(0, p.size(), p)) return false;
    size_t e = url.find_first_of("/?#", p.size());
    std::string host = url.substr(p.size(), e == std::string::npos ? std::string::npos : e - p.size());
    if (host.find('@') != std::string::npos) return false;  // user@host tricks
    std::transform(host.begin(), host.end(), host.begin(), [](unsigned char c) { return (char)tolower(c); });
    if (size_t c = host.find(':'); c != std::string::npos) {
        if (host.substr(c) != ":443") return false;
        host.resize(c);
    }
    for (const char* ok : {"api.github.com", "github.com", "release-assets.githubusercontent.com",
                           "objects.githubusercontent.com"})
        if (host == ok) return true;
    return false;
}

#ifdef _WIN32
static bool base64_decode(const std::string& in, std::vector<unsigned char>& out) {
    DWORD n = 0;
    if (!CryptStringToBinaryA(in.c_str(), (DWORD)in.size(), CRYPT_STRING_BASE64, nullptr, &n, nullptr, nullptr))
        return false;
    out.resize(n);
    return CryptStringToBinaryA(in.c_str(), (DWORD)in.size(), CRYPT_STRING_BASE64, out.data(), &n, nullptr, nullptr);
}

static bool sha256(const void* data, size_t n, unsigned char out[32]) {
    return BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, (PUCHAR)data, (ULONG)n, out, 32) == 0;
}

bool verify_signature(const unsigned char pubkey[64], const std::string& msg, const std::string& sig_b64) {
    std::vector<unsigned char> sig;
    if (sig_b64.empty() || sig_b64.size() > 200 || !base64_decode(sig_b64, sig) || sig.size() != 64) return false;
    unsigned char hash[32];
    if (!sha256(msg.data(), msg.size(), hash)) return false;
    unsigned char blob[sizeof(BCRYPT_ECCKEY_BLOB) + 64];
    auto* h = (BCRYPT_ECCKEY_BLOB*)blob;
    h->dwMagic = BCRYPT_ECDSA_PUBLIC_P256_MAGIC, h->cbKey = 32;
    memcpy(blob + sizeof *h, pubkey, 64);
    BCRYPT_KEY_HANDLE key = nullptr;
    if (BCryptImportKeyPair(BCRYPT_ECDSA_P256_ALG_HANDLE, nullptr, BCRYPT_ECCPUBLIC_BLOB, &key, blob, sizeof blob, 0))
        return false;
    bool ok = BCryptVerifySignature(key, nullptr, hash, 32, sig.data(), 64, 0) == 0;
    BCryptDestroyKey(key);
    return ok;
}

static std::string hex(const unsigned char* p, size_t n) {
    std::string s;
    char b[3];
    for (size_t i = 0; i < n; i++) snprintf(b, sizeof b, "%02x", p[i]), s += b;
    return s;
}

std::string sha256_hex(const std::string& data) {
    unsigned char h[32];
    return sha256(data.data(), data.size(), h) ? hex(h, 32) : "";
}

bool verify_file_sha256(const std::string& path, const std::string& want) {
    BCRYPT_HASH_HANDLE hh = nullptr;
    if (BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE, &hh, nullptr, 0, nullptr, 0, 0)) return false;
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    std::vector<char> buf(1 << 20);
    bool ok = (bool)f;
    while (ok && f) {
        f.read(buf.data(), (std::streamsize)buf.size());
        if (f.gcount() > 0) ok = !BCryptHashData(hh, (PUCHAR)buf.data(), (ULONG)f.gcount(), 0);
    }
    unsigned char h[32];
    ok = ok && !BCryptFinishHash(hh, h, 32, 0);
    BCryptDestroyHash(hh);
    return ok && hex(h, 32) == want;
}
#else
bool verify_signature(const unsigned char*, const std::string&, const std::string&) { return false; }
std::string sha256_hex(const std::string&) { return ""; }
bool verify_file_sha256(const std::string&, const std::string&) { return false; }
#endif

int update_selftest() {
    int fails = 0;
#define check(cond) \
    if (!(cond)) printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond), fails++

    // a test key pair made once with OpenSSL (its private half was deleted); nothing to do with the real key
    static const unsigned char TEST_PUB[64] = {
        0x68, 0x65, 0x3b, 0x56, 0x26, 0xaa, 0xda, 0x94, 0x2b, 0x1d, 0xfb, 0x03, 0x13, 0x46, 0x0f, 0xc1,
        0x37, 0xaf, 0xe5, 0xfe, 0x00, 0x7c, 0x72, 0xdd, 0x22, 0x07, 0xab, 0xa0, 0x2b, 0xcc, 0x1d, 0x67,
        0x8f, 0xdb, 0x8b, 0x4d, 0x46, 0x95, 0x91, 0x31, 0xf8, 0xb9, 0xd1, 0xde, 0x5e, 0x13, 0x34, 0xf7,
        0x5b, 0x80, 0x0d, 0xbf, 0x39, 0x80, 0xca, 0x7d, 0x89, 0x05, 0x38, 0xc8, 0xe8, 0xac, 0xd4, 0xfe};
    const std::string json = "{\"build\": 214, \"file\": \"OpenPlayer-Setup-214.exe\", \"sha256\": "
                             "\"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\", \"notes\": \"test\"}";
    const std::string sig = "pRWvTHJ6zIUMq2T+9nPB8mZj+mYtr9gRnDkjYJLPs3FfRBwmZlRzQeU383bm4zbDeLmVkdcDXRwoJrjhXlAcXg==";
    const std::string other_key_sig =  // the same json signed by a different key
        "8TPNmySjf0uth+ki+lVx3QD1FErIHqo8v99AQbB0OCOjGBy+IFWaKkikyquOmzvhI3F9dhsFoLNfIlY482h2/g==";
#ifdef _WIN32
    check(verify_signature(TEST_PUB, json, sig));
    std::string tampered = json;
    tampered[10] = '9';  // build 214 -> 914
    check(!verify_signature(TEST_PUB, tampered, sig));
    check(!verify_signature(TEST_PUB, json, other_key_sig));
    check(!verify_signature(TEST_PUB, json, "not base64!") && !verify_signature(TEST_PUB, json, ""));
    check(!verify_signature(TEST_PUB, json, sig.substr(0, 40)));  // too short
    check(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    {
        auto f = (std::filesystem::temp_directory_path() / "op-update-test.bin").string();
        std::ofstream(f, std::ios::binary) << "abc";
        check(verify_file_sha256(f, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
        std::ofstream(f, std::ios::binary) << "ab";  // cut off
        check(!verify_file_sha256(f, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
        std::filesystem::remove(f);
        check(!verify_file_sha256(f, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));  // gone
    }
#else
    printf("note: update signature checks are Windows-only, skipped\n");
    (void)TEST_PUB, (void)sig, (void)other_key_sig;
#endif
    UpdateInfo u;
    check(parse_update_json(json, u) && u.build == 214 && u.file == "OpenPlayer-Setup-214.exe" &&
          u.sha256.size() == 64 && u.notes == "test");
    UpdateInfo bad;
    check(!parse_update_json("", bad) && !parse_update_json("junk", bad) && !parse_update_json("{}", bad));
    check(!parse_update_json("{\"build\": 214, \"file\": \"OpenPlayer-Setup-214.exe\"}", bad));  // no sha256
    check(!parse_update_json("{\"build\": 214, \"file\": \"..\\\\OpenPlayer-Setup-214.exe\", \"sha256\": \"" +
                                 u.sha256 + "\", \"notes\": \"\"}", bad));
    check(!parse_update_json("{\"build\": 214, \"file\": \"OpenPlayer-Setup-215.exe\", \"sha256\": \"" + u.sha256 +
                                 "\", \"notes\": \"\"}", bad));  // file doesn't match the build
    check(!parse_update_json("{\"build\": 214, \"file\": \"OpenPlayer-Setup-214.exe\", \"sha256\": \"abc\", "
                             "\"notes\": \"\"}", bad));
    check(!parse_update_json("{\"build\": 214, \"file\": \"OpenPlayer-Setup-214.exe\", \"sha256\": \"" + u.sha256 +
                                 "\", \"notes\": \"\", \"extra\": 1}", bad));  // unknown key
    check(!parse_update_json("{\"build\": -3, \"file\": \"OpenPlayer-Setup--3.exe\", \"sha256\": \"" + u.sha256 +
                                 "\", \"notes\": \"\"}", bad));

    check(build_number("build 214 (4a30cf8)") == 214 && build_number("build 214 (4a30cf8-modified)") == -1 &&
          build_number("build unknown") == -1 && build_number("") == -1);
    UpdateInfo n;
    n.build = 214;
    check(should_offer(210, n) && !should_offer(214, n) && !should_offer(300, n) && !should_offer(-1, n));

    check(host_allowed("https://release-assets.githubusercontent.com/x") && host_allowed("https://api.github.com/x") &&
          host_allowed("https://github.com/SpaceJamp/Open-Player/releases") &&
          host_allowed("https://objects.githubusercontent.com/x"));
    check(!host_allowed("http://github.com/x") && !host_allowed("https://github.com.evil.io/x") &&
          !host_allowed("https://evil.io/?github.com") && !host_allowed("https://github.com@evil.io/x") &&
          !host_allowed("https://evil-github.com/x") && !host_allowed("ftp://github.com/x") && !host_allowed(""));
    check(host_allowed("https://GitHub.com/x") && host_allowed("https://github.com:443/x") &&
          !host_allowed("https://github.com:8443/x"));
#undef check
    return fails ? 1 : 0;
}
