#include "addon.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "trt_loader.h"
#include "update.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

#ifdef _WIN32
// Windows' own tar.exe (never one from PATH or the current folder); it refuses ".." and absolute entry names.
static bool run_tar(const fs::path& wheel, const fs::path& into) {
    wchar_t sys[MAX_PATH];
    if (!GetSystemDirectoryW(sys, MAX_PATH)) return false;
    std::wstring exe = std::wstring(sys) + L"\\tar.exe";
    std::wstring cmd = L"\"" + exe + L"\" -xf \"" + wheel.wstring() + L"\" -C \"" + into.wstring() +
                       L"\" \"tensorrt_libs/*.dll\" \"*.dist-info/LICENSE.txt\"";
    STARTUPINFOW si{sizeof si};
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
        return false;
    WaitForSingleObject(pi.hProcess, INFINITE);  // its exit code is ignored: refused entries make it non-zero
    CloseHandle(pi.hThread), CloseHandle(pi.hProcess);
    return true;
}
#endif

bool extract_tensorrt(const std::string& wheel, const std::string& dir, std::string& error) {
#ifdef _WIN32
    fs::path d = fs::u8path(dir), stage = d, raw;
    stage += ".new";
    raw = stage / "raw";
    std::error_code ec;
    fs::remove_all(stage, ec);
    auto fail = [&](const std::string& why) {
        fs::remove_all(stage, ec);
        return error = why, false;
    };
    if (!fs::create_directories(raw, ec) || ec) return fail("can't create " + stage.u8string());
    if (!run_tar(fs::u8path(wheel), raw)) return fail("couldn't run Windows' tar.exe");
    // keep only what we asked for, flattened; anything else that appeared is dropped with raw/
    int dlls = 0;
    for (auto& e : fs::directory_iterator(raw / "tensorrt_libs", ec))
        if (e.is_regular_file(ec) && e.path().extension() == ".dll")
            fs::rename(e.path(), stage / e.path().filename(), ec), dlls += !ec;
    for (auto& e : fs::directory_iterator(raw, ec))
        if (e.path().extension() == ".dist-info") fs::rename(e.path() / "LICENSE.txt", stage / "LICENSE.txt", ec);
    fs::remove_all(raw, ec);
    if (!dlls) return fail("no TensorRT DLLs in the download");
    fs::remove_all(d, ec);  // the old add-on: fails while its DLLs are loaded
    if (ec) return fail("close and reopen the player first: the current TensorRT is in use");
    fs::rename(stage, d, ec);
    return ec ? fail("can't move the add-on into place") : true;
#else
    (void)wheel, (void)dir;
    return error = "the add-on is Windows-only", false;
#endif
}

bool install_tensorrt_addon(const std::function<void(uint64_t, uint64_t)>& progress, const std::atomic<bool>& cancel,
                            std::string& error) {
    fs::path base = fs::u8path(user_data_dir()), wheel = base / "tensorrt-download.whl";
    std::error_code ec;
    fs::create_directories(base, ec);
    // the download (2 GB) and the unpacked DLLs (2.3 GB) are both on disk for a moment
    if (fs::space(base, ec).available < 4700ull << 20 && !ec) return error = "needs 4.7 GB of free disk space", false;
    bool ok = https_download(TRT_WHEEL_URL, wheel.u8string(), TRT_WHEEL_SIZE, {"pypi.nvidia.com"}, progress, cancel,
                             error);
    if (ok && !verify_file_sha256(wheel.u8string(), TRT_WHEEL_SHA256))
        ok = false, error = "the download doesn't match NVIDIA's published file";
    if (ok) ok = extract_tensorrt(wheel.u8string(), addon_dir(), error);
    if (ok && !fs::exists(fs::u8path(addon_dir()) / "nvinfer_11.dll", ec))
        ok = false, error = "the add-on is incomplete";
    fs::remove(wheel, ec);
    return ok;
}

void remove_tensorrt_addon() {
    std::error_code ec;
    fs::remove_all(fs::u8path(addon_dir()), ec);
    if (ec) std::ofstream(fs::u8path(user_data_dir()) / "remove-tensorrt.txt") << "remove at next start\n";
}

void finish_pending_addon_removal() {
    std::error_code ec;
    fs::path marker = fs::u8path(user_data_dir()) / "remove-tensorrt.txt";
    if (!fs::exists(marker, ec)) return;
    fs::remove_all(fs::u8path(addon_dir()), ec);
    if (!ec) fs::remove(marker, ec);
}

// --- test helper: a stored (uncompressed) zip with any entry names, including hostile ones ---
static uint32_t crc32(const std::string& s) {
    uint32_t c = 0xffffffff;
    for (unsigned char b : s) {
        c ^= b;
        for (int k = 0; k < 8; k++) c = c >> 1 ^ (0xedb88320 & (0 - (c & 1)));
    }
    return ~c;
}
static void write_zip(const fs::path& to, const std::vector<std::pair<std::string, std::string>>& entries) {
    std::string out, cd;
    auto u16 = [](std::string& s, uint16_t v) { s.append((const char*)&v, 2); };
    auto u32 = [](std::string& s, uint32_t v) { s.append((const char*)&v, 4); };
    for (auto& [name, data] : entries) {
        uint32_t off = (uint32_t)out.size(), crc = crc32(data), n = (uint32_t)data.size();
        u32(out, 0x04034b50), u16(out, 20), u16(out, 0), u16(out, 0), u16(out, 0), u16(out, 0);
        u32(out, crc), u32(out, n), u32(out, n), u16(out, (uint16_t)name.size()), u16(out, 0);
        out += name + data;
        u32(cd, 0x02014b50), u16(cd, 20), u16(cd, 20), u16(cd, 0), u16(cd, 0), u16(cd, 0), u16(cd, 0);
        u32(cd, crc), u32(cd, n), u32(cd, n), u16(cd, (uint16_t)name.size()), u16(cd, 0), u16(cd, 0), u16(cd, 0);
        u16(cd, 0), u32(cd, 0), u32(cd, off);
        cd += name;
    }
    std::string end;
    u32(end, 0x06054b50), u16(end, 0), u16(end, 0), u16(end, (uint16_t)entries.size()), u16(end, (uint16_t)entries.size());
    u32(end, (uint32_t)cd.size()), u32(end, (uint32_t)out.size()), u16(end, 0);
    std::ofstream(to, std::ios::binary) << out << cd << end;
}

int addon_selftest() {
    int fails = 0;
#define check(cond) \
    if (!(cond)) printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond), fails++

    check(host_in("https://pypi.nvidia.com/tensorrt-cu13-libs/x.whl", {"pypi.nvidia.com"}));
    check(!host_in("https://pypi.nvidia.com.evil.io/x", {"pypi.nvidia.com"}) &&
          !host_in("http://pypi.nvidia.com/x", {"pypi.nvidia.com"}) && !host_in("https://pypi.org/x", {"pypi.nvidia.com"}));
#ifdef _WIN32
    // a non-ASCII folder, as a user profile like C:\Users\Zoë would give
    fs::path base = fs::temp_directory_path() / fs::u8path("op-zo\xc3\xab-addon-test");
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base);
    fs::path wheel = base / "t.whl", dir = base / "tensorrt";
    write_zip(wheel, {{"tensorrt_libs/a.dll", "A"},
                      {"tensorrt_libs/../evil.dll", "E"},
                      {"other/b.dll", "B"},
                      {"tensorrt_cu13_libs-11.5.0.63.dist-info/LICENSE.txt", "L"}});
    std::string err;
    bool ok = extract_tensorrt(wheel.u8string(), dir.u8string(), err);
    check(ok);
    if (!ok) printf("  (%s)\n", err.c_str());
    check(fs::exists(dir / "a.dll") && fs::exists(dir / "LICENSE.txt") && !fs::exists(dir / "b.dll"));
    check(!fs::exists(base / "evil.dll") && !fs::exists(dir / "evil.dll") && !fs::exists(base / "tensorrt.new"));
    // a failure (the destination's parent is a file) leaves nothing behind
    std::ofstream(base / "blocker") << "x";
    check(!extract_tensorrt(wheel.u8string(), (base / "blocker" / "tensorrt").u8string(), err));
    check(!fs::exists(base / "blocker" / "tensorrt.new", ec));
    // a wheel without TensorRT's DLLs is refused, and the old add-on stays
    write_zip(base / "empty.whl", {{"other/b.dll", "B"}});
    check(!extract_tensorrt((base / "empty.whl").u8string(), dir.u8string(), err) && fs::exists(dir / "a.dll"));
    fs::remove_all(base, ec);
#endif
#undef check
    return fails ? 1 : 0;
}
