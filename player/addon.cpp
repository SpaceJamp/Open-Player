#include "addon.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "update.h"

namespace fs = std::filesystem;

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
