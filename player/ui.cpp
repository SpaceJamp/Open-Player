#include "ui.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static constexpr size_t RECENT_MAX = 10;

std::vector<std::string> add_recent(std::vector<std::string> list, const std::string& path) {
    list.erase(std::remove(list.begin(), list.end(), path), list.end());
    list.insert(list.begin(), path);
    if (list.size() > RECENT_MAX) list.resize(RECENT_MAX);
    return list;
}

std::vector<std::string> existing_only(const std::vector<std::string>& list) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& p : list)
        if (fs::exists(p, ec)) out.push_back(p);
    return out;
}

std::vector<std::string> load_recent(const std::string& file) {
    std::vector<std::string> out;
    std::ifstream in(file);
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) out.push_back(line);
    return out;
}

void save_recent(const std::string& file, const std::vector<std::string>& list) {
    std::error_code ec;
    fs::create_directories(fs::path(file).parent_path(), ec);
    std::ofstream out(file);
    for (const auto& p : list) out << p << '\n';
}

std::string recent_file() {
    const char* xdg = getenv("XDG_CONFIG_HOME");
    const char* home = getenv("HOME");
    fs::path base = xdg && *xdg ? fs::path(xdg) : fs::path(home ? home : ".") / ".config";
    return (base / "open-player" / "recent.txt").string();
}

int ui_selftest() {
    int fails = 0;
#define check(cond) \
    if (!(cond)) printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond), fails++

    auto l = add_recent({}, "/a.mp4");
    for (int i = 0; i < 12; i++) l = add_recent(l, "/v" + std::to_string(i) + ".mkv");
    check(l.size() == 10 && l.front() == "/v11.mkv");  // cap 10, newest first
    l = add_recent(l, "/v5.mkv");
    check(l.front() == "/v5.mkv" && std::count(l.begin(), l.end(), "/v5.mkv") == 1);  // no duplicates
    check(load_recent("/nonexistent/dir/recent.txt").empty());  // missing file -> empty
    save_recent("/proc/forbidden/recent.txt", l);                // unwritable -> no crash
    // round trip through a temp file, including a path with spaces and non-ASCII
    std::string tmp = (std::filesystem::temp_directory_path() / "open-player-uitest-recent.txt").string();
    std::string odd = "/mnt/c/Users/Yeti/Vid\xc3\xa9os/My Movie (2012).mkv";
    save_recent(tmp, {odd, "/b.mp4"});
    check(load_recent(tmp) == (std::vector<std::string>{odd, "/b.mp4"}));
    check(existing_only({tmp, "/definitely/missing.mkv"}) == std::vector<std::string>{tmp});
    std::filesystem::remove(tmp);

#undef check
    if (!fails) printf("ui checks passed\n");
    return fails ? 1 : 0;
}
