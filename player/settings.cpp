#include "settings.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "ui.h"

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

Settings settings_parse(const std::string& text) {
    Settings s;
    std::istringstream in(text.compare(0, 3, "\xEF\xBB\xBF") ? text : text.substr(3));  // Notepad's UTF-8 BOM
    for (std::string line; std::getline(in, line);) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
        auto flag = [&](bool& out) {
            if (v == "0" || v == "1") out = v == "1";
        };
        if (k == "upscaler") {
            if (v == "rtx") s.upscaler = Upscaler::Rtx;
            else if (v == "pixelflow") s.upscaler = Upscaler::PixelFlow;
            else if (v == "off") s.upscaler = Upscaler::Off;
        } else if (k == "decoder") {
            if (v == "gpu" || v == "cpu") s.cpu_decode = v == "cpu";
        } else if (k == "screen") s.screen = v;
        else if (k == "start_fullscreen") flag(s.start_fullscreen);
        else if (k == "remember_volume") flag(s.remember_volume);
        else if (k == "info_at_start") flag(s.info_at_start);
        else if (k == "trt_folder") s.trt_folder = v;
        else if (k == "check_updates") flag(s.check_updates);
        else if (k == "remember_position") flag(s.remember_position);
        else if (k == "settings_tab") {
            if (v == "0" || v == "1" || v == "2") s.settings_tab = v[0] - '0';
        }
        else if (k == "last_update_check") {
            char* end = nullptr;
            long long t = strtoll(v.c_str(), &end, 10);
            if (!v.empty() && *end == 0 && t >= 0) s.last_update_check = t;
        }
        else if (k == "volume") {
            char* end = nullptr;
            float f = strtof(v.c_str(), &end);
            if (!v.empty() && *end == 0 && f >= 0 && f <= 2) s.volume = f;
        } else if (k == "hide_after") {
            if (v == "1" || v == "2" || v == "3" || v == "5") s.hide_after = v[0] - '0';
        }
    }
    return s;
}

std::string settings_format(const Settings& s) {
    const char* up = s.upscaler == Upscaler::Rtx ? "rtx" : s.upscaler == Upscaler::PixelFlow ? "pixelflow" : "off";
    char vol[32];
    snprintf(vol, sizeof vol, "%g", s.volume);
    std::string screen = trim(s.screen);  // a monitor name comes from the OS: no line break may add a line
    std::replace_if(screen.begin(), screen.end(), [](char c) { return c == '\n' || c == '\r'; }, ' ');
    std::string trt = trim(s.trt_folder);  // a folder name with a line break can't add lines either
    std::replace_if(trt.begin(), trt.end(), [](char c) { return c == '\n' || c == '\r'; }, ' ');
    return std::string("upscaler=") + up + "\ndecoder=" + (s.cpu_decode ? "cpu" : "gpu") + "\nscreen=" + screen +
           "\nstart_fullscreen=" + (s.start_fullscreen ? "1" : "0") + "\nremember_volume=" +
           (s.remember_volume ? "1" : "0") + "\nvolume=" + vol + "\nhide_after=" + std::to_string(s.hide_after) +
           "\ninfo_at_start=" + (s.info_at_start ? "1" : "0") + "\ntrt_folder=" + trt + "\ncheck_updates=" + (s.check_updates ? "1" : "0") +
           "\nlast_update_check=" + std::to_string(s.last_update_check) +
           "\nremember_position=" + (s.remember_position ? "1" : "0") + "\nsettings_tab=" +
           std::to_string(s.settings_tab) + "\n";
}

Settings load_settings(const std::string& file) {
    std::ifstream f(file, std::ios::binary);
    std::string text(64 * 1024, '\0');  // a few hundred bytes in practice; a huge file is cut, not loaded whole
    f.read(text.data(), (std::streamsize)text.size());
    text.resize((size_t)f.gcount());
    return settings_parse(text);
}

bool save_settings(const std::string& file, const Settings& s) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(file).parent_path(), ec);  // first run: no folder yet
    std::string tmp = file + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f << settings_format(s);
        f.close();
        if (!f) return std::filesystem::remove(tmp, ec), false;
    }
    std::error_code moved;
    std::filesystem::rename(tmp, file, moved);
    if (moved) std::filesystem::remove(tmp, ec);  // its own error code: this one must not clear the rename's
    return !moved;
}

std::string settings_file() {
    return (std::filesystem::path(recent_file()).parent_path() / "settings.txt").string();
}

static std::string name_size(const ScreenInfo& s) { return s.name + " " + std::to_string(s.w) + "x" + std::to_string(s.h); }

static constexpr size_t POSITIONS_MAX = 200;

Positions load_positions(const std::string& file) {
    std::ifstream f(std::filesystem::u8path(file), std::ios::binary);
    Positions p;
    std::string line;
    size_t read = 0;
    while (p.size() < POSITIONS_MAX && read < (1 << 20) && std::getline(f, line)) {  // a huge file is cut, not loaded
        read += line.size() + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t tab = line.find('\t');
        if (tab == std::string::npos || tab + 1 >= line.size()) continue;
        char* end = nullptr;
        double t = strtod(line.c_str(), &end);
        if (end != line.c_str() + tab || !(t >= 0 && t < 1e7)) continue;  // also rejects NaN
        p.emplace_back(t, line.substr(tab + 1));
    }
    return p;
}

bool save_positions(const std::string& file, const Positions& p) {
    std::string text;
    char t[32];
    for (auto& [s, path] : p) snprintf(t, sizeof t, "%.1f\t", s), text += t + path + "\n";
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::u8path(file).parent_path(), ec);
    std::string tmp = file + ".tmp";
    {
        std::ofstream f(std::filesystem::u8path(tmp), std::ios::binary);
        f << text;
        f.close();
        if (!f) return std::filesystem::remove(std::filesystem::u8path(tmp), ec), false;
    }
    std::error_code moved;
    std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(file), moved);
    if (moved) std::filesystem::remove(std::filesystem::u8path(tmp), ec);
    return !moved;
}

Positions positions_add(Positions p, const std::string& path, double t) {
    if (path.empty() || path.find_first_of("\t\r\n") != std::string::npos || !(t >= 0)) return p;
    p.erase(std::remove_if(p.begin(), p.end(), [&](auto& e) { return e.second == path; }), p.end());
    p.insert(p.begin(), {t, path});
    if (p.size() > POSITIONS_MAX) p.resize(POSITIONS_MAX);
    return p;
}

double resume_point(const Positions& p, const std::string& path, double dur) {
    for (auto& [t, f] : p)
        if (f == path) return t >= 10 && (dur <= 0 || t < dur - 30) ? t : -1;
    return -1;
}

std::string positions_file() {
    return (std::filesystem::path(recent_file()).parent_path() / "positions.txt").string();
}

// training.json: one flat object of strings, numbers and null (nested values are skipped); strict, small
bool parse_training_status(const std::string& j, TrainingStatus& out) {
    if (j.size() > (1 << 20)) return false;
    size_t i = 0;
    auto ws = [&] { while (i < j.size() && isspace((unsigned char)j[i])) i++; };
    auto str = [&](std::string& s) {
        if (i >= j.size() || j[i] != '"') return false;
        for (i++; i < j.size() && j[i] != '"'; i++) {
            if (j[i] != '\\') { s += j[i]; continue; }
            if (++i >= j.size()) return false;
            char c = j[i];
            s += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c;  // \" \\ \/ as themselves; no \u (UTF-8)
        }
        return i++ < j.size();
    };
    auto skip = [&] {  // a nested array or object, strings included
        int depth = 0;
        do {
            if (i >= j.size()) return false;
            if (j[i] == '"') {
                std::string dummy;
                if (!str(dummy)) return false;
                continue;
            }
            depth += j[i] == '[' || j[i] == '{';
            depth -= j[i] == ']' || j[i] == '}';
            i++;
        } while (depth > 0);
        return true;
    };
    TrainingStatus t;
    ws();
    if (i >= j.size() || j[i++] != '{') return false;
    for (bool first = true;; first = false) {
        ws();
        if (i < j.size() && j[i] == '}' && first) { i++; break; }
        std::string k;
        if (!str(k)) return false;
        ws();
        if (i >= j.size() || j[i++] != ':') return false;
        ws();
        if (i >= j.size()) return false;
        if (j[i] == '"') {
            std::string v;
            if (!str(v)) return false;
            if (k == "state") t.state = v;
            else if (k == "suggest") t.suggest = v;
            else if (k == "out") t.out = v;
        } else if (j[i] == '[' || j[i] == '{') {
            if (!skip()) return false;
        } else if (!j.compare(i, 4, "null")) {
            i += 4;
        } else if (!j.compare(i, 4, "true") || !j.compare(i, 5, "false")) {
            i += j[i] == 't' ? 4 : 5;
        } else {
            char* end = nullptr;
            double v = strtod(j.c_str() + i, &end);
            if (end == j.c_str() + i || !std::isfinite(v)) return false;
            i = end - j.c_str();
            if (k == "step") t.step = (long long)v;
            else if (k == "steps") t.steps = (long long)v;
            else if (k == "updated") t.updated = (long long)v;
            else if (k == "loss") t.loss = v;
            else if (k == "val_db") t.val_db = v, t.has_val = true;
            else if (k == "best_db") t.best_db = v;
            else if (k == "s_per_step") t.s_per_step = v;
            else if (k == "eta_s") t.eta_s = v;
        }
        ws();
        if (i < j.size() && j[i] == ',') { i++; continue; }
        if (i < j.size() && j[i] == '}') { i++; break; }
        return false;
    }
    if (t.state.empty()) return false;
    out = t;
    return true;
}

bool training_stale(const TrainingStatus& t, long long now) { return t.state == "training" && now - t.updated > 300; }

std::string training_file() {
    return (std::filesystem::path(recent_file()).parent_path() / "training.json").string();
}

std::string screen_key(const std::vector<ScreenInfo>& screens, int i) {
    std::string k = name_size(screens[i]);
    int same = 0;  // identical monitors before this one
    for (int j = 0; j < i; j++) same += name_size(screens[j]) == k;
    return same ? k + " #" + std::to_string(same + 1) : k;
}

int match_screen(const std::vector<ScreenInfo>& screens, const std::string& key) {
    for (size_t i = 0; i < screens.size(); i++)
        if (screen_key(screens, (int)i) == key) return (int)i;
    std::string base = key;  // "LG 2560x1440 #3" whose twin is gone -> the first "LG 2560x1440"
    if (size_t h = base.rfind(" #"); h != std::string::npos) base.resize(h);
    for (size_t i = 0; i < screens.size(); i++)
        if (name_size(screens[i]) == base) return (int)i;
    std::string name = base.substr(0, base.rfind(' '));  // "LG ULTRAGEAR 2560x1440" -> "LG ULTRAGEAR"
    for (size_t i = 0; i < screens.size() && !key.empty(); i++)
        if (screens[i].name == name) return (int)i;
    for (size_t i = 0; i < screens.size(); i++)
        if (screens[i].primary) return (int)i;
    return 0;
}

std::vector<std::string> reopen_args(const std::vector<std::string>& argv, const std::string& video, double start,
                                     bool paused) {
    std::vector<std::string> kept;
    for (size_t i = 0; i < argv.size(); i++) {
        if (i && argv[i] == "--screen" && i + 1 < argv.size()) i++;  // and its value
        else if (!i || (argv[i] != "--no-rtx" && argv[i] != "--no-vsr" && argv[i] != "--cpu-decode"))
            kept.push_back(argv[i]);
    }
    std::vector<std::string> args = restart_args(kept, video);  // also drops an earlier --start/--paused
    char t[32];
    snprintf(t, sizeof t, "%.3f", start);  // ms: %g would round to whole seconds after 2.8 h
    args.insert(args.end(), {"--start", t});
    if (paused) args.push_back("--paused");
    return args;
}

std::vector<std::string> live_flags(bool fullscreen, bool vsr_off, bool pixel_flow, int screen) {
    std::vector<std::string> f;
    if (fullscreen) f.push_back("--fullscreen");
    if (vsr_off) f.push_back("--no-vsr");
    else if (pixel_flow) f.push_back("--no-rtx");
    if (screen > 0) f.insert(f.end(), {"--screen", std::to_string(screen)});
    return f;
}

double start_position(double start, double dur) {
    return dur > 0 ? std::clamp(start, 0.0, std::max(0.0, dur - 1)) : std::max(0.0, start);
}

int settings_selftest() {
    int fails = 0;
#define check(cond) \
    if (!(cond)) printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond), fails++

    Settings d = settings_parse("");
    check(d.upscaler == Upscaler::Rtx && d.hide_after == 2 && d.remember_volume && !d.cpu_decode && d.volume == 1);
    // a hand-edited file: CRLF, spaces around '=', out-of-range values, unknown keys and junk are ignored
    Settings s = settings_parse("upscaler=pixelflow\r\ndecoder = cpu\nhide_after=999\nvolume=7\nbogus=1\n\x01\x02\n"
                                "screen=LG ULTRAGEAR 2560x1440\ninfo_at_start=1\n");
    check(s.upscaler == Upscaler::PixelFlow && s.cpu_decode && s.hide_after == 2 && s.volume == 1 &&
          s.screen == "LG ULTRAGEAR 2560x1440" && s.info_at_start);
    s.hide_after = 5, s.volume = 0.5f, s.upscaler = Upscaler::Off, s.remember_volume = false;
    {  // a monitor name with a line break can't smuggle in other settings
        Settings n;
        n.screen = "Evil\nupscaler=off\r";
        Settings b = settings_parse(settings_format(n));
        check(b.upscaler == Upscaler::Rtx && b.screen == "Evil upscaler=off");
    }
    Settings r = settings_parse(settings_format(s));
    check(r.screen == s.screen && r.upscaler == Upscaler::Off && r.hide_after == 5 && r.volume == 0.5f &&
          !r.remember_volume && r.cpu_decode && r.info_at_start);
    {
        auto dir = std::filesystem::temp_directory_path() / "open-player-settings-test";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        std::string f = (dir / "settings.txt").string();
        check(load_settings(f).upscaler == Upscaler::Rtx);  // missing: defaults
        check(save_settings(f, s) && load_settings(f).hide_after == 5 && !std::filesystem::exists(f + ".tmp"));
        check(save_settings((dir / "new" / "settings.txt").string(), s));  // first run: the folder is made
        std::ofstream((dir / "blocker").string()) << "x";                  // a file where the folder should be
        check(!save_settings((dir / "blocker" / "settings.txt").string(), s));
        std::filesystem::create_directories(dir / "taken");  // the target can't be replaced: report it, no .tmp left
        check(!save_settings((dir / "taken").string(), s) && !std::filesystem::exists(dir / "taken.tmp"));
        // a huge or hostile file: only the first 64 KB are read
        std::ofstream((dir / "big.txt").string()) << std::string(70000, 'x') << "\nupscaler=off\n";
        check(load_settings((dir / "big.txt").string()).upscaler == Upscaler::Rtx);
        std::filesystem::remove_all(dir);
    }
    check(std::filesystem::path(settings_file()).filename() == "settings.txt" &&
          std::filesystem::path(settings_file()).parent_path() == std::filesystem::path(recent_file()).parent_path());

    std::vector<ScreenInfo> mon = {{"PG27UCDM", 3840, 2160, 240, true}, {"LG ULTRAGEAR", 2560, 1440, 60, false},
                                   {"LG ULTRAGEAR", 1920, 1080, 60, false}};
    check(screen_key(mon, 1) == "LG ULTRAGEAR 2560x1440");
    std::vector<ScreenInfo> twins = {{"LG", 2560, 1440, 60, true}, {"LG", 2560, 1440, 60, false}};  // identical monitors
    check(screen_key(twins, 0) == "LG 2560x1440" && screen_key(twins, 1) == "LG 2560x1440 #2");
    check(match_screen(twins, "LG 2560x1440 #2") == 1 && match_screen(twins, "LG 2560x1440") == 0 &&
          match_screen(twins, "LG 2560x1440 #3") == 0);  // the third is gone: the first of that kind
    check(match_screen(mon, "LG ULTRAGEAR 1920x1080") == 2 && match_screen(mon, "LG ULTRAGEAR 3840x2160") == 1);
    check(match_screen(mon, "Gone 1x1") == 0 && match_screen(mon, "") == 0 && match_screen({}, "x") == 0);
    std::vector<ScreenInfo> mon2 = {{"A", 1, 1, 60, false}, {"B", 2, 2, 60, true}};
    check(match_screen(mon2, "Gone 1x1") == 1);  // the primary, wherever it is in the list

    check(reopen_args({"open-player", "--screen", "2", "--no-rtx", "a.mkv", "--start", "3", "--paused"},
                      "/x/My Movie (2012) \xc3\xa9.mkv", 61.5, false) ==
          (std::vector<std::string>{"open-player", "/x/My Movie (2012) \xc3\xa9.mkv", "--start", "61.500"}));
    check(reopen_args({"open-player", "--fullscreen", "a.mkv"}, "a.mkv", 0, true) ==
          (std::vector<std::string>{"open-player", "--fullscreen", "a.mkv", "--start", "0.000", "--paused"}));
    check(reopen_args({"open-player", "--model", "m.onnx", "--cpu-decode", "--no-vsr", "a.mkv"}, "a.mkv", 1, false) ==
          (std::vector<std::string>{"open-player", "--model", "m.onnx", "a.mkv", "--start", "1.000"}));

    // what a reopen keeps of this run without saving it: fullscreen, VSR off or Pixel Flow, the monitor (0: none)
    check(live_flags(true, false, true, 2) == (std::vector<std::string>{"--fullscreen", "--no-rtx", "--screen", "2"}));
    check(live_flags(false, true, true, 0) == (std::vector<std::string>{"--no-vsr"}));  // off wins over the upscaler
    check(live_flags(false, false, false, 0).empty());

    check(start_position(5, .5) == 0 && start_position(5, 0) == 5);  // a clip of 1 s or less starts at 0
    check(settings_parse("\xEF\xBB\xBFupscaler=off\n").upscaler == Upscaler::Off);  // Notepad's UTF-8 BOM
    {  // resume positions
        Settings t;
        t.remember_position = false, t.settings_tab = 2;
        Settings b = settings_parse(settings_format(t));
        check(!b.remember_position && b.settings_tab == 2 && settings_parse("settings_tab=7\n").settings_tab == 0);
        Positions p;
        for (int i = 0; i < 205; i++) p = positions_add(p, "/v/" + std::to_string(i) + ".mkv", i);
        check(p.size() == 200 && p.front().second == "/v/204.mkv");  // newest first, cap 200
        p = positions_add(p, "/v/10.mkv", 77);
        check(p.front() == (std::pair<double, std::string>{77, "/v/10.mkv"}) &&
              std::count_if(p.begin(), p.end(), [](auto& e) { return e.second == "/v/10.mkv"; }) == 1);
        check(positions_add({}, "a\tb.mkv", 50).empty() && positions_add({}, "a\nb.mkv", 50).empty());
        Positions r = {{120, "/a.mkv"}, {5, "/b.mkv"}, {3590, "/c.mkv"}, {20, "/d.mkv"}, {15, "/e.mkv"}};
        check(resume_point(r, "/a.mkv", 3600) == 120 && resume_point(r, "/b.mkv", 3600) == -1 &&
              resume_point(r, "/c.mkv", 3600) == -1 && resume_point(r, "/d.mkv", 35) == -1 &&
              resume_point(r, "/e.mkv", 0) == 15 && resume_point(r, "/nope.mkv", 3600) == -1);
        auto dir = std::filesystem::temp_directory_path() / "open-player-positions-test";
        std::filesystem::remove_all(dir);
        std::string f = (dir / "positions.txt").string();
        check(load_positions(f).empty() && save_positions(f, r) && !std::filesystem::exists(f + ".tmp"));
        check(load_positions(f) == r);
        std::ofstream(f, std::ios::app) << "junk\nnan\t/x.mkv\n-5\t/y.mkv\n12\t\n";
        check(load_positions(f) == r);  // junk lines skipped
        std::filesystem::remove_all(dir);
        check(std::filesystem::path(positions_file()).filename() == "positions.txt");
    }
    {  // training.json as train.py writes it
        TrainingStatus t;
        check(parse_training_status(
                  "{\"state\": \"training\", \"steps\": 100000, \"out\": \"D:\\\\Tr\xc3\xa4ning\\\\vsr.pt\", \"step\": 42000, "
                  "\"loss\": 0.0312, \"s_per_step\": 0.13, \"eta_s\": 7540.0, \"val_db\": 1.84, \"best_db\": 1.91, "
                  "\"suggest\": \"plateau: try --fft-loss 0.1\", \"updated\": 1795300000, \"extra\": [1, 2]}",
                  t) &&
              t.state == "training" && t.step == 42000 && t.steps == 100000 && t.loss == 0.0312 && t.has_val &&
              t.best_db == 1.91 && t.out == "D:\\Tr\xc3\xa4ning\\vsr.pt" && t.suggest.find("fft") != std::string::npos);
        TrainingStatus n;
        check(parse_training_status("{\"state\": \"finished\", \"step\": 5, \"steps\": 5, \"val_db\": null}", n) &&
              !n.has_val);
        check(!parse_training_status("", n) && !parse_training_status("{\"state\": \"trai", n) &&
              !parse_training_status("junk", n) && !parse_training_status("{\"step\": 5}", n));  // no state
        check(!training_stale(t, 1795300000 + 100) && training_stale(t, 1795300000 + 400) &&
              !training_stale(t, 1795300000 - 3600));  // updated "in the future" (clock set back): running
        n.state = "finished";
        check(!training_stale(n, 1795300000 + 400000));
        check(std::filesystem::path(training_file()).filename() == "training.json");
    }
    {
        Settings u;
        u.check_updates = false, u.last_update_check = 1795270000;
        Settings b = settings_parse(settings_format(u));
        check(!b.check_updates && b.last_update_check == 1795270000);
        check(settings_parse("last_update_check=-5\n").last_update_check == 0 && settings_parse("").check_updates);
    }
    check(start_position(-3, 100) == 0 && start_position(500, 100) == 99 && start_position(42, 100) == 42 &&
          start_position(42, 0) == 42);
#undef check
    return fails ? 1 : 0;
}
