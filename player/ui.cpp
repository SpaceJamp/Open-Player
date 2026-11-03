#include "ui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "imgui.h"
#include "theme.h"

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

std::string clock_text(double t) {
    int s = (int)std::fmax(t, 0);
    char buf[32];
    if (s >= 3600) snprintf(buf, sizeof buf, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60);
    else snprintf(buf, sizeof buf, "%02d:%02d", s / 60, s % 60);
    return buf;
}

std::vector<std::string> restart_args(const std::vector<std::string>& argv, const std::string& video) {
    std::vector<std::string> out = argv;
    for (size_t i = 1; i < out.size(); i++) {
        if (out[i] == "--model" || out[i] == "--screen") i++;  // skip the flag's value
        else if (out[i].empty() || out[i][0] != '-') return out[i] = video, out;
    }
    out.push_back(video);
    return out;
}

bool can_open(const std::string& path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

bool ui_has_keyboard() {
    return ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId) || ImGui::GetIO().WantTextInput;
}

bool controls_visible(double now, double last_mouse, bool paused, bool hovered) {
    return paused || hovered || now - last_mouse < 2;
}

float fade_alpha(double now, double changed_at, bool visible, double in, double out) {
    double t = std::clamp((now - changed_at) / (visible ? in : out), 0.0, 1.0);
    return (float)(visible ? t : 1 - t);
}

float notice_alpha(double now, const UiNotice& n) {
    if (now < n.shown_at) return 0;
    return fade_alpha(now, n.shown_at + n.hold, false, 0, 0.3);  // hidden after the hold, fades over 0.3 s
}

float notice_level(float volume) { return std::clamp(volume / 2, 0.0f, 1.0f); }

std::vector<UiAction> draw_ui(const UiState& s) {
    std::vector<UiAction> out;
    ImGuiIO& io = ImGui::GetIO();
    const float bar_h = ImGui::GetFrameHeight() * 2;
    bool hovered = io.MousePos.y >= io.DisplaySize.y - bar_h && io.MousePos.y <= io.DisplaySize.y;

    if (controls_visible(s.now, s.last_mouse, s.paused, hovered)) {
        ImGui::SetNextWindowPos({0, io.DisplaySize.y - bar_h});
        ImGui::SetNextWindowSize({io.DisplaySize.x, bar_h});
        ImGui::SetNextWindowBgAlpha(0.6f);
        ImGui::Begin("controls", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                              ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing);
        if (ImGui::Button(s.paused ? "Play" : "Pause", {ImGui::GetFontSize() * 4, 0}))
            out.push_back({UiAction::TogglePause});
        ImGui::SameLine();
        static double drag = 0;  // the seek slider's value while dragged; follows playback otherwise
        static bool dragging = false;
        // while dragging, the time shows where the seek will land
        ImGui::TextUnformatted((clock_text(dragging ? drag : s.pos) + " / " + (s.dur > 0 ? clock_text(s.dur) : "--:--")).c_str());
        ImGui::SameLine();
        // the right side: volume, VSR, open, fullscreen; the seek bar takes what's left
        const float right = ImGui::GetFontSize() * 26;
        ImGui::SetNextItemWidth(std::fmax(ImGui::GetContentRegionAvail().x - right, 50));
        ImGui::BeginDisabled(s.dur <= 0);
        double lo = 0, hi = std::fmax(s.dur, 1);
        if (!dragging) drag = std::clamp(s.pos, lo, hi);
        ImGui::SliderScalar("##seek", ImGuiDataType_Double, &drag, &lo, &hi, "", ImGuiSliderFlags_NoInput);
        dragging = ImGui::IsItemActive();
        if (ImGui::IsItemDeactivatedAfterEdit()) out.push_back({UiAction::Seek, drag});  // on release only
        ImGui::EndDisabled();
        ImGui::SameLine();
        float pct = s.volume * 100;
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 7);
        if (ImGui::SliderFloat("##volume", &pct, 0, 200, "vol %.0f%%", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput))
            out.push_back({UiAction::SetVolume, pct / 100});
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Volume (Up/Down)");
        ImGui::SameLine();
        ImGui::BeginDisabled(!s.vsr_available);
        if (ImGui::Button(!s.vsr_available ? "VSR n/a" : s.vsr_on ? "VSR on" : "VSR off")) out.push_back({UiAction::ToggleVsr});
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(s.vsr_available ? "Upscaling (V)" : "No upscaling for this video or screen");
        ImGui::SameLine();
        if (ImGui::Button("Open")) out.push_back({UiAction::Open});
        ImGui::SameLine();
        if (ImGui::Button(s.fullscreen ? "Window" : "Fullscreen")) out.push_back({UiAction::ToggleFullscreen});
        ImGui::End();
    }

    if (s.show_info) {
        ImGui::SetNextWindowPos({10, 10});
        ImGui::SetNextWindowBgAlpha(0.6f);
        ImGui::Begin("info", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                                          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings);
        ImGui::TextUnformatted(s.info.c_str());
        ImGui::End();
    }

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
        ImGui::OpenPopup("menu");
    if (ImGui::BeginPopup("menu")) {
        if (ImGui::MenuItem(s.paused ? "Play" : "Pause", "Space")) out.push_back({UiAction::TogglePause});
        if (ImGui::MenuItem("Open file...")) out.push_back({UiAction::Open});
        if (ImGui::BeginMenu("Recent files")) {
            if (s.recent.empty()) ImGui::TextDisabled("none");
            for (const auto& p : s.recent)  // missing files were dropped at startup
                if (ImGui::MenuItem(fs::path(p).filename().string().c_str())) out.push_back({UiAction::OpenPath, 0, p});
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Upscaling (VSR)", "V", s.vsr_on, s.vsr_available)) out.push_back({UiAction::ToggleVsr});
        if (ImGui::MenuItem("Fullscreen", "F", s.fullscreen)) out.push_back({UiAction::ToggleFullscreen});
        if (ImGui::MenuItem("Info", "I", s.show_info)) out.push_back({UiAction::ToggleInfo});
        ImGui::Separator();
        if (ImGui::MenuItem("Quit", "Q")) out.push_back({UiAction::Quit});
        ImGui::EndPopup();
    }
    return out;
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
    check(can_open(tmp) && !can_open("/tmp") && !can_open("/definitely/missing.mkv"));  // files only
    std::filesystem::remove(tmp);

    check(controls_visible(10, 9, false, false) && !controls_visible(10, 7, false, false));  // 2 s rule
    check(controls_visible(10, 0, true, false) && controls_visible(10, 0, false, true));    // paused, hovered
    check(restart_args({"open-player", "--model", "m.onnx", "a.mkv", "--screen", "2"}, "/x/My Movie (2012).mkv") ==
          (std::vector<std::string>{"open-player", "--model", "m.onnx", "/x/My Movie (2012).mkv", "--screen", "2"}));
    check(restart_args({"open-player", "a.mkv"}, "/b.mp4") == (std::vector<std::string>{"open-player", "/b.mp4"}));
    check(fade_alpha(10, 10, true, .15, .3) == 0 && std::fabs(fade_alpha(10.075, 10, true, .15, .3) - .5f) < 1e-3 && fade_alpha(10.2, 10, true, .15, .3) == 1);
    check(fade_alpha(10, 10, false, .15, .3) == 1 && std::fabs(fade_alpha(10.15, 10, false, .15, .3) - .5f) < 1e-3 && fade_alpha(10.4, 10, false, .15, .3) == 0);
    UiNotice n{nullptr, "Volume 80%", .4f, 5, 1};
    check(notice_alpha(4, n) == 0 && notice_alpha(5.5, n) == 1 && std::fabs(notice_alpha(6.15, n) - .5f) < 1e-3 && notice_alpha(6.4, n) == 0);
    n.shown_at = 6.2;  // a new notice replaces the fading one: full again
    check(notice_alpha(6.25, n) == 1);
    check(notice_level(0.8f) == 0.4f && notice_level(2.0f) == 1.0f && notice_level(-1) == 0);
    // headless ImGui: frames without any backend or present must not assert
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = {1920, 1080};
    io.IniFilename = nullptr;
    unsigned char* px;
    int fw, fh;
    io.Fonts->GetTexDataAsRGBA32(&px, &fw, &fh);
    for (double dur : {600.0, 0.0})    // normal and unknown-duration video
        for (int i = 0; i < 3; i++) {  // several frames, as when minimized
            ImGui::NewFrame();
            draw_ui({/*pos*/ 12, dur, /*now*/ 5, /*last_mouse*/ 4.5, /*paused*/ false, true, true, false, true, 0.8f, "info", {}});
            ImGui::Render();
        }
    // shortcut keys stay with the menu while it's open (Esc closes it instead of quitting)
    ImGui::NewFrame();
    check(!ui_has_keyboard());
    ImGui::OpenPopup("menu");
    check(ui_has_keyboard());
    ImGui::Render();
    ImGui::DestroyContext();

    // icon strings encode their codepoints
    for (auto [s, cp] : {std::pair{ICON_PLAY, CP_PLAY}, {ICON_PAUSE, CP_PAUSE}, {ICON_VOLUME, CP_VOLUME},
                         {ICON_VOLUME_OFF, CP_VOLUME_OFF}, {ICON_FOLDER, CP_FOLDER}, {ICON_MAXIMIZE, CP_MAXIMIZE},
                         {ICON_MINIMIZE, CP_MINIMIZE}, {ICON_SPARKLES, CP_SPARKLES}, {ICON_FORWARD_5, CP_FORWARD_5},
                         {ICON_BACKWARD_5, CP_BACKWARD_5}, {ICON_INFO, CP_INFO}, {ICON_X, CP_X}, {ICON_CLOCK, CP_CLOCK}}) {
        const unsigned char* u = (const unsigned char*)s;  // 3-byte sequence
        check(((u[0] & 0x0f) << 12 | (u[1] & 0x3f) << 6 | (u[2] & 0x3f)) == cp);
    }
    // fonts and theme: missing files fall back, bundled files load with every icon present
    ImGui::CreateContext();
    check(!load_theme("/definitely/missing") && !theme_loaded());  // fallback: default font stays usable
    ImGui::DestroyContext();
    ImGui::CreateContext();
    if (std::filesystem::exists(assets_dir() + "/fonts/Inter-Medium.ttf")) {
        check(load_theme(assets_dir() + "/fonts") && theme_loaded());
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = {1920, 1080};
        unsigned char* px;
        int fw, fh;
        io.Fonts->GetTexDataAsRGBA32(&px, &fw, &fh);
        for (ImWchar c : {CP_PLAY, CP_PAUSE, CP_VOLUME, CP_FOLDER, CP_MAXIMIZE, CP_SPARKLES, (ImWchar)'A', (ImWchar)0xE9})
            check(io.Fonts->Fonts[0]->IsGlyphInFont(c));
    } else printf("note: assets not found next to the binary, font checks skipped\n");
    ImGui::DestroyContext();

#undef check
    if (!fails) printf("ui checks passed\n");
    return fails ? 1 : 0;
}
