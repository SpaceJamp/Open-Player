#include "ui.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "collect.h"
#include "settings.h"
#include "logfile.h"
#include "update.h"
#include "imgui.h"
#include "imgui_internal.h"  // the self-test finds the volume panel's window
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
    const char* xdg = getenv("XDG_CONFIG_HOME");  // also on Windows: the tests point it at a temp folder
#ifdef _WIN32
    const char* appdata = getenv("APPDATA");
    fs::path base = xdg && *xdg ? fs::path(xdg) : appdata && *appdata ? fs::path(appdata) : fs::path(".");
#else
    const char* home = getenv("HOME");
    fs::path base = xdg && *xdg ? fs::path(xdg) : fs::path(home ? home : ".") / ".config";
#endif
    return (base / "open-player" / "recent.txt").string();
}

std::string clock_text(double t) {
    int s = (int)std::fmax(t, 0);
    char buf[32];
    if (s >= 3600) snprintf(buf, sizeof buf, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60);
    else snprintf(buf, sizeof buf, "%02d:%02d", s / 60, s % 60);
    return buf;
}

float ui_scale(float display_h) { return std::clamp(display_h / 1080.f, 1.f, 2.5f); }

std::string seek_notice_text(double delta, double target, double dur) {
    if (dur > 0) target = std::fmin(target, dur);
    return std::string(delta > 0 ? "+5 s" : "−5 s") + " · " + clock_text(target);
}

std::string quote_arg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\n\v\"") == std::string::npos) return a;
    std::string out = "\"";
    for (size_t i = 0;; i++) {
        size_t slashes = 0;
        while (i < a.size() && a[i] == '\\') i++, slashes++;
        if (i == a.size()) {  // backslashes before the closing quote must not escape it
            out.append(slashes * 2, '\\');
            break;
        }
        if (a[i] == '"') out.append(slashes * 2 + 1, '\\');
        else out.append(slashes, '\\');
        out += a[i];
    }
    return out + '"';
}

std::vector<std::string> restart_args(const std::vector<std::string>& argv, const std::string& video) {
    std::vector<std::string> out;  // without --start T and --paused: they belonged to the video being replaced
    for (size_t i = 0; i < argv.size(); i++) {
        if (i && argv[i] == "--start") i++;
        else if (!i || argv[i] != "--paused") out.push_back(argv[i]);
    }
    for (size_t i = 1; i < out.size(); i++) {
        if (out[i] == "--model" || out[i] == "--screen" || out[i] == "--collect") i++;  // skip the flag's value
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

bool controls_visible(double now, double last_mouse, bool paused, bool hovered, double hide_after) {
    return paused || hovered || now - last_mouse < hide_after;
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

namespace {
bool g_shown = false;        // the capsule's visibility target at the last draw
double g_changed_at = -1e9;  // when that target last changed (shifted so a reversed fade continues smoothly)
float g_alpha = 0;           // the capsule's opacity at the last draw
UiNotice g_notice;           // the last notice drawn
bool g_vol_open = false, g_vol_held = false;  // the volume panel

constexpr double FADE_IN = .15, FADE_OUT = .3;

ImU32 col(ImU32 c, float a = 1) { return ImGui::GetColorU32(c, a); }  // times the style alpha (the capsule fade)

ImVec2 text_size(float px, const char* t) { return ImGui::GetFont()->CalcTextSizeA(px, FLT_MAX, 0, t); }

void text_centred(ImDrawList* dl, float px, ImVec2 c, ImU32 colour, const char* t) {
    ImVec2 sz = text_size(px, t);
    dl->AddText(ImGui::GetFont(), px, {std::round(c.x - sz.x / 2), std::round(c.y - sz.y / 2)}, colour, t);
}

// Buttons are circles concentric with the capsule's ends; without the icon font, pills with the text label.
float button_w(const char* label, float h, float sc) {
    return theme_loaded() ? h : text_size(16 * sc, label).x + 24 * sc;
}

bool icon_button(const char* id, const char* icon, const char* label, ImVec2 p, float h, float sc) {
    const float w = button_w(label, h, sc);
    ImGui::SetCursorScreenPos(p);
    bool pressed = ImGui::InvisibleButton(id, {w, h});
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (ImGui::IsItemHovered())
        dl->AddRectFilled(p, {p.x + w, p.y + h}, col(IM_COL32(255, 255, 255, ImGui::IsItemActive() ? 52 : 30)), h / 2);
    text_centred(dl, theme_loaded() ? 20 * sc : 16 * sc, {p.x + w / 2, p.y + h / 2}, col(COL_TEXT),
                 theme_loaded() ? icon : label);
    return pressed;
}

struct Track {
    bool hovered, held, released;
    float at;  // the mouse's 0..1 position along the track
};
// A slim track centred on y, filled to frac; thicker with a knob on hover. Grey and inert when disabled.
Track track(const char* id, ImVec2 p, float w, float frac, float sc, bool enabled) {
    const float hit = 20 * sc;
    ImGui::SetCursorScreenPos({p.x, p.y - hit / 2});
    ImGui::BeginDisabled(!enabled);
    ImGui::InvisibleButton(id, {w, hit});
    Track t{ImGui::IsItemHovered(), ImGui::IsItemActive(), ImGui::IsItemDeactivated(),
            std::clamp((ImGui::GetIO().MousePos.x - p.x) / w, 0.0f, 1.0f)};
    ImGui::EndDisabled();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const bool big = enabled && (t.hovered || t.held);
    const float th = (big ? 6 : 4) * sc, x = p.x + w * std::clamp(frac, 0.0f, 1.0f);
    dl->AddRectFilled({p.x, p.y - th / 2}, {p.x + w, p.y + th / 2}, col(COL_TRACK, enabled ? 1 : .5f), th / 2);
    if (!enabled) return t;
    if (x > p.x) dl->AddRectFilled({p.x, p.y - th / 2}, {x, p.y + th / 2}, col(COL_PLAYED), th / 2);
    if (big) dl->AddCircleFilled({x, p.y}, 6 * sc, col(IM_COL32_WHITE));
    return t;
}

// The centre card: an icon, one line of text and, for a level, a bar under it.
void draw_notice(const UiNotice& n, float sc, float a) {
    const ImGuiIO& io = ImGui::GetIO();
    const char* icon = theme_loaded() ? n.icon : nullptr;
    std::string text = n.text;
    if (text.empty() && !icon && n.icon)  // icon-only notice without the icon font: say it
        text = !strcmp(n.icon, ICON_PAUSE) ? "Pause" : !strcmp(n.icon, ICON_PLAY) ? "Play" : "";
    if (text.empty() && !icon) return;
    const bool only_icon = text.empty(), bar = n.level >= 0;
    const float tp = 18 * sc, ip = (only_icon ? 36 : 24) * sc, gap = 10 * sc;
    const float padx = (only_icon ? 18 : 22) * sc, pady = (only_icon ? 18 : 14) * sc;
    const ImVec2 ts = only_icon ? ImVec2{0, 0} : text_size(tp, text.c_str());
    const float iw = icon ? text_size(ip, icon).x : 0, row_w = iw + (icon && !only_icon ? gap : 0) + ts.x;
    const float row_h = std::fmax(icon ? ip : 0, ts.y), bar_h = 4 * sc, bar_gap = 12 * sc;
    // long file names are clipped to the window rather than overflowing it
    const float w = std::fmin(std::fmax(row_w, bar ? 200 * sc : 0), io.DisplaySize.x - 2 * padx - 32 * sc);
    const float h = row_h + (bar ? bar_gap + bar_h : 0);
    const ImVec2 c{io.DisplaySize.x / 2, io.DisplaySize.y / 2};
    const ImVec2 p0{std::round(c.x - w / 2 - padx), std::round(c.y - h / 2 - pady)};
    const ImVec2 p1{p0.x + w + 2 * padx, p0.y + h + 2 * pady};
    ImDrawList* dl = ImGui::GetBackgroundDrawList();  // over the video, under every window
    dl->AddRectFilled(p0, p1, ImGui::GetColorU32(COL_CAPSULE, a), 14 * sc);
    float x = c.x - std::fmin(row_w, w) / 2;
    const float ry = p0.y + pady + row_h / 2;
    const ImU32 tc = ImGui::GetColorU32(COL_TEXT, a);
    if (icon) {
        dl->AddText(ImGui::GetFont(), ip, {std::round(x), std::round(ry - ip / 2)}, tc, icon);
        x += iw + gap;
    }
    if (!only_icon) {
        const ImVec4 clip{p0.x + padx, p0.y, p1.x - padx, p1.y};
        dl->AddText(ImGui::GetFont(), tp, {std::round(x), std::round(ry - ts.y / 2)}, tc, text.c_str(), nullptr, 0,
                    &clip);
    }
    if (bar) {
        const ImVec2 b0{p0.x + padx, p1.y - pady - bar_h}, b1{p1.x - padx, p1.y - pady};
        dl->AddRectFilled(b0, b1, ImGui::GetColorU32(COL_TRACK, a), bar_h / 2);
        const float bx = b0.x + (b1.x - b0.x) * std::clamp(n.level, 0.0f, 1.0f);
        if (bx > b0.x) dl->AddRectFilled(b0, {bx, b1.y}, ImGui::GetColorU32(COL_PLAYED, a), bar_h / 2);
    }
}

// --- the settings card: drawn at the window's cursor, each piece ends with an item that moves the cursor down ---
constexpr ImU32 COL_SUBTLE = IM_COL32(255, 255, 255, 20), COL_DIM = IM_COL32(240, 242, 248, 150);

// A section title: icon and name, dimmed, with space above
void card_heading(const char* icon, const char* text, float sc) {
    ImGui::Dummy({1, 6 * sc});
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float px = 13 * sc;
    std::string t = theme_loaded() ? std::string(icon) + "  " + text : text;
    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), px, p, col(COL_DIM), t.c_str());
    ImGui::Dummy({1, px + 4 * sc});
}

// A row: its label on the left, the control right-aligned by the caller at the returned position; when both don't
// fit side by side (a narrow card) the control goes under the label. h: the control's height; row_h gets the row's.
ImVec2 card_row(const char* label, float h, float control_w, float sc, float& row_h) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x, px = 15 * sc;
    const bool stack = text_size(px, label).x + 12 * sc + control_w > w;
    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), px, {p.x, std::round(stack ? p.y : p.y + (h - px) / 2)},
                                        col(COL_TEXT), label);
    row_h = stack ? px + 6 * sc + h : h;
    return stack ? ImVec2{p.x, p.y + px + 6 * sc} : ImVec2{p.x + w - control_w, p.y};
}
void card_row_end(ImVec2 row_start, float h) {
    ImGui::SetCursorScreenPos(row_start);
    ImGui::Dummy({ImGui::GetContentRegionAvail().x, h});
}

struct Seg {
    const char* label;
    bool enabled;
    const char* why;  // tooltip when disabled
};
float segmented_w(const std::vector<Seg>& segs, float sc) {
    float w = 0;
    for (auto& s : segs) w += text_size(14 * sc, s.label).x + 24 * sc;
    return w + 4 * sc;
}
// Pill-shaped choice; the selected segment is violet. Returns the clicked index, or -1.
int segmented(const char* id, ImVec2 p, const std::vector<Seg>& segs, int selected, float sc) {
    const float h = 28 * sc, px = 14 * sc, inner = 2 * sc;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, {p.x + segmented_w(segs, sc), p.y + h}, col(COL_SUBTLE), h / 2);
    float x = p.x + inner;
    int clicked = -1;
    for (int i = 0; i < (int)segs.size(); i++) {
        const float w = text_size(px, segs[i].label).x + 24 * sc;
        const ImVec2 a{x, p.y + inner}, b{x + w, p.y + h - inner};
        ImGui::SetCursorScreenPos(a);
        ImGui::PushID(id);
        ImGui::PushID(i);
        ImGui::BeginDisabled(!segs[i].enabled);
        if (ImGui::InvisibleButton("##seg", {w, b.y - a.y}) && i != selected) clicked = i;
        ImGui::EndDisabled();
        const bool hov = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
        ImGui::PopID();
        ImGui::PopID();
        if (i == selected) dl->AddRectFilled(a, b, col(COL_VSR_ON), (b.y - a.y) / 2);
        else if (hov && segs[i].enabled) dl->AddRectFilled(a, b, col(IM_COL32(0x3f, 0xd0, 0xe0, 48)), (b.y - a.y) / 2);
        if (hov && !segs[i].enabled && segs[i].why) ImGui::SetTooltip("%s", segs[i].why);
        text_centred(dl, px, {x + w / 2, p.y + h / 2}, col(COL_TEXT, segs[i].enabled ? 1 : .35f), segs[i].label);
        x += w;
    }
    return clicked;
}

// On/off switch. Returns true when clicked.
bool toggle(const char* id, ImVec2 p, bool on, float sc) {
    const float w = 40 * sc, h = 22 * sc, r = h / 2 - 3 * sc;
    ImGui::SetCursorScreenPos(p);
    const bool pressed = ImGui::InvisibleButton(id, {w, h});
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, {p.x + w, p.y + h}, col(on ? COL_VSR_ON : COL_TRACK), h / 2);
    if (ImGui::IsItemHovered()) dl->AddRect(p, {p.x + w, p.y + h}, col(COL_PLAYED), h / 2, 0, 1.5f * sc);
    dl->AddCircleFilled({on ? p.x + w - h / 2 : p.x + h / 2, p.y + h / 2}, r, col(IM_COL32_WHITE));
    return pressed;
}

// capsule_top: the card stays above the control capsule, which stays visible and clickable while it's open
void draw_settings(const UiState& s, std::vector<UiAction>& out, float capsule_top) {
    const ImGuiIO& io = ImGui::GetIO();
    const float sc = s.scale, W = std::fmin(520 * sc, io.DisplaySize.x - 32 * sc);
    const float top = 8 * sc, bottom = std::fmax(capsule_top - 8 * sc, top + 40 * sc);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {14 * sc, 12 * sc});
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8 * sc);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(20, 22, 30, 235));  // opaque enough to read over video
    ImGui::SetNextWindowPos({io.DisplaySize.x / 2, (top + bottom) / 2}, ImGuiCond_Always, {.5f, .5f});
    ImGui::SetNextWindowSizeConstraints({W, 0}, {W, std::fmin(io.DisplaySize.y * .8f, bottom - top)});
    ImGui::Begin("settings", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float bh = 32 * sc, rh = 30 * sc;

    // header: gear, title, X
    {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float aw = ImGui::GetContentRegionAvail().x, px = 18 * sc;
        std::string t = theme_loaded() ? std::string(ICON_SETTINGS) + "  Settings" : "Settings";
        dl->AddText(ImGui::GetFont(), px, {p.x, std::round(p.y + (bh - px) / 2)}, col(COL_TEXT), t.c_str());
        if (icon_button("##close", ICON_X, "Close", {p.x + aw - button_w("Close", bh, sc), p.y}, bh, sc))
            out.push_back({UiAction::ToggleSettings});
        card_row_end(p, bh);
    }

    card_heading(ICON_SPARKLES, "UPSCALING", sc);
    if (!s.vsr_unavailable.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, COL_DIM);
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(("Not available: " + s.vsr_unavailable).c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    } else {
        std::vector<Seg> segs = {{"NVIDIA RTX", s.rtx_available, "Not available on this GPU or driver"},
                                 {"Pixel Flow", s.pxf_available, "No Pixel Flow model (vsr.onnx) loaded"},
                                 {"Off", true, nullptr}};
#ifndef HAVE_RTX_VSR
        segs.erase(segs.begin());  // this build has no RTX VSR
        const int base = 1;
#else
        const int base = 0;
#endif
        const float sw = segmented_w(segs, sc);
        float rh = 0;
        ImVec2 row = ImGui::GetCursorScreenPos(), c = card_row("Upscaler", 28 * sc, sw, sc, rh);
        int i = segmented("up", c, segs, (int)s.settings.upscaler - base, sc);
        if (i >= 0) out.push_back({UiAction::SetUpscaler, (double)(i + base)});
        card_row_end(row, rh + 4 * sc);
    }

    card_heading(ICON_CPU, "DECODING", sc);
    {
        std::vector<Seg> segs = {{"GPU (NVDEC)", s.nvdec_available, "NVDEC isn't available on this GPU or driver"},
                                 {"CPU", true, nullptr}};
        const float sw = segmented_w(segs, sc);
        float rh = 0;
        ImVec2 row = ImGui::GetCursorScreenPos(), c = card_row("Decoder", 28 * sc, sw, sc, rh);
        int i = segmented("dec", c, segs, s.settings.cpu_decode ? 1 : 0, sc);
        if (i >= 0) out.push_back({UiAction::SetDecoder, (double)i});
        card_row_end(row, rh + 4 * sc);
        ImGui::PushStyleColor(ImGuiCol_Text, COL_DIM);
        ImGui::TextUnformatted("Changing it reopens the video where it is.");
        ImGui::PopStyleColor();
    }

    card_heading(ICON_DESKTOP, "DISPLAY", sc);
    if (s.screens.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, COL_DIM);
        ImGui::TextUnformatted("No monitors reported.");
        ImGui::PopStyleColor();
    }
    for (int i = 0; i < (int)s.screens.size(); i++) {  // one row per monitor; picking one reopens the video there
        const ScreenInfo& m = s.screens[i];
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float aw = ImGui::GetContentRegionAvail().x, px = 15 * sc, tpx = 12 * sc;
        ImGui::PushID(i);
        if (ImGui::InvisibleButton("##screen", {aw, rh}) && i != s.screen_index)
            out.push_back({UiAction::SetScreen, (double)i});
        const bool hov = ImGui::IsItemHovered();
        ImGui::PopID();
        if (hov) dl->AddRectFilled(p, {p.x + aw, p.y + rh}, col(COL_SUBTLE), 8 * sc);
        const float cy = p.y + rh / 2, r = 7 * sc;
        dl->AddCircle({p.x + 10 * sc + r, cy}, r, col(i == s.screen_index ? COL_VSR_ON : COL_DIM), 0, 1.5f * sc);
        if (i == s.screen_index) dl->AddCircleFilled({p.x + 10 * sc + r, cy}, r - 3 * sc, col(COL_VSR_ON));
        char mode[64];
        snprintf(mode, sizeof mode, "%dx%d @ %.0f Hz", m.w, m.h, m.hz);
        std::string tags = std::string(m.primary ? "primary" : "") + (m.primary && i == s.playing_on ? " · " : "") +
                           (i == s.playing_on ? "playing here" : "");
        const float mode_w = text_size(px, mode).x, x_num = p.x + 10 * sc + 2 * r + 10 * sc, x_name = x_num + 20 * sc;
        float tags_w = text_size(tpx, tags.c_str()).x;
        if (p.x + aw - 8 * sc - tags_w - 14 * sc - mode_w < x_name + 60 * sc) tags.clear(), tags_w = 0;  // narrow card
        const float x_tags = p.x + aw - 8 * sc - tags_w, x_mode = std::fmax(x_tags - 14 * sc - mode_w, x_name);
        const ImVec4 clip{x_name, p.y, x_mode - 10 * sc, p.y + rh};  // long names are cut, not overlapped
        dl->AddText(ImGui::GetFont(), px, {x_num, std::round(cy - px / 2)}, col(COL_DIM), std::to_string(i + 1).c_str());
        if (clip.z > clip.x + 20 * sc)  // no room for the name: the number and the mode say which it is
            dl->AddText(ImGui::GetFont(), px, {x_name, std::round(cy - px / 2)}, col(COL_TEXT), m.name.c_str(), nullptr,
                        0, &clip);
        dl->AddText(ImGui::GetFont(), px, {x_mode, std::round(cy - px / 2)}, col(COL_TEXT), mode);
        dl->AddText(ImGui::GetFont(), tpx, {x_tags, std::round(cy - tpx / 2)}, col(COL_PLAYED), tags.c_str());
    }
    {
        float rh = 0;
        ImVec2 row = ImGui::GetCursorScreenPos(), c = card_row("Start fullscreen", 22 * sc, 40 * sc, sc, rh);
        if (toggle("##fs", c, s.settings.start_fullscreen, sc))
            out.push_back({UiAction::SetStartFullscreen, s.settings.start_fullscreen ? 0.0 : 1.0});
        card_row_end(row, rh + 4 * sc);
    }

    card_heading(ICON_ADJUSTMENTS, "PLAYBACK & CONTROLS", sc);
    {
        float rh = 0;
        ImVec2 row = ImGui::GetCursorScreenPos(), c = card_row("Remember volume", 22 * sc, 40 * sc, sc, rh);
        if (toggle("##vol", c, s.settings.remember_volume, sc))
            out.push_back({UiAction::SetRememberVolume, s.settings.remember_volume ? 0.0 : 1.0});
        card_row_end(row, rh + 4 * sc);
    }
    {
        static const int secs[] = {1, 2, 3, 5};
        std::vector<Seg> segs = {{"1 s", true, nullptr}, {"2 s", true, nullptr}, {"3 s", true, nullptr},
                                 {"5 s", true, nullptr}};
        int sel = int(std::find(std::begin(secs), std::end(secs), s.settings.hide_after) - std::begin(secs));
        const float sw = segmented_w(segs, sc);
        float rh = 0;
        ImVec2 row = ImGui::GetCursorScreenPos(), c = card_row("Hide controls after", 28 * sc, sw, sc, rh);
        int i = segmented("hide", c, segs, sel, sc);
        if (i >= 0) out.push_back({UiAction::SetHideAfter, (double)secs[i]});
        card_row_end(row, rh + 4 * sc);
    }
    {
        float rh = 0;
        ImVec2 row = ImGui::GetCursorScreenPos(), c = card_row("Show info overlay at start", 22 * sc, 40 * sc, sc, rh);
        if (toggle("##info", c, s.settings.info_at_start, sc))
            out.push_back({UiAction::SetInfoAtStart, s.settings.info_at_start ? 0.0 : 1.0});
        card_row_end(row, rh + 4 * sc);
    }

    card_heading(ICON_INFO, "ABOUT", sc);
    ImGui::PushStyleColor(ImGuiCol_Text, COL_DIM);
    ImGui::PushTextWrapPos(0);
    ImGui::TextUnformatted(about_text(s.build, s.log_path).c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    // a click anywhere outside every window (the video) closes the card; the capsule and the menu don't
    // (a button pressed this frame is "active", which hides the hover without AllowWhenBlockedByActiveItem)
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
        !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId))
        out.push_back({UiAction::ToggleSettings});
}
}  // namespace

std::string about_text(const std::string& build, const std::string& log) {
    std::string t = "Open Player " + (build.empty() ? std::string("build unknown") : build) +
                    "\nPixel Flow (PXF): Open Player's own upscaler.";
#ifdef HAVE_RTX_VSR
    t += "\nThis software uses the NVIDIA RTX Video SDK. NVIDIA, GeForce RTX and NVIDIA RTX are trademarks of "
         "NVIDIA Corporation.";
#endif
    return t + "\nLicensed under the GNU AGPL v3." + (log.empty() ? "" : "\nLog: " + log);
}

std::string vsr_badge_text(bool available, bool on, bool rtx, int factor) {
    return !available ? "" : on ? (rtx ? "RTX " : "PXF ") + std::to_string(factor) + "x" : "VSR off";
}

bool ui_animating(double now) {
    return now - g_changed_at < (g_shown ? FADE_IN : FADE_OUT) || notice_alpha(now, g_notice) > 0;
}

bool ui_controls_hidden() { return g_alpha == 0; }

std::vector<UiAction> draw_ui(const UiState& s) {
    std::vector<UiAction> out;
    ImGuiIO& io = ImGui::GetIO();
    const float sc = s.scale;
    ImGui::PushFont(nullptr, 16 * sc);
    const float fs = ImGui::GetFontSize();
    const bool icons = theme_loaded();

    // the capsule: 78% wide up to 1100, 48 tall, 14 above the bottom (at scale 1)
    const float cw = std::fmin(io.DisplaySize.x * .78f, 1100 * sc), ch = 48 * sc;
    const ImVec2 c0{std::round((io.DisplaySize.x - cw) / 2), std::round(io.DisplaySize.y - 14 * sc - ch)};
    const ImVec2 c1{c0.x + cw, c0.y + ch};
    const float cy = c0.y + ch / 2, pad = 8 * sc, bh = 32 * sc, gap = 6 * sc, seek_gap = 12 * sc;
    const bool hovered = g_vol_open || ImGui::IsMouseHoveringRect(c0, c1, false);
    const bool vis = controls_visible(s.now, s.last_mouse, s.paused, hovered || s.settings_open, s.settings.hide_after);
    if (vis != g_shown) {  // start the new fade from the current opacity, so a reversal doesn't jump
        g_changed_at = s.now - (vis ? g_alpha * FADE_IN : (1 - g_alpha) * FADE_OUT);
        g_shown = vis;
    }
    g_alpha = fade_alpha(s.now, g_changed_at, vis, FADE_IN, FADE_OUT);

    static double drag = 0;  // the seek target while dragged
    static bool dragging = false;
    bool vol_hover = false;
    if (g_alpha > 0) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, g_alpha);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
        const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                       ImGuiWindowFlags_NoBackground;
        ImGui::SetNextWindowPos(c0);
        ImGui::SetNextWindowSize({cw, ch});
        ImGui::Begin("controls", nullptr, flags);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(c0, c1, col(COL_CAPSULE), ch / 2);

        // widths first: the seek bar takes what the rest leaves. Both times get a fixed slot (digits as zeros) so
        // the bar doesn't twitch as proportional digits change.
        const std::string elapsed = clock_text(dragging ? drag : s.pos), total = s.dur > 0 ? clock_text(s.dur) : "--:--";
        std::string slot = clock_text(std::fmax(s.dur, s.pos));
        for (char& c : slot) c = isdigit((unsigned char)c) ? '0' : c;
        const float tw = std::fmax(text_size(fs, slot.c_str()).x, text_size(fs, total.c_str()).x);
        const char* play_l = s.paused ? "Play" : "Pause";
        const char* vol_l = "Volume";
        const char* fs_l = s.fullscreen ? "Window" : "Fullscreen";
        const std::string vsr_s = vsr_badge_text(s.vsr_available, s.vsr_on, s.vsr_rtx, s.vsr_factor);
        const char* vsr_l = vsr_s.c_str();
        const float badge_px = 14 * sc, badge_w = text_size(badge_px, vsr_l).x + 20 * sc, badge_h = 24 * sc;
        float x = c0.x + pad;
        const float seek_x = x + button_w(play_l, bh, sc) + gap + tw + seek_gap;
        const float right = seek_gap + tw + gap + button_w(vol_l, bh, sc) + gap +
                            (s.vsr_available ? badge_w + gap : 0) + button_w("Settings", bh, sc) + gap +
                            button_w("Open", bh, sc) + gap +
                            button_w(fs_l, bh, sc) + pad;
        const float seek_w = std::fmax(c1.x - right - seek_x, 1);

        if (icon_button("##play", s.paused ? ICON_PLAY : ICON_PAUSE, play_l, {x, cy - bh / 2}, bh, sc))
            out.push_back({UiAction::TogglePause});
        x += button_w(play_l, bh, sc) + gap;
        const ImU32 tc = col(COL_TEXT);
        dl->AddText({std::round(x + tw - text_size(fs, elapsed.c_str()).x), std::round(cy - fs / 2)}, tc,
                    elapsed.c_str());  // right-aligned against the bar

        const bool known = s.dur > 0;
        if (!dragging) drag = std::clamp(s.pos, 0.0, std::fmax(s.dur, 0.0));
        Track t = track("##seek", {seek_x, cy}, seek_w, known ? (float)(drag / s.dur) : 0, sc, known);
        if (t.held) drag = t.at * s.dur;  // the time text shows where the seek will land
        if (t.released && known) out.push_back({UiAction::Seek, drag});  // on release only, click or drag
        dragging = t.held;
        if (known && (t.hovered || t.held)) {  // the time under the mouse, above it
            const std::string at = clock_text(t.at * s.dur);
            const float px = 14 * sc;
            const ImVec2 sz = text_size(px, at.c_str()), tip{sz.x + 16 * sc, sz.y + 8 * sc};
            const float mx = std::clamp(io.MousePos.x, tip.x / 2, io.DisplaySize.x - tip.x / 2);
            const ImVec2 p0{std::round(mx - tip.x / 2), std::round(c0.y - 8 * sc - tip.y)};
            ImDrawList* fg = ImGui::GetForegroundDrawList();
            fg->AddRectFilled(p0, {p0.x + tip.x, p0.y + tip.y}, col(COL_CAPSULE), tip.y / 2);
            text_centred(fg, px, {p0.x + tip.x / 2, p0.y + tip.y / 2}, tc, at.c_str());
        }
        x = seek_x + seek_w + seek_gap;
        dl->AddText({std::round(x), std::round(cy - fs / 2)}, col(COL_TEXT, known ? 1 : .6f), total.c_str());
        x += tw + gap;

        icon_button("##volume", s.volume <= 0 ? ICON_VOLUME_OFF : ICON_VOLUME, vol_l, {x, cy - bh / 2}, bh, sc);
        vol_hover = ImGui::IsItemHovered();
        const float vol_cx = x + button_w(vol_l, bh, sc) / 2;
        x += button_w(vol_l, bh, sc) + gap;

        if (s.vsr_available) {  // no badge when this video isn't upscaled
            const ImVec2 b0{x, std::round(cy - badge_h / 2)}, b1{x + badge_w, b0.y + badge_h};
            ImGui::SetCursorScreenPos(b0);
            if (ImGui::InvisibleButton("##vsr", {badge_w, badge_h})) out.push_back({UiAction::ToggleVsr});
            dl->AddRectFilled(b0, b1, col(s.vsr_on ? COL_VSR_ON : COL_VSR_OFF), badge_h / 2);
            if (ImGui::IsItemHovered()) {
                dl->AddRectFilled(b0, b1, col(IM_COL32(255, 255, 255, 36)), badge_h / 2);
                ImGui::SetTooltip("Upscaling (V)");
            }
            text_centred(dl, badge_px, {x + badge_w / 2, cy}, col(IM_COL32_WHITE), vsr_l);
            x += badge_w + gap;
        }
        if (icon_button("##settings", ICON_SETTINGS, "Settings", {x, cy - bh / 2}, bh, sc))
            out.push_back({UiAction::ToggleSettings});
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Settings (S)");
        x += button_w("Settings", bh, sc) + gap;
        if (icon_button("##open", ICON_FOLDER, "Open", {x, cy - bh / 2}, bh, sc)) out.push_back({UiAction::Open});
        x += button_w("Open", bh, sc) + gap;
        if (icon_button("##full", s.fullscreen ? ICON_MINIMIZE : ICON_MAXIMIZE, fs_l, {x, cy - bh / 2}, bh, sc))
            out.push_back({UiAction::ToggleFullscreen});
        ImGui::End();

        // volume panel: above the icon, open while the icon, the panel or the gap between them is hovered (down to
        // the icon's top, so the strip above it inside the capsule doesn't close the panel on the way up)
        const float pw = 176 * sc, ph = 36 * sc;
        const ImVec2 p0{std::round(std::clamp(vol_cx - pw / 2, 0.0f, io.DisplaySize.x - pw)), c0.y - 8 * sc - ph};
        if (g_vol_open) vol_hover |= g_vol_held || ImGui::IsMouseHoveringRect(p0, {p0.x + pw, c0.y + pad}, false);
        if (vol_hover) {
            ImGui::SetNextWindowPos(p0);
            ImGui::SetNextWindowSize({pw, ph});
            ImGui::Begin("volume", nullptr, flags);
            ImGui::GetWindowDrawList()->AddRectFilled(p0, {p0.x + pw, p0.y + ph}, col(COL_CAPSULE), 10 * sc);
            const float px = 14 * sc, tx = 40 * sc, inset = 14 * sc;
            Track v = track("##vol", {p0.x + inset, p0.y + ph / 2}, pw - 2 * inset - tx, s.volume / 2, sc, true);
            g_vol_held = v.held;
            const float nv = std::round(v.at * 200) / 100;  // whole percents
            if (v.held && std::fabs(nv - s.volume) > 1e-4f) out.push_back({UiAction::SetVolume, nv});
            const std::string pct = std::to_string((int)std::lround(s.volume * 100)) + "%";
            const ImVec2 sz = text_size(px, pct.c_str());
            ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), px,
                                                {std::round(p0.x + pw - inset - sz.x), std::round(p0.y + (ph - sz.y) / 2)},
                                                tc, pct.c_str());
            ImGui::End();
        }
        ImGui::PopStyleVar(3);
    }
    g_vol_open = vol_hover;
    if (!vol_hover) g_vol_held = false;

    g_notice = s.notice;
    if (float a = notice_alpha(s.now, s.notice); a > 0) draw_notice(s.notice, sc, a);

    // the rounded card look, shared by the info card and the menu
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12 * sc);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 10 * sc);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {14 * sc, 10 * sc});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8 * sc, 6 * sc});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * sc, 4 * sc});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, COL_CAPSULE);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, IM_COL32(20, 22, 30, 235));  // the capsule colour, opaque enough to read
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, IM_COL32(0x3f, 0xd0, 0xe0, 64));
    ImGui::PushStyleColor(ImGuiCol_Separator, IM_COL32(255, 255, 255, 30));
    if (s.show_info) {
        ImGui::SetNextWindowPos({10 * sc, 10 * sc});
        ImGui::Begin("info", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                                          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings);
        ImGui::TextUnformatted(s.info.c_str());
        ImGui::End();
    }
    if (s.settings_open) draw_settings(s, out, c0.y);

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
        ImGui::OpenPopup("menu");
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {8 * sc, 8 * sc});
    auto item = [&](const char* icon, const char* text) { return icons ? std::string(icon) + "  " + text : text; };
    if (ImGui::BeginPopup("menu")) {
        if (ImGui::MenuItem(item(s.paused ? ICON_PLAY : ICON_PAUSE, s.paused ? "Play" : "Pause").c_str(), "Space"))
            out.push_back({UiAction::TogglePause});
        if (ImGui::MenuItem(item(ICON_FOLDER, "Open file...").c_str())) out.push_back({UiAction::Open});
        if (ImGui::BeginMenu(item(ICON_CLOCK, "Recent files").c_str())) {
            if (s.recent.empty()) ImGui::TextDisabled("none");
            for (const auto& p : s.recent)  // missing files were dropped at startup
                if (ImGui::MenuItem(fs::path(p).filename().string().c_str())) out.push_back({UiAction::OpenPath, 0, p});
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem(item(ICON_SPARKLES, "Upscaling (VSR)").c_str(), "V", s.vsr_on, s.vsr_available))
            out.push_back({UiAction::ToggleVsr});
        if (ImGui::MenuItem(item(s.fullscreen ? ICON_MINIMIZE : ICON_MAXIMIZE, "Fullscreen").c_str(), "F", s.fullscreen))
            out.push_back({UiAction::ToggleFullscreen});
        if (ImGui::MenuItem(item(ICON_INFO, "Info").c_str(), "I", s.show_info)) out.push_back({UiAction::ToggleInfo});
        if (ImGui::MenuItem(item(ICON_SETTINGS, "Settings").c_str(), "S", s.settings_open))
            out.push_back({UiAction::ToggleSettings});
        ImGui::Separator();
        if (ImGui::MenuItem(item(ICON_X, "Quit").c_str(), "Q")) out.push_back({UiAction::Quit});
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar(8);
    ImGui::PopStyleColor(4);
    ImGui::PopFont();
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
    // round trip through a temp file, including a path with spaces and non-ASCII
    std::string tmp = (std::filesystem::temp_directory_path() / "open-player-uitest-recent.txt").string();
    std::string odd = "/mnt/c/Users/Yeti/Vid\xc3\xa9os/My Movie (2012).mkv";
    save_recent(tmp, {odd, "/b.mp4"});
    save_recent(tmp + "/under-a-file/recent.txt", l);  // unwritable (parent is a file) -> no crash
    check(load_recent(tmp) == (std::vector<std::string>{odd, "/b.mp4"}));
    check(existing_only({tmp, "/definitely/missing.mkv"}) == std::vector<std::string>{tmp});
    check(can_open(tmp) && !can_open(std::filesystem::temp_directory_path().string()) && !can_open("/definitely/missing.mkv"));  // files only
    std::filesystem::remove(tmp);

    check(ui_scale(1080) == 1 && ui_scale(2160) == 2 && ui_scale(720) == 1 && ui_scale(4320) == 2.5f);  // logical points, clamped 1-2.5
    check(seek_notice_text(5, 100, 90) == "+5 s · " + clock_text(90));   // target past the end
    check(seek_notice_text(-5, -2, 90) == "−5 s · 00:00");              // before the start
    check(seek_notice_text(5, 100, 0) == "+5 s · " + clock_text(100));  // unknown duration: unclamped above
    check(seek_notice_text(5, 40, 90) == "+5 s · 00:40");

    check(controls_visible(10, 9, false, false) &&!controls_visible(10, 7, false, false));  // 2 s rule
    check(controls_visible(10, 0, true, false) && controls_visible(10, 0, false, true));    // paused, hovered
    check(controls_visible(10, 7.5, false, false, 3) && !controls_visible(10, 7.5, false, false, 2));  // hide_after
    check(restart_args({"open-player", "--model", "m.onnx", "a.mkv", "--screen", "2"}, "/x/My Movie (2012).mkv") ==
          (std::vector<std::string>{"open-player", "--model", "m.onnx", "/x/My Movie (2012).mkv", "--screen", "2"}));
    check(restart_args({"open-player", "a.mkv"}, "/b.mp4") == (std::vector<std::string>{"open-player", "/b.mp4"}));
    check(restart_args({"open-player", "a.mkv", "--start", "61.5", "--paused"}, "/b.mp4") ==  // they were a.mkv's
          (std::vector<std::string>{"open-player", "/b.mp4"}));
    check(restart_args({"open-player", "--collect", "pairs", "a.mkv"}, "/b.mp4") ==  // a folder, not the video
          (std::vector<std::string>{"open-player", "--collect", "pairs", "/b.mp4"}));
    // the pair files train.py --pairs reads: header back as written
    {
        std::string pf = (std::filesystem::temp_directory_path() / "open-player-uitest.pair").string();
        std::vector<uint8_t> lr(2 * 3 * 4 * 4, 7), hr(2 * 3 * 8 * 8, 9);
        int T = 0, lw = 0, lh = 0, hw = 0, hh = 0;
        check(write_pair_file(pf, 2, 4, 4, 8, 8, lr.data(), hr.data()) && read_pair_header(pf, T, lw, lh, hw, hh) &&
              T == 2 && lw == 4 && lh == 4 && hw == 8 && hh == 8);
        check(std::filesystem::file_size(pf) == 8 + 5 * 4 + lr.size() + hr.size());
        std::filesystem::remove(pf);
    }
    // Windows restart: one argument each after the C runtime splits the command line again
    check(quote_arg("plain.mkv") == "plain.mkv" && quote_arg("") == "\"\"");
    // a test run must never write the user's own recent list
#ifdef _WIN32
    _putenv_s("XDG_CONFIG_HOME", "uitest-config");
#else
    setenv("XDG_CONFIG_HOME", "uitest-config", 1);
#endif
    check(recent_file().rfind("uitest-config", 0) == 0);
    check(quote_arg("C:\\Videos\\My Movie (2012).mkv") == "\"C:\\Videos\\My Movie (2012).mkv\"");
    check(quote_arg("C:\\dir with space\\") == "\"C:\\dir with space\\\\\"");  // trailing backslash doubled
    check(quote_arg("say \"hi\"") == "\"say \\\"hi\\\"\"");
    check(fade_alpha(10, 10, true, .15, .3) == 0 && std::fabs(fade_alpha(10.075, 10, true, .15, .3) - .5f) < 1e-3 && fade_alpha(10.2, 10, true, .15, .3) == 1);
    check(fade_alpha(10, 10, false, .15, .3) == 1 && std::fabs(fade_alpha(10.15, 10, false, .15, .3) - .5f) < 1e-3 && fade_alpha(10.4, 10, false, .15, .3) == 0);
    UiNotice n{nullptr, "Volume 80%", .4f, 5, 1};
    check(notice_alpha(4, n) == 0 && notice_alpha(5.5, n) == 1 && std::fabs(notice_alpha(6.15, n) - .5f) < 1e-3 && notice_alpha(6.4, n) == 0);
    n.shown_at = 6.2;  // a new notice replaces the fading one: full again
    check(notice_alpha(6.25, n) == 1);
    check(vsr_badge_text(true, true, false, 2) == "PXF 2x" && vsr_badge_text(true, true, true, 2) == "RTX 2x" &&
          vsr_badge_text(true, false, true, 2) == "VSR off" && vsr_badge_text(false, false, false, 2).empty());
    check(notice_level(0.8f) == 0.4f && notice_level(2.0f) == 1.0f && notice_level(-1) == 0);
    // headless ImGui: frames without any backend or present must not assert, themed or not, at every scale
    for (bool themed : {false, true}) {
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        if (themed && !load_theme(assets_dir() + "/fonts")) {
            printf("note: themed headless frames skipped\n");
            ImGui::DestroyContext();
            continue;
        }
        unsigned char* px;
        int fw, fh;
        io.Fonts->GetTexDataAsRGBA32(&px, &fw, &fh);
        std::vector<UiAction> acts;
        auto frames = [&](UiState st, int n = 3, double step = .2) {  // several frames, as when minimized
            for (int i = 0; i < n; i++, st.now += step) {
                ImGui::NewFrame();
                auto a = draw_ui(st);
                acts.insert(acts.end(), a.begin(), a.end());
                ImGui::Render();
            }
        };
        for (ImVec2 size : {ImVec2{1920, 1080}, ImVec2{3840, 2160}, ImVec2{320, 180}})
            for (double dur : {600.0, 0.0}) {  // normal and unknown-duration video
                io.DisplaySize = size;
                UiState st{12, dur, 5, 4.5, false, true, true, false, true, 0.8f, "info", {"/v/a.mkv"}};
                st.scale = size.y > 1080 ? 2.5f : 1;  // 2160/1080 = 2; 2.5 is the clamp's top
                st.notice = {ICON_VOLUME, "Volume 80%", .4f, st.now, 1};
                frames(st);
                check(ui_animating(st.now));
                st.notice = {nullptr, "Am\xc3\xa9lie (2001).mkv", -1, st.now, 2};
                frames(st);
                // sweep the mouse along the capsule: hovers every button, the seek bar and the volume panel
                float sc = st.scale, y = size.y - 14 * sc - 24 * sc;
                for (float x = 0; x < size.x; x += 8 * sc) io.AddMousePosEvent(x, y), frames(st, 1, 0);
                io.AddMousePosEvent(size.x / 2, y - 60 * sc);  // over the volume panel's height, not on it
                st.notice = {};
                st.last_mouse = st.now - 10;  // idle: the capsule fades out over 0.3 s
                frames(st);
                check(ui_controls_hidden() && !ui_animating(st.now + 5));
            }
        // a click on the middle of the seek bar seeks there on release
        io.DisplaySize = {1920, 1080};
        UiState st{12, 600, 50, 50, false, true, true, false, false, 0.8f, "", {}};
        io.AddMousePosEvent(960, 1080 - 14 - 24);
        frames(st, 3, .1);  // fade in
        st.now += .3;
        acts.clear();
        io.AddMouseButtonEvent(0, true);
        frames(st, 2, 0);
        check(acts.empty());  // nothing while held
        io.AddMouseButtonEvent(0, false);
        frames(st, 2, 0);
        // mid-track: about the middle of the video (labels without the icon font push the bar left)
        check(acts.size() == 1 && acts[0].kind == UiAction::Seek && acts[0].value > 200 && acts[0].value < 560);
        // the volume panel stays open on the way up from the icon, through the capsule's edge, to its slider
        const float cy = 1080 - 14 - 24, top = cy - 24;
        float vol_cx = -1;
        for (float x = 960; x < 1920 && vol_cx < 0; x += 4) {  // find the icon: the panel opens over it
            io.AddMousePosEvent(x, cy);
            frames(st, 1, 0);
            if (ImGuiWindow* w = ImGui::FindWindowByName("volume"); w && w->Active) vol_cx = w->Pos.x + w->Size.x / 2;
        }
        check(vol_cx > 0);
        io.AddMousePosEvent(vol_cx, cy);
        frames(st, 1, 0);
        io.AddMousePosEvent(vol_cx, top + 2);  // above the icon, below the capsule's top edge
        frames(st, 1, 0);
        io.AddMousePosEvent(vol_cx, top - 8 - 18);  // the panel's middle, on its slider
        frames(st, 1, 0);
        acts.clear();
        io.AddMouseButtonEvent(0, true);
        frames(st, 1, 0);
        io.AddMouseButtonEvent(0, false);
        frames(st, 1, 0);
        check(!acts.empty() && acts[0].kind == UiAction::SetVolume);
        // the settings card: draws at both scales with 0, 1 and 3 monitors and with nothing available, never takes
        // the player's keys, closes on its X and on a click outside it, and shows the build in About
        {
            std::vector<ScreenInfo> mons = {{"PG27UCDM", 3840, 2160, 240, true}, {"LG ULTRAGEAR", 2560, 1440, 60, false},
                                            {"Very Long Monitor Name That Goes On And On", 1920, 1080, 60, false}};
            for (ImVec2 size : {ImVec2{1920, 1080}, ImVec2{3840, 2160}, ImVec2{640, 360}, ImVec2{320, 180}})
                for (size_t n : {0, 1, 3})
                    for (bool avail : {true, false}) {
                        const float scale = size.y > 1080 ? 2.5f : 1;
                        io.DisplaySize = size;
                        UiState cs{12, 600, 70, 70, false, avail, avail, false, false, 0.8f, "", {}};
                        cs.scale = scale, cs.settings_open = true, cs.build = "build 7 (abc1234)";
                        cs.screens.assign(mons.begin(), mons.begin() + n);
                        cs.playing_on = n ? (int)n - 1 : -1;
                        cs.rtx_available = cs.pxf_available = cs.nvdec_available = avail;
                        if (!avail) cs.vsr_unavailable = "no vsr.onnx";
                        io.AddMousePosEvent(5, 5);
                        frames(cs, 2, .1);
                        ImGuiWindow* w = ImGui::FindWindowByName("settings");
                        check(w && w->Active && w->DrawList->VtxBuffer.Size > 0 && !ui_has_keyboard());
                        check(w && w->Size.y <= io.DisplaySize.y * .8f + 1);  // scrolls rather than overflowing
                        // never over the capsule, which stays visible (and clickable) while the card is open
                        check(w && w->Pos.y >= 0 && w->Pos.y + w->Size.y <= size.y - (14 + 48) * scale + .5f);
                    }
            io.DisplaySize = {1920, 1080};
            UiState cs{12, 600, 80, 80, false, true, true, false, false, 0.8f, "", {}};
            cs.settings_open = true, cs.screens = mons, cs.build = "build 7 (abc1234)";
            frames(cs, 2, .1);
            ImGuiWindow* w = ImGui::FindWindowByName("settings");
            check(w != nullptr);
            auto click = [&](float x, float y) {
                acts.clear();
                io.AddMousePosEvent(x, y);
                frames(cs, 1, 0);
                io.AddMouseButtonEvent(0, true);
                frames(cs, 1, 0);
                io.AddMouseButtonEvent(0, false);
                frames(cs, 1, 0);
                return std::count_if(acts.begin(), acts.end(),
                                     [](const UiAction& a) { return a.kind == UiAction::ToggleSettings; });
            };
            if (w) {
                check(click(w->Pos.x + w->Size.x / 2, w->Pos.y + w->Size.y / 2) == 0);  // inside: stays open
                check(click(20, 20) == 1);                                              // outside: closes
                check(click(w->Pos.x + w->Size.x - 14 - 16, w->Pos.y + 12 + 16) == 1);  // the X
            }
            std::string about = about_text("build 7 (abc1234)", "C:/u/open-player.log");  // what the card's About section shows
            check(about.find("Open Player build 7 (abc1234)") != std::string::npos &&
                  about.find("Pixel Flow") != std::string::npos && about.find("Log: C:/u/open-player.log") != std::string::npos);
            io.AddMousePosEvent(5, 5);
        }
        // shortcut keys stay with the menu while it's open (Esc closes it instead of quitting); the menu draws
        ImGui::NewFrame();
        check(!ui_has_keyboard());
        ImGui::OpenPopup("menu");
        check(ui_has_keyboard());
        draw_ui({12, 600, 60, 60, true, true, false, true, false, 0, "", {"/v/a.mkv", "/v/b.mkv"}});
        ImGui::Render();
        ImGui::DestroyContext();
    }

    // icon strings encode their codepoints
    for (auto [s, cp] : {std::pair{ICON_PLAY, CP_PLAY}, {ICON_PAUSE, CP_PAUSE}, {ICON_VOLUME, CP_VOLUME},
                         {ICON_VOLUME_OFF, CP_VOLUME_OFF}, {ICON_FOLDER, CP_FOLDER}, {ICON_MAXIMIZE, CP_MAXIMIZE},
                         {ICON_MINIMIZE, CP_MINIMIZE}, {ICON_SPARKLES, CP_SPARKLES}, {ICON_FORWARD_5, CP_FORWARD_5},
                         {ICON_BACKWARD_5, CP_BACKWARD_5}, {ICON_INFO, CP_INFO}, {ICON_X, CP_X}, {ICON_CLOCK, CP_CLOCK},
                         {ICON_SETTINGS, CP_SETTINGS}, {ICON_DESKTOP, CP_DESKTOP}, {ICON_CPU, CP_CPU},
                         {ICON_ADJUSTMENTS, CP_ADJUSTMENTS}}) {
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
        for (ImWchar c : {CP_PLAY, CP_PAUSE, CP_VOLUME, CP_FOLDER, CP_MAXIMIZE, CP_SPARKLES, CP_SETTINGS, CP_DESKTOP,
                          CP_CPU, CP_ADJUSTMENTS, (ImWchar)'A', (ImWchar)0xE9})
            check(io.Fonts->Fonts[0]->IsGlyphInFont(c));
    } else printf("note: assets not found next to the binary, font checks skipped\n");
    ImGui::DestroyContext();

#undef check
    fails += settings_selftest();
    fails += log_selftest();
    fails += update_selftest();
    if (!fails) printf("ui checks passed\n");
    return fails ? 1 : 0;
}
