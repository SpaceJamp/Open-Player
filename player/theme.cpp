#include "theme.h"

#include <cstdio>
#include <filesystem>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
bool g_loaded = false;
}

std::string exe_path() {
#ifdef _WIN32
    static wchar_t buf[32768];  // the longest path Windows allows
    DWORD n = GetModuleFileNameW(nullptr, buf, 32768);
    return std::filesystem::path(std::wstring(buf, n)).string();  // UTF-8: the manifest sets the code page
#else
    std::error_code ec;
    return std::filesystem::read_symlink("/proc/self/exe", ec).string();
#endif
}

std::string assets_dir() { return assets_dir_from(std::filesystem::path(exe_path()).parent_path()); }

std::string assets_dir_from(const std::filesystem::path& dir) {
    // the installer puts assets\ beside the program; build/open-player -> assets/, and a multi-config Windows
    // build puts the exe one level deeper (build\Release)
    for (const char* up : {".", "..", "../.."}) {
        std::error_code ec;
        auto a = (dir / up / "assets").lexically_normal();
        if (std::filesystem::is_directory(a, ec)) return a.string();
    }
    return (dir / ".." / "assets").lexically_normal().string();
}

bool theme_loaded() { return g_loaded; }

bool load_theme(const std::string& font_dir) {
    g_loaded = false;
    const std::string inter = font_dir + "/Inter-Medium.ttf", icons = font_dir + "/tabler-icons.ttf";
    if (!std::filesystem::exists(inter) || !std::filesystem::exists(icons)) {
        fprintf(stderr, "ui: fonts not found in %s, using the default font\n", font_dir.c_str());
        return false;
    }
    ImGuiIO& io = ImGui::GetIO();
    ImFont* base = io.Fonts->AddFontFromFileTTF(inter.c_str(), 16.0f);
    if (!base) {
        fprintf(stderr, "ui: cannot read %s, using the default font\n", inter.c_str());
        return false;
    }
    static const ImWchar ranges[] = {CP_PLAY,      CP_PLAY,      CP_PAUSE,     CP_PAUSE,     CP_VOLUME,     CP_VOLUME,
                                     CP_VOLUME_OFF, CP_VOLUME_OFF, CP_FOLDER,   CP_FOLDER,    CP_MAXIMIZE,   CP_MAXIMIZE,
                                     CP_MINIMIZE,  CP_MINIMIZE,  CP_SPARKLES,  CP_SPARKLES,  CP_FORWARD_5,  CP_FORWARD_5,
                                     CP_BACKWARD_5, CP_BACKWARD_5, CP_INFO,     CP_INFO,      CP_X,          CP_X,
                                     CP_CLOCK,     CP_CLOCK,     CP_SETTINGS,  CP_SETTINGS,  CP_DESKTOP,    CP_DESKTOP,
                                     CP_CPU,       CP_CPU,       CP_ADJUSTMENTS, CP_ADJUSTMENTS, CP_BADGE_CC, CP_BADGE_CC, 0};
    ImFontConfig cfg;
    cfg.MergeMode = true;
    cfg.GlyphMinAdvanceX = 16.0f;
    cfg.GlyphRanges = ranges;
    if (!io.Fonts->AddFontFromFileTTF(icons.c_str(), 16.0f, &cfg)) {
        fprintf(stderr, "ui: cannot read %s, using the default font\n", icons.c_str());
        io.Fonts->Clear();  // drop the half-built font so ImGui falls back to its default
        return false;
    }

    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = s.PopupRounding = s.FrameRounding = 10.0f;
    s.Colors[ImGuiCol_Text] = ImGui::ColorConvertU32ToFloat4(COL_TEXT);
    s.Colors[ImGuiCol_PopupBg] = ImGui::ColorConvertU32ToFloat4(COL_CAPSULE);
    s.Colors[ImGuiCol_WindowBg] = ImGui::ColorConvertU32ToFloat4(COL_CAPSULE);
    s.Colors[ImGuiCol_HeaderHovered] = ImGui::ColorConvertU32ToFloat4(IM_COL32(255, 255, 255, 36));
    s.Colors[ImGuiCol_Border] = ImVec4(1, 1, 1, 0.10f);
    g_loaded = true;
    return true;
}
