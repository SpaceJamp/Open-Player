#include "theme.h"

#include <cstdio>
#include <filesystem>

namespace {
bool g_loaded = false;
}

std::string assets_dir() {
    std::error_code ec;
    auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    return (exe.parent_path() / ".." / "assets").lexically_normal().string();
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
                                     CP_CLOCK,     CP_CLOCK,     0};
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
