// Fonts, colours and icon codepoints for the player UI.
#pragma once
#include <string>

#include "imgui.h"

std::string assets_dir();                          // <dir of /proc/self/exe>/../assets
bool load_theme(const std::string& font_dir);      // after CreateContext, before the first frame
bool theme_loaded();

// Tabler icons (codepoints from tabler-icons-webfont 3.31.0), each with its UTF-8 string.
inline constexpr ImWchar CP_PLAY = 0xed46;
inline constexpr const char* ICON_PLAY = "\xee\xb5\x86";
inline constexpr ImWchar CP_PAUSE = 0xed45;
inline constexpr const char* ICON_PAUSE = "\xee\xb5\x85";
inline constexpr ImWchar CP_VOLUME = 0xeb51;
inline constexpr const char* ICON_VOLUME = "\xee\xad\x91";
inline constexpr ImWchar CP_VOLUME_OFF = 0xf1c3;
inline constexpr const char* ICON_VOLUME_OFF = "\xef\x87\x83";
inline constexpr ImWchar CP_FOLDER = 0xeaad;
inline constexpr const char* ICON_FOLDER = "\xee\xaa\xad";
inline constexpr ImWchar CP_MAXIMIZE = 0xeaea;
inline constexpr const char* ICON_MAXIMIZE = "\xee\xab\xaa";
inline constexpr ImWchar CP_MINIMIZE = 0xeaf1;
inline constexpr const char* ICON_MINIMIZE = "\xee\xab\xb1";
inline constexpr ImWchar CP_SPARKLES = 0xf6d7;
inline constexpr const char* ICON_SPARKLES = "\xef\x9b\x97";
inline constexpr ImWchar CP_FORWARD_5 = 0xfac7;
inline constexpr const char* ICON_FORWARD_5 = "\xef\xab\x87";
inline constexpr ImWchar CP_BACKWARD_5 = 0xfabf;
inline constexpr const char* ICON_BACKWARD_5 = "\xef\xaa\xbf";
inline constexpr ImWchar CP_INFO = 0xeac5;
inline constexpr const char* ICON_INFO = "\xee\xab\x85";
inline constexpr ImWchar CP_X = 0xeb55;
inline constexpr const char* ICON_X = "\xee\xad\x95";
inline constexpr ImWchar CP_CLOCK = 0xea70;
inline constexpr const char* ICON_CLOCK = "\xee\xa9\xb0";

inline constexpr ImU32 COL_CAPSULE = IM_COL32(20, 22, 30, 184);   // rgba(20,22,30,0.72)
inline constexpr ImU32 COL_TRACK = IM_COL32(255, 255, 255, 64);   // rgba(255,255,255,0.25)
inline constexpr ImU32 COL_PLAYED = IM_COL32(0x3f, 0xd0, 0xe0, 255);
inline constexpr ImU32 COL_VSR_ON = IM_COL32(0x5b, 0x4f, 0xd6, 255);
inline constexpr ImU32 COL_VSR_OFF = IM_COL32(120, 124, 136, 255);
inline constexpr ImU32 COL_TEXT = IM_COL32(240, 242, 248, 255);
