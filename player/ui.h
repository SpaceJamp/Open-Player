#pragma once
// The player's on-screen UI and the small pieces of state it keeps (recent files).
#include <string>
#include <vector>

// Recent files: newest first, no duplicates, at most 10.
std::vector<std::string> add_recent(std::vector<std::string> list, const std::string& path);
std::vector<std::string> existing_only(const std::vector<std::string>& list);  // drops files that no longer exist
std::vector<std::string> load_recent(const std::string& file);               // missing or unreadable: empty
void save_recent(const std::string& file, const std::vector<std::string>& list);  // failures are ignored
std::string recent_file();  // $XDG_CONFIG_HOME/open-player/recent.txt, default ~/.config/open-player/recent.txt

std::string clock_text(double t);
bool can_open(const std::string& path);  // an existing regular file: worth restarting the player on
bool ui_has_keyboard();  // a menu or text field has the keys, so the player's shortcuts must not act on them
// The player's own arguments with the video replaced, to restart it on another file. The video is the first
// argument that isn't a flag or the value of --model/--screen; none: the video is added at the end.
std::vector<std::string> restart_args(const std::vector<std::string>& argv, const std::string& video);  // 01:05 or 1:02:03

struct UiNotice {  // a brief message over the video (volume, VSR), shown for hold seconds then faded
    const char* icon = nullptr;
    std::string text;
    float level = -1;      // 0..1 bar; < 0: no bar
    double shown_at = -1e9;
    double hold = 1;       // seconds
};
// 0..1 opacity: fades in over `in` seconds after changed_at when visible, out over `out` seconds when hidden
float fade_alpha(double now, double changed_at, bool visible, double in, double out);
float notice_alpha(double now, const UiNotice& n);  // 0 before shown_at, 1 for hold seconds, then fades out
float notice_level(float volume);                   // volume 0..2 -> bar 0..1

struct UiState {
    double pos, dur, now, last_mouse;  // seconds; dur <= 0: unknown length
    bool paused, vsr_available, vsr_on, fullscreen, show_info;
    float volume;      // 0..2
    std::string info;  // info overlay text
    std::vector<std::string> recent;
    float scale = 1;
    UiNotice notice;
};
struct UiAction {
    enum Kind { TogglePause, Seek, SetVolume, ToggleVsr, ToggleFullscreen, ToggleInfo, Open, OpenPath, Quit } kind;
    double value = 0;  // Seek: seconds; SetVolume: 0..2
    std::string path;  // OpenPath
};
// The control bar shows while the mouse moved in the last 2 s, while paused, or while the mouse is over it.
bool controls_visible(double now, double last_mouse, bool paused, bool hovered);
// Draws the control capsule, the notice card, the info card (s.show_info) and the right-click menu; returns what
// the user did. Call between ImGui::NewFrame() and ImGui::Render().
std::vector<UiAction> draw_ui(const UiState& s);
bool ui_animating(double now);  // the capsule is fading or a notice is showing: keep redrawing
bool ui_controls_hidden();      // the capsule was fully faded out at the last draw_ui

int ui_selftest();  // --uitest: checks the UI logic without a window or GPU; 0 on pass
