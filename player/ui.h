#pragma once
// The player's on-screen UI and the small pieces of state it keeps (recent files).
#include <string>
#include <vector>

#include "captions.h"
#include "settings.h"

// Recent files: newest first, no duplicates, at most 10.
std::vector<std::string> add_recent(std::vector<std::string> list, const std::string& path);
std::vector<std::string> existing_only(const std::vector<std::string>& list);  // drops files that no longer exist
std::vector<std::string> load_recent(const std::string& file);               // missing or unreadable: empty
void save_recent(const std::string& file, const std::vector<std::string>& list);  // failures are ignored
std::string recent_file();  // $XDG_CONFIG_HOME/open-player/recent.txt, default ~/.config/open-player/recent.txt

std::string clock_text(double t);
// The screen's height / 1080, clamped to 1..2.5, so a window gets the same size controls as fullscreen; less (never
// below 1) when the window can't hold 640x360 at that size. All in window coordinates; screen_h 0: unknown.
float ui_scale(float display_w, float display_h, float screen_h);
// Notice for a key seek of delta seconds that lands on target; the time is clamped to 0 and, when dur > 0, to dur.
std::string seek_notice_text(double delta, double target, double dur);
bool can_open(const std::string& path);  // an existing regular file: worth restarting the player on
bool ui_has_keyboard();  // a menu or text field has the keys, so the player's shortcuts must not act on them
// The player's own arguments with the video replaced, to restart it on another file. The video is the first
// argument that isn't a flag or the value of --model/--screen/--collect; none: the video is added at the end.
// --start T and --paused are dropped: they belong to the video being replaced.
// One command-line argument quoted the way the Windows C runtime splits it again (restart on Windows)
std::string quote_arg(const std::string& a);
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
    int vsr_factor = 2;
    bool vsr_rtx = false;  // NVIDIA RTX VSR upscales (else Pixel Flow)
    // the settings card
    bool settings_open = false;
    Settings settings;  // as this run plays (upscaler Off whenever upscaling is off, whichever way)
    std::vector<ScreenInfo> screens;
    int screen_index = 0;  // the chosen monitor
    int playing_on = -1;   // the monitor the window is on now; -1: unknown
    bool rtx_available = false, pxf_available = false, nvdec_available = true;
    std::string pxf_why;          // why Pixel Flow can't run (tooltip)
    std::string vsr_unavailable;  // why neither upscaler can run here, shown instead of the choice
    std::string build;            // "build 214 (4a30cf8)"
    std::string log_path;         // open-player.log, shown in About
    std::vector<std::string> versions;  // TensorRT, RTX VSR and driver lines, shown in About
    // the TensorRT add-on (Pixel Flow needs TensorRT): its status line, and the download's progress (< 0: none)
    std::string trt_status;
    bool trt_addon = false;  // loaded from the add-on (so Remove applies)
    bool trt_loaded = false;  // the full TensorRT is loaded (add-on, your folder or next to the player)
    bool own_folder = false;  // Pixel Flow's engine comes from a folder you chose (so Forget folder applies)
    float addon_progress = -1;
    bool rtx_addon = false;     // the TensorRT for RTX add-on is installed (so Remove applies)
    float rtx_progress = -1;    // its download's progress (< 0: none)
    // updates (installed copies): the status line, a newer build to install (0: none), the download's progress
    std::string update_status;
    int update_build = 0;
    float update_progress = -1;
    // the Training tab: train.py's training.json (training_known false: no file yet), and the time now
    TrainingStatus training;
    bool training_known = false;
    long long now_unix = 0;
    // training from the player (the videos folder is settings.train_videos)
    int train_video_count = -1;           // videos found in that folder; -1: not counted yet
    int training_addon = 0;               // training_addon_installed(): 0 none, 1 current, 2 outdated
    float training_addon_progress = -1;   // the add-on's download (< 0: none)
    std::string training_error;           // the last thing that went wrong here (download, start, export)
    std::string training_log;             // train.log, named when a run failed
    bool training_own = false;            // training.json is the player's own run (it may control it)
    bool has_checkpoint = false, has_best = false, exporting = false, gpu_ok = true;
    bool training_alive = false;    // the player's run is going (its process, from run.txt)
    bool training_current = false;  // training.json is that run's, not one before it
    // captions: all known cues (owned by the caller), the playback time to show them at, the source picture's size
    // (to place picture subtitles), and the English tracks for the CC button and the menu
    const std::vector<Cue>* cues = nullptr;
    double cue_pos = 0;
    int video_w = 0, video_h = 0;
    std::vector<std::string> caption_names;
    int caption_track = -1;  // the chosen one; -1: off
    bool caption_forced = false;  // captions off, but a forced English track is shown
};
std::string vsr_badge_text(bool available, bool on, bool rtx, int factor);  // "PXF 2x", "RTX 2x", "VSR off"; "" without VSR
std::string about_text(const std::string& build, const std::string& log, const std::vector<std::string>& versions);  // About
struct UiAction {
    enum Kind {
        TogglePause, Seek, SetVolume, ToggleVsr, ToggleFullscreen, ToggleInfo, Open, OpenPath, Quit,
        // the settings card; value: an Upscaler, a screen index, seconds, or 0/1
        ToggleSettings, SetUpscaler, SetDecoder, SetScreen, SetStartFullscreen, SetRememberVolume, SetHideAfter,
        SetInfoAtStart, DownloadTensorRT, CancelTensorRT, RemoveTensorRT, ChooseTensorRTFolder,
        DownloadTrtRtx, CancelTrtRtx, RemoveTrtRtx, ForgetTensorRTFolder, SetCheckUpdates,
        SetSettingsTab, SetRememberPosition,
        ToggleCaptions, SetCaptionTrack,  // SetCaptionTrack value: a track index, -1 off
        // the Training tab
        ChooseTrainVideos, StartTraining, StopTraining, UseModel, StartOver, CancelTrainingAddon, RemoveTrainingAddon,
        InstallUpdate
    } kind;
    double value = 0;  // Seek: seconds; SetVolume: 0..2
    std::string path;  // OpenPath
};
// The Training tab's buttons and its extra lines of text, for the state in s (drawn by the tab, checked by tests)
struct TrainButton {
    std::string label;
    UiAction::Kind kind;
    bool enabled = true;
};
std::vector<TrainButton> training_buttons(const UiState& s);
std::vector<TrainButton> engine_buttons(const UiState& s);  // Settings > Upscaling: Pixel Flow's engine
std::vector<std::string> training_notes(const UiState& s);
// The control bar shows while the mouse moved in the last hide_after s, while paused, or while the mouse is over it.
bool controls_visible(double now, double last_mouse, bool paused, bool hovered, double hide_after = 2);
// Draws the control capsule, the notice card, the info card (s.show_info), the settings card (s.settings_open) and
// the right-click menu; returns what the user did. Call between ImGui::NewFrame() and ImGui::Render().
std::vector<UiAction> draw_ui(const UiState& s);
bool ui_animating(double now);  // the capsule is fading or a notice is showing: keep redrawing
bool ui_controls_hidden();      // the capsule was fully faded out at the last draw_ui

Rect last_caption_rect();  // where the latest frame drew captions (w 0: none); for tests
int ui_selftest();  // --uitest: checks the UI logic without a window or GPU; 0 on pass
