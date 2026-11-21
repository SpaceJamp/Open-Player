#pragma once
// The player's saved settings (settings.txt next to recent.txt) and the pure logic around them: reading and
// writing the file, finding the saved monitor again, and the arguments for reopening the video in place.
#include <string>
#include <vector>

enum class Upscaler { Rtx, PixelFlow, Off };

struct Settings {
    Upscaler upscaler = Upscaler::Rtx;  // NVIDIA RTX where available; Pixel Flow otherwise
    bool cpu_decode = false;
    std::string screen;  // screen_key() of the chosen monitor; empty: the primary
    bool start_fullscreen = false;
    bool remember_volume = true;
    float volume = 1;    // 0..2
    int hide_after = 2;  // seconds the controls stay after the mouse stops: 1, 2, 3 or 5
    bool info_at_start = false;
    std::string trt_folder;  // the user's own TensorRT 11.x folder; empty: the add-on or next to the program
};

// One "key=value" per line; unknown keys, bad values, blank lines and \r are ignored (defaults kept).
Settings settings_parse(const std::string& text);
std::string settings_format(const Settings& s);
Settings load_settings(const std::string& file);                // missing or unreadable: defaults
bool save_settings(const std::string& file, const Settings& s);  // atomic (temp file + rename); false on failure
std::string settings_file();                                    // settings.txt in recent_file()'s folder

struct ScreenInfo {
    std::string name;
    int w = 0, h = 0;  // physical pixels
    float hz = 0;
    bool primary = false;
};
// "PG27UCDM 3840x2160"; a second identical monitor (same name and size) is "... #2", and so on
std::string screen_key(const std::vector<ScreenInfo>& screens, int i);
// Index of the monitor saved as key: that exact one, else the first with its name and size, else the first with
// its name, else the primary, else 0.
int match_screen(const std::vector<ScreenInfo>& screens, const std::string& key);

// restart_args() for the same video, minus the flags a settings change overrides (--screen N, --no-rtx,
// --no-vsr, --cpu-decode) and any earlier --start/--paused, plus --start <start> and --paused when paused.
std::vector<std::string> reopen_args(const std::vector<std::string>& argv, const std::string& video, double start,
                                     bool paused);
// The flags that keep this run's state across a reopen without saving it: --fullscreen, then --no-vsr when
// upscaling is off or --no-rtx when Pixel Flow runs, then --screen N when screen > 0.
std::vector<std::string> live_flags(bool fullscreen, bool vsr_off, bool pixel_flow, int screen);
// Where --start T lands: within [0, max(0, dur - 1)] when the length is known (dur > 0), else max(0, T).
double start_position(double start, double dur);

int settings_selftest();  // part of --uitest; 0 on pass
