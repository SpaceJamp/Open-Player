#pragma once
// open-player.log: everything the player writes to stderr (its own lines and FFmpeg's, SDL's, TensorRT's and
// NVIDIA's), with a timestamp per line, also saved to a file next to settings.txt, so errors survive the console
// window. The previous run's log is kept as open-player.old.log.
#include <cstddef>
#include <functional>
#include <string>

std::string log_file();  // open-player.log in settings_file()'s folder

// Starts copying stderr into path (rotating the old file first); the console still shows everything. Also notes a
// crash as the log's last line. false: no log (the player runs on without one).
bool start_log(const std::string& path);
void stop_log();  // flushes and stops; call before the process is replaced (restart) or exits

// --- the pieces start_log uses, separate so --uitest can check them ---
// Splits a byte stream into lines, puts stamp() before each line, and stops after cap bytes with one last note.
class LogWriter {
public:
    LogWriter(std::function<void(const std::string&)> out, size_t cap, std::function<std::string()> stamp);
    void feed(const char* p, size_t n);

private:
    std::function<void(const std::string&)> out_;
    std::function<std::string()> stamp_;
    size_t cap_, written_ = 0;
    bool line_start_ = true, cut_ = false;
};
std::string log_stamp();                    // "2026-11-21 14:03:22.123 " (local time)
bool rotate_log(const std::string& path);   // path -> <stem>.old.log, replacing the older one; true if nothing to do
int log_selftest();                         // part of --uitest; 0 on pass
