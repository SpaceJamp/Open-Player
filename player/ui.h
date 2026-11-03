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

int ui_selftest();  // --uitest: checks the UI logic without a window or GPU; 0 on pass
