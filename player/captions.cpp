#include "captions.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static std::string lower(std::string s) {
    for (auto& c : s) c = (char)tolower((unsigned char)c);
    return s;
}
static bool has(const std::string& s, const char* word) { return lower(s).find(word) != std::string::npos; }

bool english(const char* lang) {
    if (!lang || !*lang) return true;  // untagged: most likely the film's own language
    const std::string l = lower(lang);
    return l == "eng" || l == "en";
}

std::string track_name(const char* title, bool sdh, bool forced) {
    std::string t = title ? title : "", rest;
    // a title that is only "SDH" or "Forced" names the kind, not the track
    for (char c : lower(t)) rest += isalpha((unsigned char)c) ? c : ' ';
    for (const char* w : {"sdh", "forced"})
        for (size_t i; (i = rest.find(w)) != std::string::npos;) rest.replace(i, strlen(w), "");
    if (rest.find_first_not_of(' ') == std::string::npos) t.clear();
    std::string name = t.empty() ? "English" : t;
    if (sdh && !has(name, "sdh")) name += " (SDH)";
    if (forced && !has(name, "forced")) name += " (forced)";
    return name;
}

int default_track(const std::vector<CaptionTrack>& t) {
    for (size_t i = 0; i < t.size(); i++)
        if (t[i].sdh && !t[i].forced) return (int)i;  // closed captions describe sounds too
    for (size_t i = 0; i < t.size(); i++)
        if (!t[i].forced) return (int)i;
    return t.empty() ? -1 : 0;
}

int forced_track(const std::vector<CaptionTrack>& t) {
    for (size_t i = 0; i < t.size(); i++)
        if (t[i].forced) return (int)i;
    return -1;
}

std::string clean_text(const std::string& raw) {
    std::string out;
    for (size_t i = 0; i < raw.size() && out.size() < 4000; i++) {
        char c = raw[i];
        if (c == '{') {  // ASS override block {\an8}
            size_t e = raw.find('}', i);
            if (e != std::string::npos) { i = e; continue; }
        }
        if (c == '<') {  // <i>, </font>, <font color=...>
            size_t e = raw.find('>', i);
            if (e != std::string::npos) { i = e; continue; }
        }
        if (c == '\\' && i + 1 < raw.size() && (raw[i + 1] == 'N' || raw[i + 1] == 'n')) { out += '\n', i++; continue; }
        if (c == '\\' && i + 1 < raw.size() && raw[i + 1] == 'h') { out += ' ', i++; continue; }  // ASS hard space
        if (c == '&') {
            static const std::pair<const char*, const char*> ents[] = {
                {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&#39;", "'"}, {"&apos;", "'"},
                {"&nbsp;", " "}};
            bool hit = false;
            for (auto& [e, v] : ents)
                if (!raw.compare(i, strlen(e), e)) { out += v, i += strlen(e) - 1, hit = true; break; }
            if (hit) continue;
        }
        if (c == '\r') continue;
        out += c;
    }
    // trim each line, drop empty ones
    std::string clean, line;
    auto flush = [&] {
        size_t a = line.find_first_not_of(" \t"), b = line.find_last_not_of(" \t");
        if (a != std::string::npos) clean += (clean.empty() ? "" : "\n") + line.substr(a, b - a + 1);
        line.clear();
    };
    for (char c : out) c == '\n' ? flush() : (void)(line += c);
    flush();
    if (clean.size() > 1000) {  // a hostile file's endless line: cut between characters
        clean.resize(1000);
        while (!clean.empty() && ((unsigned char)clean.back() & 0xC0) == 0x80) clean.pop_back();
        if (!clean.empty() && ((unsigned char)clean.back() & 0x80)) clean.pop_back();
    }
    return clean;
}

std::vector<const Cue*> active_cues(const std::vector<Cue>& cues, double pos) {
    std::vector<const Cue*> by_start;
    for (auto& c : cues) by_start.push_back(&c);
    std::stable_sort(by_start.begin(), by_start.end(), [](const Cue* a, const Cue* b) { return a->start < b->start; });
    std::vector<const Cue*> on;
    for (size_t i = 0; i < by_start.size(); i++) {
        const Cue* c = by_start[i];
        double end = c->end;
        if (end < 0) {  // no end given: until the next cue, at most 5 s
            end = c->start + 5;
            for (size_t j = i + 1; j < by_start.size(); j++)
                if (by_start[j]->start > c->start) { end = std::min(end, by_start[j]->start); break; }
        }
        if (c->start <= pos && pos < end) on.push_back(c);
    }
    return on;
}

std::vector<std::string> find_sidecars(const std::string& video_path) {
    const fs::path v = fs::u8path(video_path);
    const std::string stem = lower(v.stem().u8string());
    std::vector<std::string> out;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(v.parent_path(), ec)) {
        const std::string n = lower(e.path().filename().u8string());
        if (n == stem + ".srt" || n == stem + ".en.srt" || n == stem + ".eng.srt") out.push_back(e.path().u8string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

static bool valid_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        int n = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : -1;
        if (n < 0 || i + n >= s.size() + (n ? 0 : 1)) return false;
        for (int k = 1; k <= n; k++)
            if (i + k >= s.size() || ((unsigned char)s[i + k] >> 6) != 2) return false;
        i += n + 1;
    }
    return true;
}

std::string to_utf8(const std::string& bytes) {
    if (valid_utf8(bytes)) return bytes;
    // Windows-1252: 0x80-0x9F are typographic characters, the rest is Latin-1
    static const unsigned short hi[32] = {0x20AC, 0xFFFD, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
                                          0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0xFFFD, 0x017D, 0xFFFD,
                                          0xFFFD, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
                                          0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0xFFFD, 0x017E, 0x0178};
    std::string out;
    for (unsigned char c : bytes) {
        unsigned cp = c >= 0x80 && c < 0xA0 ? hi[c - 0x80] : c;
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) out += (char)(0xC0 | cp >> 6), out += (char)(0x80 | (cp & 0x3F));
        else out += (char)(0xE0 | cp >> 12), out += (char)(0x80 | (cp >> 6 & 0x3F)), out += (char)(0x80 | (cp & 0x3F));
    }
    return out;
}

Rect place_picture(const Cue& c, Rect picture, Rect window, float bottom_limit) {
    const float cw = c.canvas_w > 0 ? (float)c.canvas_w : 1920, ch = c.canvas_h > 0 ? (float)c.canvas_h : 1080;
    const float s = picture.w / cw;  // the canvas is as wide as the picture, centred on it (cropped Blu-rays)
    const float left = picture.x + picture.w / 2 - cw * s / 2, top = picture.y + picture.h / 2 - ch * s / 2;
    Rect r{left + c.x * s, top + c.y * s, c.w * s, c.h * s};
    r.y = std::min(r.y, bottom_limit - r.h);  // above the control bar, never under the window's edge
    r.y = std::max(r.y, window.y);
    r.x = std::clamp(r.x, window.x, std::max(window.x, window.x + window.w - r.w));
    return r;
}

int captions_selftest() {
    int fails = 0;
#define check(cond) \
    if (!(cond)) printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond), fails++

    check(english("eng") && english("en") && english("ENG") && english(nullptr) && english("") && !english("ger"));
    check(track_name("SDH", true, false) == "English (SDH)");
    check(track_name("British", false, false) == "British");
    check(track_name(nullptr, false, true) == "English (forced)");
    check(track_name("English SDH", true, false) == "English SDH");  // the title says it already
    check(track_name("English (SDH)", true, false) == "English (SDH)");

    CaptionTrack plain{2, "", "English"}, forced{3, "", "English (forced)", false, true}, sdh{4, "", "English (SDH)", true};
    check(default_track({plain, forced, sdh}) == 2 && default_track({forced, plain}) == 1);
    check(default_track({forced}) == 0 && default_track({}) == -1);
    check(forced_track({plain, forced}) == 1 && forced_track({plain}) == -1);

    check(clean_text("<i>Hi</i> &amp; {\\an8}bye\\Nnow") == "Hi & bye\nnow");
    check(clean_text("<font color=\"#ff0\">a</font>\\nb\r\n c ") == "a\nb\nc");
    check(clean_text("&lt;3 &quot;x&quot; &#39;y&#39;") == "<3 \"x\" 'y'");
    std::string big;
    for (int i = 0; i < 50000; i++) big += "\xc3\xa9";  // 100 000 bytes of "é"
    const std::string cut = clean_text(big);
    check(cut.size() <= 1000 && cut.size() % 2 == 0);  // cut between characters

    std::vector<Cue> cues = {{0, 2, "a"}, {1, 3, "b"}, {5, -1, "c"}, {20, -1, "d"}};
    auto texts = [&](double pos) {
        std::string s;
        for (const Cue* c : active_cues(cues, pos)) s += c->text;
        return s;
    };
    check(texts(1.5) == "ab" && texts(2.5) == "b" && texts(9.9) == "c" && texts(10.1).empty() && texts(21) == "d");
    check(texts(4) == "");

    fs::path d = fs::temp_directory_path() / "op-captions-test";
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    for (const char* f : {"movie.mkv", "movie.srt", "movie.en.srt", "movie.ger.srt", "other.srt", "movie.eng.SRT"})
        std::ofstream(d / f) << "x";
    auto side = find_sidecars((d / "movie.mkv").u8string());
    std::vector<std::string> names;
    for (auto& s : side) names.push_back(fs::u8path(s).filename().u8string());
    check(names == (std::vector<std::string>{"movie.en.srt", "movie.eng.SRT", "movie.srt"}));
    fs::remove_all(d, ec);

    check(to_utf8("caf\xe9") == "caf\xc3\xa9" && to_utf8("caf\xc3\xa9") == "caf\xc3\xa9" && to_utf8("") == "");
    check(to_utf8("\x93quoted\x94") == "\xe2\x80\x9cquoted\xe2\x80\x9d");  // Windows-1252 curly quotes

    Cue pic;
    pic.x = 660, pic.y = 980, pic.w = 600, pic.h = 60, pic.canvas_w = 1920, pic.canvas_h = 1080;
    auto same = [](Rect a, Rect b) {
        return std::abs(a.x - b.x) < .5f && std::abs(a.y - b.y) < .5f && std::abs(a.w - b.w) < .5f &&
               std::abs(a.h - b.h) < .5f;
    };
    check(same(place_picture(pic, {0, 140, 1920, 800}, {0, 0, 1920, 1080}, 1080), {660, 980, 600, 60}));
    check(same(place_picture(pic, {0, 140, 1920, 800}, {0, 0, 1920, 1080}, 1000), {660, 940, 600, 60}));
    check(same(place_picture(pic, {0, 70, 960, 400}, {0, 0, 960, 540}, 540), {330, 490, 300, 30}));
    pic.y = 1070;
    Rect r = place_picture(pic, {0, 140, 1920, 800}, {0, 0, 1920, 1080}, 1050);
    check(std::abs(r.y + r.h - 1050) < .5f);
#undef check
    return fails ? 1 : 0;
}
