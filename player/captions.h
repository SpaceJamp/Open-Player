#pragma once
// Captions: the English subtitle tracks of a video (embedded, or a .srt beside it), the default choice, and the
// cues to show at a given time. No GPU here: the UI draws the cues, Media feeds the chosen stream's packets.
#include <cstdint>
#include <string>
#include <vector>

struct CaptionTrack {
    int stream = -1;      // embedded stream index; -1: a .srt beside the video (sidecar)
    std::string sidecar;  // its path
    std::string name;     // "English (SDH)", "British", ...
    bool sdh = false, forced = false;
};

struct Cue {
    double start = 0, end = -1;  // seconds; end < 0: until the next cue, at most 5 s
    std::string text;            // plain text, '\n' between lines; empty for a picture
    std::vector<uint8_t> rgba;   // a picture subtitle (w x h RGBA) at x, y on a canvas_w x canvas_h canvas
    int x = 0, y = 0, w = 0, h = 0, canvas_w = 0, canvas_h = 0;
};

bool english(const char* lang);  // "eng", "en", untagged
std::string track_name(const char* title, bool sdh, bool forced);  // "English (SDH)", "British (forced)"
int default_track(const std::vector<CaptionTrack>& t);  // SDH, else not forced-only, else the first; -1: none
int forced_track(const std::vector<CaptionTrack>& t);   // the forced one, shown with captions off; -1: none
std::string clean_text(const std::string& raw);         // SubRip/ASS markup -> plain text, at most 1000 bytes
std::vector<const Cue*> active_cues(const std::vector<Cue>& cues, double pos);  // by start time
std::vector<std::string> find_sidecars(const std::string& video_path);  // movie.srt, movie.en(g).srt
std::string to_utf8(const std::string& bytes);  // valid UTF-8 as it is, anything else read as Windows-1252

struct Rect {
    float x, y, w, h;
};
// Where a picture subtitle goes: its canvas centred on the shown picture, then kept inside the window and above
// bottom_limit (the control bar while it shows).
Rect place_picture(const Cue& c, Rect picture, Rect window, float bottom_limit);

inline constexpr size_t MAX_CUES = 20000;  // a hostile file can't grow the list without end
bool picture_ok(int w, int h);             // a picture subtitle small enough to keep (4096 x 4096 at most)

struct AVFormatContext;
struct AVStream;
struct AVPacket;
struct AVCodecContext;
// The English caption tracks: the file's English subtitle streams in order, then the .srt files beside it.
std::vector<CaptionTrack> list_tracks(const AVFormatContext* fmt, const std::string& video_path);
// A .srt file, whole: Windows-1252 files read too, markup cleaned, at most MAX_CUES cues.
std::vector<Cue> read_sidecar(const std::string& srt_path);

// One subtitle stream's decoder, on the decode thread.
class CueDecoder {
public:
    CueDecoder() = default;
    CueDecoder(const CueDecoder&) = delete;
    CueDecoder& operator=(const CueDecoder&) = delete;
    ~CueDecoder();
    bool open(const AVStream* st);  // false: no decoder for this codec
    // Appends this packet's cues (a picture track's "clear" is an empty cue that ends the one before).
    void feed(const AVPacket* pkt, std::vector<Cue>& out);

private:
    AVCodecContext* ctx_ = nullptr;
    double tb_ = 0;
};

int captions_selftest();  // part of --uitest; 0 on pass
