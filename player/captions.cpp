#include "captions.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

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

bool picture_ok(int w, int h) { return w > 0 && h > 0 && w <= 4096 && h <= 4096; }

std::vector<CaptionTrack> list_tracks(const AVFormatContext* fmt, const std::string& video_path) {
    std::vector<CaptionTrack> out;
    for (unsigned i = 0; fmt && i < fmt->nb_streams; i++) {
        const AVStream* st = fmt->streams[i];
        if (st->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE || !avcodec_find_decoder(st->codecpar->codec_id))
            continue;
        const AVDictionaryEntry *lang = av_dict_get(st->metadata, "language", nullptr, 0),
                                *title = av_dict_get(st->metadata, "title", nullptr, 0);
        if (!english(lang ? lang->value : nullptr)) continue;
        CaptionTrack t;
        t.stream = (int)i;
        t.sdh = (st->disposition & AV_DISPOSITION_HEARING_IMPAIRED) || (title && has(title->value, "sdh"));
        t.forced = st->disposition & AV_DISPOSITION_FORCED;
        t.name = track_name(title ? title->value : nullptr, t.sdh, t.forced);
        out.push_back(t);
    }
    for (auto& f : find_sidecars(video_path)) {
        CaptionTrack t;
        t.sidecar = f;
        t.name = "English (" + fs::u8path(f).filename().u8string() + ")";
        out.push_back(t);
    }
    return out;
}

// "00:01:02,345" -> seconds; -1 if it isn't a time
static double srt_time(const char* s) {
    int h, m, sec, ms;
    if (sscanf(s, "%d:%d:%d%*[,.]%d", &h, &m, &sec, &ms) != 4) return -1;
    return h * 3600.0 + m * 60 + sec + ms / 1000.0;
}

std::vector<Cue> read_sidecar(const std::string& srt_path) {
    std::ifstream f(fs::u8path(srt_path), std::ios::binary);
    std::string raw((std::istreambuf_iterator<char>(f)), {}), text = to_utf8(raw.substr(0, 64u << 20)), line;
    if (!text.compare(0, 3, "\xef\xbb\xbf")) text.erase(0, 3);  // a UTF-8 byte order mark
    std::vector<Cue> out;
    Cue cur;
    bool in_cue = false;
    auto done = [&] {
        if (in_cue && out.size() < MAX_CUES) cur.text = clean_text(cur.text), out.push_back(cur);
        cur = Cue(), in_cue = false;
    };
    size_t at = 0;
    while (at <= text.size() && out.size() < MAX_CUES) {
        size_t nl = text.find('\n', at);
        line = text.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        at = nl == std::string::npos ? text.size() + 1 : nl + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t arrow = line.find("-->");
        if (arrow != std::string::npos) {  // "00:00:01,000 --> 00:00:02,500": a new cue
            done();
            cur.start = srt_time(line.c_str()), cur.end = srt_time(line.c_str() + arrow + 3 + strspn(line.c_str() + arrow + 3, " "));
            in_cue = cur.start >= 0;
        } else if (line.empty()) done();
        else if (in_cue) cur.text += (cur.text.empty() ? "" : "\n") + line;
    }
    done();
    return out;
}

CueDecoder::~CueDecoder() { avcodec_free_context(&ctx_); }

bool CueDecoder::open(const AVStream* st) {
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec || !(ctx_ = avcodec_alloc_context3(codec))) return false;
    avcodec_parameters_to_context(ctx_, st->codecpar);
    ctx_->pkt_timebase = st->time_base;
    tb_ = av_q2d(st->time_base);
    if (avcodec_open2(ctx_, codec, nullptr) < 0) return avcodec_free_context(&ctx_), false;
    return true;
}

void CueDecoder::feed(const AVPacket* pkt, std::vector<Cue>& out) {
    if (!ctx_ || pkt->pts == AV_NOPTS_VALUE) return;
    AVSubtitle sub{};
    int got = 0;
    if (avcodec_decode_subtitle2(ctx_, &sub, &got, const_cast<AVPacket*>(pkt)) < 0 || !got) return;
    if (out.size() >= MAX_CUES) out.erase(out.begin(), out.begin() + MAX_CUES / 2);  // the oldest half
    const double at = pkt->pts * tb_;
    Cue base;
    base.start = at + sub.start_display_time / 1000.0;
    // no end (PGS shows until the next display set), else the packet's duration or the subtitle's own end
    base.end = sub.end_display_time && sub.end_display_time != UINT32_MAX ? at + sub.end_display_time / 1000.0
               : pkt->duration > 0 && ctx_->codec_id != AV_CODEC_ID_HDMV_PGS_SUBTITLE ? at + pkt->duration * tb_
                                                                                     : -1;
    base.canvas_w = ctx_->width > 0 ? ctx_->width : 1920, base.canvas_h = ctx_->height > 0 ? ctx_->height : 1080;
    if (!sub.num_rects) out.push_back(base);  // a picture track's "clear": ends the picture before it
    for (unsigned i = 0; i < sub.num_rects; i++) {
        const AVSubtitleRect* r = sub.rects[i];
        Cue c = base;
        if (r->type == SUBTITLE_ASS && r->ass) {  // "ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text"
            const char* t = r->ass;
            for (int commas = 0; *t && commas < 8; t++) commas += *t == ',';
            c.text = clean_text(t);
        } else if (r->type == SUBTITLE_TEXT && r->text) c.text = clean_text(r->text);
        else if (r->type == SUBTITLE_BITMAP && picture_ok(r->w, r->h) && r->data[0] && r->data[1]) {
            const uint32_t* pal = (const uint32_t*)r->data[1];  // ARGB, one per palette index
            c.x = r->x, c.y = r->y, c.w = r->w, c.h = r->h;
            c.rgba.resize((size_t)r->w * r->h * 4);
            for (int y = 0; y < r->h; y++)
                for (int x = 0; x < r->w; x++) {
                    uint32_t argb = pal[r->data[0][y * r->linesize[0] + x] % (r->nb_colors > 0 ? r->nb_colors : 256)];
                    uint8_t* px = &c.rgba[((size_t)y * r->w + x) * 4];
                    px[0] = argb >> 16, px[1] = argb >> 8, px[2] = (uint8_t)argb, px[3] = argb >> 24;
                }
        } else continue;
        if (!c.text.empty() || !c.rgba.empty()) out.push_back(std::move(c));
    }
    avsubtitle_free(&sub);
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

    // a .srt file: cues with their times, markup cleaned, a Windows-1252 one read right, and FFmpeg's own .srt
    // reader + SubRip decoder (the path embedded tracks take) giving the same cues
    fs::create_directories(d);
    const fs::path srt = d / "movie.en.srt";
    std::ofstream(srt, std::ios::binary) << "1\r\n00:00:01,000 --> 00:00:02,500\r\n<i>Hello</i> there\r\n\r\n"
                                            "2\r\n00:00:03,000 --> 00:00:04,000\r\nTwo\r\nlines\r\n\r\n"
                                            "3\r\n01:02:03,040 --> 01:02:04,000\r\ncaf\xe9\r\n";
    auto sc = read_sidecar(srt.u8string());
    check(sc.size() == 3);
    if (sc.size() == 3) {
        check(sc[0].start == 1 && sc[0].end == 2.5 && sc[0].text == "Hello there");
        check(sc[1].text == "Two\nlines" && std::abs(sc[2].start - 3723.04) < 1e-6 && sc[2].text == "caf\xc3\xa9");
    }
    {
        std::ofstream big_srt(d / "big.srt", std::ios::binary);
        for (int i = 0; i < 30000; i++) big_srt << i + 1 << "\n00:00:01,000 --> 00:00:02,000\nx\n\n";
    }
    check(read_sidecar((d / "big.srt").u8string()).size() == MAX_CUES);
    check(read_sidecar((d / "missing.srt").u8string()).empty());
    check(picture_ok(1920, 200) && !picture_ok(0, 10) && !picture_ok(5000, 10) && !picture_ok(10, 16000));

    std::ofstream(d / "utf8.srt", std::ios::binary) << "1\n00:00:01,000 --> 00:00:02,500\n<i>Hello</i>\n\n";
    AVFormatContext* fmt = nullptr;
    check(avformat_open_input(&fmt, (d / "utf8.srt").string().c_str(), nullptr, nullptr) == 0);
    if (fmt) {
        avformat_find_stream_info(fmt, nullptr);
        auto tracks = list_tracks(fmt, (d / "movie.mkv").u8string());
        // the stream itself (untagged), then the movie.en.srt beside "movie.mkv"
        check(tracks.size() == 2 && tracks[0].stream == 0 && tracks[0].name == "English");
        check(tracks.size() == 2 && tracks[1].stream == -1 && fs::u8path(tracks[1].sidecar) == srt);
        CueDecoder dec;
        check(dec.open(fmt->streams[0]));
        std::vector<Cue> got;
        AVPacket* pkt = av_packet_alloc();
        while (av_read_frame(fmt, pkt) >= 0) dec.feed(pkt, got), av_packet_unref(pkt);
        av_packet_free(&pkt);
        check(got.size() == 1 && got.size() && got[0].text == "Hello" && std::abs(got[0].start - 1) < 1e-6 &&
              std::abs(got[0].end - 2.5) < 1e-6);
        avformat_close_input(&fmt);
    }
    fs::remove_all(d, ec);
#undef check
    return fails ? 1 : 0;
}
