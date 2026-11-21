#include "logfile.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "settings.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static constexpr size_t LOG_CAP = 5u << 20;  // one run's log stops at 5 MB

std::string log_file() {
    return (std::filesystem::path(settings_file()).parent_path() / "open-player.log").string();
}

LogWriter::LogWriter(std::function<void(const std::string&)> out, size_t cap, std::function<std::string()> stamp)
    : out_(std::move(out)), stamp_(std::move(stamp)), cap_(cap) {}

void LogWriter::feed(const char* p, size_t n) {
    std::string s;
    auto flush = [&] {  // also part-lines: output that never ends a line must hit the cap too
        if (s.empty() || cut_) return;
        if (written_ + s.size() > cap_)  // the note says why the log ends here
            s = std::string(line_start_ ? "" : "\n") + "[log cut at " + std::to_string(cap_ >> 20) + " MB]\n", cut_ = true;
        written_ += s.size();
        out_(s);
        s.clear();
    };
    for (size_t i = 0; i < n && !cut_; i++) {
        const unsigned char c = (unsigned char)p[i];
        if (c == '\r' && i + 1 < n && p[i + 1] == '\n') continue;  // CRLF -> LF
        if (line_start_) s += stamp_(), line_start_ = false;
        // control bytes (from a file name, say) can't fake a line or drive a terminal that shows the log
        s += (c < 0x20 && c != '\n' && c != '\t') || c == 0x7f ? '?' : (char)c;
        if (c == '\n') line_start_ = true, flush();
    }
    flush();
}

std::string log_stamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    int ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char b[40];
    snprintf(b, sizeof b, "%04d-%02d-%02d %02d:%02d:%02d.%03d ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, ms);
    return b;
}

bool rotate_log(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return true;
    auto old = std::filesystem::path(path).replace_extension(".old.log");
    // rename replaces .old.log; when another running player holds the log open it fails and nothing is touched
    std::filesystem::rename(path, old, ec);
    return !ec;
}

// stderr (fd 2) goes into a pipe; a thread copies it to the console (the original fd 2) and the file.
namespace {
std::mutex g_mu;
FILE* g_file = nullptr;
LogWriter* g_writer = nullptr;
int g_console = -1, g_pipe_w = -1;
std::thread g_thread;

void write_file(const std::string& s) {  // under g_mu
    fwrite(s.data(), 1, s.size(), g_file);
    fflush(g_file);  // a crash must not lose the last lines
}

#ifdef _WIN32
LONG WINAPI on_crash(EXCEPTION_POINTERS* e) {
    Sleep(200);  // let the reader thread write the last lines still in the pipe (the error before the crash)
    if (std::unique_lock l(g_mu, std::try_to_lock); l && g_writer) {
        char b[96];
        int n = snprintf(b, sizeof b, "crashed: exception 0x%08lX at %p\n", e->ExceptionRecord->ExceptionCode,
                         e->ExceptionRecord->ExceptionAddress);
        g_writer->feed(b, (size_t)n);
    }
    return EXCEPTION_CONTINUE_SEARCH;  // Windows still reports it as usual
}
#else
void on_crash(int sig) {  // only async-signal-safe calls: write the note straight to the file
    timespec wait{0, 200000000};
    nanosleep(&wait, nullptr);  // let the reader thread write the last lines still in the pipe first
    if (g_file) {
        char b[] = "crashed: signal   \n";
        b[16] = char('0' + sig / 10), b[17] = char('0' + sig % 10);
        (void)!write(fileno(g_file), b, sizeof b - 1);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}
#endif
}  // namespace

bool start_log(const std::string& path, bool append) {
    if (g_thread.joinable()) return true;
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    if (!append && !rotate_log(path)) return false;  // another player is writing it: leave its log alone
    if (!(g_file = fopen(path.c_str(), append ? "ab" : "wb"))) return false;
    g_writer = new LogWriter(write_file, LOG_CAP, log_stamp);
    int fds[2];
#ifdef _WIN32
    if (_get_osfhandle(2) < 0) {  // started without an error output: give fd 2 one, or the console dup fails
        int nul = _open("NUL", _O_WRONLY);
        if (nul >= 0) _dup2(nul, 2), _close(nul);
    }
    if (_pipe(fds, 1 << 16, _O_BINARY | _O_NOINHERIT) != 0) return fclose(g_file), g_file = nullptr, false;
    fflush(stderr);
    g_console = _dup(2);
    _dup2(fds[1], 2);
    SetStdHandle(STD_ERROR_HANDLE, (HANDLE)_get_osfhandle(2));  // libraries writing through the Win32 handle too
    SetUnhandledExceptionFilter(on_crash);
#else
    if (fcntl(2, F_GETFD) == -1) {  // started with 2>&-: otherwise pipe() would take fd 2 itself
        int nul = open("/dev/null", O_WRONLY);
        if (nul >= 0 && nul != 2) dup2(nul, 2), close(nul);
    }
    if (pipe(fds) != 0) return fclose(g_file), g_file = nullptr, false;
    fcntl(fds[0], F_SETFD, FD_CLOEXEC), fcntl(fds[1], F_SETFD, FD_CLOEXEC);  // not inherited by a restart
    fflush(stderr);
    g_console = dup(2);
    if (g_console >= 0) fcntl(g_console, F_SETFD, FD_CLOEXEC);
    dup2(fds[1], 2);
    for (int s : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT}) signal(s, on_crash);
#endif
    g_pipe_w = fds[1];
    setvbuf(stderr, nullptr, _IONBF, 0);
    int r = fds[0];
    g_thread = std::thread([r] {
        char buf[4096];
        for (;;) {
#ifdef _WIN32
            int n = _read(r, buf, sizeof buf);
            if (n <= 0) break;
            if (g_console >= 0) (void)_write(g_console, buf, (unsigned)n);
#else
            ssize_t n = read(r, buf, sizeof buf);
            if (n <= 0) break;
            if (g_console >= 0) (void)!write(g_console, buf, (size_t)n);
#endif
            std::lock_guard l(g_mu);
            g_writer->feed(buf, (size_t)n);
        }
#ifdef _WIN32
        _close(r);
#else
        close(r);
#endif
    });
    return true;
}

void stop_log() {
    if (!g_thread.joinable()) return;
    fflush(stderr);
    // put the console back on fd 2 and close every write end of the pipe, so the thread reads to the end
    // (no console copy: just close fd 2, which is the pipe's other write end)
#ifdef _WIN32
    if (g_console >= 0) _dup2(g_console, 2);
    else _close(2);
    SetStdHandle(STD_ERROR_HANDLE, (HANDLE)_get_osfhandle(2));
    _close(g_pipe_w);
#else
    if (g_console >= 0) dup2(g_console, 2);
    else close(2);
    close(g_pipe_w);
#endif
    g_thread.join();  // the reader still copies to g_console until here
#ifdef _WIN32
    if (g_console >= 0) _close(g_console);
#else
    if (g_console >= 0) close(g_console);
#endif
    g_console = -1;
    std::lock_guard l(g_mu);
    fclose(g_file), g_file = nullptr;
    delete g_writer, g_writer = nullptr;
}

int log_selftest() {
    int fails = 0;
#define check(cond) \
    if (!(cond)) printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond), fails++

    {  // a timestamp at the start of every line, also when a line arrives in pieces
        std::string got;
        LogWriter w([&](const std::string& s) { got += s; }, 1000, [] { return std::string("T "); });
        w.feed("VSR: Pixel", 10);
        w.feed(" Flow\nNVDEC unavail", 19);
        w.feed("able\n\n", 6);
        check(got == "T VSR: Pixel Flow\nT NVDEC unavailable\nT \n");
    }
    {  // stops at the cap with one note, then writes nothing more
        std::string got;
        LogWriter w([&](const std::string& s) { got += s; }, 20, [] { return std::string(""); });
        std::string line(15, 'x');
        line += '\n';
        for (int i = 0; i < 5; i++) w.feed(line.data(), line.size());
        check(got.size() < 120 && got.find("log cut at") != std::string::npos);
        size_t before = got.size();
        w.feed("more\n", 5);
        check(got.size() == before);
    }
    {  // output that never ends a line is capped too (a progress counter must not fill the disk)
        std::string got;
        LogWriter w([&](const std::string& s) { got += s; }, 20, [] { return std::string(""); });
        for (int i = 0; i < 50; i++) w.feed("0123456789", 10);
        check(got.size() < 120 && got.find("log cut at") != std::string::npos);
    }
    {  // control bytes from file names or metadata can't fake lines or reach a terminal; CRLF becomes LF
        std::string got;
        LogWriter w([&](const std::string& s) { got += s; }, 1000, [] { return std::string("T "); });
        const char evil[] = "a\x1b[31mb\rc\r\nd\x7f\te\n";
        w.feed(evil, sizeof evil - 1);
        check(got == "T a?[31mb?c\nT d?\te\n");
    }
    check(log_stamp().size() == 24 && log_stamp()[4] == '-' && log_stamp()[10] == ' ');
    {  // the previous run is kept as .old.log, replacing an older one
        auto dir = std::filesystem::temp_directory_path() / "open-player-log-test";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        std::string f = (dir / "open-player.log").string(), old = (dir / "open-player.old.log").string();
        check(rotate_log(f));  // nothing yet: fine
        std::ofstream(f) << "run 1\n";
        std::ofstream(old) << "run 0\n";
        check(rotate_log(f) && !std::filesystem::exists(f));
        std::ifstream in(old);
        std::string l;
        std::getline(in, l);
        check(l == "run 1");
        in.close();
        std::filesystem::remove_all(dir);
    }
    check(std::filesystem::path(log_file()).filename() == "open-player.log");
#undef check
    return fails ? 1 : 0;
}
