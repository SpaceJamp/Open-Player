#include "training_run.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "training_addon.h"
#include "trt_loader.h"
#include "ui.h"

#ifdef _WIN32
#include <windows.h>
// after windows.h
#include <shellapi.h>
#endif

namespace fs = std::filesystem;

#ifdef _WIN32
static std::wstring wide(const std::string& u8) {
    int n = MultiByteToWideChar(CP_UTF8, 0, u8.data(), (int)u8.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, u8.data(), (int)u8.size(), w.data(), n);
    return w;
}
#endif

std::wstring train_command(const std::string& python, const std::string& scripts, const std::string& videos,
                           const std::string& work, StartMode m) {
    const std::string ckpt = (fs::u8path(work) / "vsr.pt").u8string();
    std::string cmd = quote_arg(python) + " -u " + quote_arg((fs::u8path(scripts) / "train.py").u8string()) + " " +
                      quote_arg(videos) + " --out " + quote_arg(ckpt);
    if (m == StartMode::Resume) cmd += " --resume " + quote_arg(ckpt) + " --more-if-done 100000";
#ifdef _WIN32
    return wide(cmd);
#else
    return std::wstring(cmd.begin(), cmd.end());
#endif
}

#ifdef _WIN32
// Runs a command on the add-on's Python: no window, FFmpeg first on PATH, output appended to train.log. It gets
// only its own log and a NUL input, none of the player's handles: an inherited pipe or log would keep the player
// waiting at exit, or hold its log file open, for as long as training runs.
static HANDLE run_python(const std::wstring& cmd, std::string& error, DWORD* pid = nullptr) {
    fs::path work = fs::u8path(training_work_dir());
    std::error_code ec;
    fs::create_directories(work, ec);
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE log = CreateFileW((work / "train.log").wstring().c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE) return error = "can't write " + (work / "train.log").u8string(), nullptr;
    // this process's environment, with the add-on's FFmpeg first on PATH (train.py runs ffmpeg and ffprobe)
    std::wstring env, ff = fs::u8path(training_ffmpeg_dir()).wstring();
    bool had_path = false;
    LPWCH all = GetEnvironmentStringsW();
    for (LPWCH e = all; *e; e += wcslen(e) + 1) {
        std::wstring v = e;
        if (_wcsnicmp(v.c_str(), L"Path=", 5) == 0) v = L"Path=" + ff + L";" + v.substr(5), had_path = true;
        env += v;
        env += L'\0';
    }
    FreeEnvironmentStringsW(all);
    if (!had_path) env += L"Path=" + ff, env += L'\0';
    env += L"PYTHONDONTWRITEBYTECODE=1";  // no __pycache__ in the program folder (the uninstaller wouldn't know it)
    env += L'\0';
    env += L'\0';
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    HANDLE inherit[] = {log, nul};
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    std::vector<char> attrs(size);
    auto list = (LPPROC_THREAD_ATTRIBUTE_LIST)attrs.data();
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof si;
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = nul, si.StartupInfo.hStdOutput = log, si.StartupInfo.hStdError = log;
    BOOL ok = nul != INVALID_HANDLE_VALUE && InitializeProcThreadAttributeList(list, 1, 0, &size) &&
              UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, sizeof inherit, nullptr,
                                        nullptr);
    if (ok) si.lpAttributeList = list;
    PROCESS_INFORMATION pi{};
    std::wstring line = cmd, exe = fs::u8path(training_python()).wstring();
    ok = ok && CreateProcessW(exe.c_str(), line.data(), nullptr, nullptr, TRUE,
                              CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT, env.data(),
                              work.wstring().c_str(), &si.StartupInfo, &pi);
    if (si.lpAttributeList) DeleteProcThreadAttributeList(list);
    CloseHandle(log);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) return error = "can't start the Training add-on's Python", nullptr;
    CloseHandle(pi.hThread);
    if (pid) *pid = pi.dwProcessId;
    return pi.hProcess;
}
#endif

bool start_training(const std::string& videos, StartMode m, std::string& error) {
#ifdef _WIN32
    if (training_alive()) return error = "training is already running", false;
    if (training_addon_installed() != 1) return error = "the Training add-on is missing or outdated", false;
    const std::string scripts = training_scripts_dir();
    if (!fs::exists(fs::u8path(scripts) / "train.py")) return error = "train.py is missing in " + scripts, false;
    // the scripts folder as it is today (a reinstall elsewhere moves it)
    if (!write_python_pth((fs::u8path(training_addon_dir()) / "python").u8string(), scripts))
        return error = "can't set up the Training add-on's Python", false;
    DWORD pid = 0;
    HANDLE h = run_python(train_command(training_python(), scripts, videos, training_work_dir(), m), error, &pid);
    if (!h) return false;
    CloseHandle(h);
    write_run({pid, (long long)time(nullptr)});
    return true;
#else
    (void)videos, (void)m;
    return error = "training from the player is Windows-only", false;
#endif
}

static fs::path run_file() { return fs::u8path(training_work_dir()) / "run.txt"; }

bool read_run(RunInfo& r) {
    std::ifstream f(run_file());
    RunInfo v;
    if (!(f >> v.pid >> v.started) || !v.pid) return false;
    r = v;
    return true;
}

void write_run(const RunInfo& r) {
    std::error_code ec;
    fs::create_directories(run_file().parent_path(), ec);
    std::ofstream(run_file()) << r.pid << " " << r.started << "\n";
}

void clear_run() {
    std::error_code ec;
    fs::remove(run_file(), ec);
}

bool process_alive(unsigned long pid, const std::string& exe) {
#ifdef _WIN32
    if (!pid) return false;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return false;
    DWORD code = 0, n = 32768;
    std::wstring name(n, L'\0');
    // a process id is reused once its process ends: it must also still be the program we started
    bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE && QueryFullProcessImageNameW(h, 0, name.data(), &n);
    CloseHandle(h);
    if (!alive) return false;
    name.resize(n);
    std::wstring want = fs::u8path(exe).wstring();
    return CompareStringOrdinal(name.c_str(), -1, want.c_str(), -1, TRUE) == CSTR_EQUAL;
#else
    (void)pid, (void)exe;
    return false;
#endif
}

bool training_alive() {
    RunInfo r;
    return read_run(r) && process_alive(r.pid, training_python());
}

bool status_current(const TrainingStatus& t, long long started) { return t.updated >= started; }

std::string stop_file() { return (fs::path(training_file()).parent_path() / "training.stop").string(); }

void request_stop() {
    std::error_code ec;
    fs::create_directories(fs::path(stop_file()).parent_path(), ec);
    std::ofstream(fs::path(stop_file())) << "stop\n";
}

bool own_run(const TrainingStatus& t) {
    if (t.out.empty()) return false;
    auto norm = [](const fs::path& p) {
        std::wstring w = p.lexically_normal().wstring();
        for (auto& c : w) c = c == L'/' ? L'\\' : (wchar_t)towlower(c);
        return w;
    };
    return norm(fs::u8path(t.out)) == norm(fs::u8path(training_work_dir()) / "vsr.pt");
}

std::string train_log() { return (fs::u8path(training_work_dir()) / "train.log").u8string(); }
bool has_checkpoint() {
    std::error_code ec;
    return fs::exists(fs::u8path(training_work_dir()) / "vsr.pt", ec);
}
bool has_best() {
    std::error_code ec;
    return fs::exists(fs::u8path(training_work_dir()) / "vsr_best.pt", ec);
}

bool install_model(const std::string& work, const std::string& data_dir, std::string& error) {
    fs::path w = fs::u8path(work), d = fs::u8path(data_dir);
    std::error_code ec;
    const char* names[][2] = {{"vsr_best.onnx", "vsr.onnx"}, {"vsr_best.ref", "vsr.ref"}};
    for (auto& n : names)
        if (!fs::exists(w / n[0], ec)) return error = std::string("the export made no ") + n[0], false;
    fs::create_directories(d, ec);
    auto tmp = [&](const char* name) { return fs::path(d / name) += ".tmp"; };
    auto prev = [&](const char* name) { return d / (std::string("vsr.prev") + fs::path(name).extension().string()); };
    auto drop_tmps = [&] {
        for (auto& m : names) fs::remove(tmp(m[1]), ec);
    };
    // both copies first: a failure there leaves the old model untouched
    for (auto& n : names)
        if (!fs::copy_file(w / n[0], tmp(n[1]), fs::copy_options::overwrite_existing, ec))
            return drop_tmps(), error = "can't copy the model into " + d.u8string(), false;
    // then per file: the old one to vsr.prev.*, the new one in. A failure undoes the files done so far, so the
    // pair is never half new, half old.
    int done = 0;
    for (auto& n : names) {
        std::error_code e1, e2;
        const bool had = fs::exists(d / n[1], e1);
        if (had) fs::remove(prev(n[1]), e1), fs::rename(d / n[1], prev(n[1]), e1);
        if (!e1) fs::rename(tmp(n[1]), d / n[1], e2);
        if (e1 || e2) {
            if (had && !e1) fs::rename(prev(n[1]), d / n[1], e1);  // this file's old one back
            for (int k = 0; k < done; k++) {                       // and the files before it
                fs::remove(d / names[k][1], e1);
                fs::rename(prev(names[k][1]), d / names[k][1], e1);
            }
            drop_tmps();
            return error = "can't replace " + (d / n[1]).u8string() + " (is it in use?)", false;
        }
        done++;
    }
    return true;
}

bool export_and_install(std::string& error) {
#ifdef _WIN32
    const fs::path best = fs::u8path(training_work_dir()) / "vsr_best.pt";
    std::error_code ec;
    if (!fs::exists(best, ec)) return error = "no model yet: train first", false;
    if (!write_python_pth((fs::u8path(training_addon_dir()) / "python").u8string(), training_scripts_dir()))
        return error = "can't set up the Training add-on's Python", false;
    // from a copy: train.py replaces vsr_best.pt whenever it improves, and must never find it held open
    const fs::path dir = fs::u8path(training_work_dir()) / "export", copy = dir / "vsr_best.pt";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    if (!fs::copy_file(best, copy, ec)) return error = "can't copy " + best.u8string(), false;
    const std::string cmd = quote_arg(training_python()) + " " +
                            quote_arg((fs::u8path(training_scripts_dir()) / "export.py").u8string()) + " " +
                            quote_arg(copy.u8string());
    HANDLE h = run_python(wide(cmd), error);
    if (!h) return false;
    WaitForSingleObject(h, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(h, &code);
    CloseHandle(h);
    if (code) return error = "the export failed: see " + train_log(), false;
    return install_model(dir.u8string(), user_data_dir(), error);
#else
    return error = "training from the player is Windows-only", false;
#endif
}

bool start_over(std::string& error) {
#ifdef _WIN32
    if (training_alive()) return error = "stop training first", false;
#endif
    std::error_code ec;
    fs::remove_all(fs::u8path(training_work_dir()), ec);
    return ec ? (error = "can't delete " + training_work_dir(), false) : true;
}

static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

int training_run_selftest() {
    int fails = 0;
#define check(cond) \
    if (!(cond)) printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond), fails++
#ifdef _WIN32
    // a videos folder with a space, a non-ASCII letter and a trailing backslash reaches train.py exactly
    const std::string py = "C:\\Users\\Zo\xc3\xab\\AppData\\Local\\Open Player\\training\\python\\python.exe",
                      scripts = "C:\\Program Files\\Open Player\\training", videos = "D:\\My Vids\\Zo\xc3\xab\\",
                      work = "C:\\Users\\Zo\xc3\xab\\AppData\\Local\\Open Player\\training-work";
    auto args = [](const std::wstring& cmd) {
        int n = 0;
        LPWSTR* a = CommandLineToArgvW(cmd.c_str(), &n);
        std::vector<std::wstring> out(a, a + n);
        LocalFree(a);
        return out;
    };
    auto w = [](const std::string& s) { return fs::u8path(s).wstring(); };
    std::vector<std::wstring> base = {w(py), L"-u", w(scripts + "\\train.py"), w(videos), L"--out", w(work + "\\vsr.pt")};
    check(args(train_command(py, scripts, videos, work, StartMode::New)) == base);
    auto resume = base;
    resume.insert(resume.end(), {L"--resume", w(work + "\\vsr.pt"), L"--more-if-done", L"100000"});
    check(args(train_command(py, scripts, videos, work, StartMode::Resume)) == resume);

    // the run on disk: whether its process still runs (and is the program it should be), and whose status is whose
    wchar_t me[MAX_PATH];
    GetModuleFileNameW(nullptr, me, MAX_PATH);
    const std::string self = fs::path(me).u8string();
    check(process_alive(GetCurrentProcessId(), self));
    check(!process_alive(GetCurrentProcessId(), "C:\\nowhere\\python.exe"));  // a reused id running something else
    check(!process_alive(0, self) && !process_alive(0xFFFFFFF0u, self));
    TrainingStatus st;
    st.updated = 1000;
    check(status_current(st, 1000) && status_current(st, 990) && !status_current(st, 1001));

    // Stop: training.stop next to training.json (XDG_CONFIG_HOME moves both, as train.py's tests do)
    fs::path cfg = fs::temp_directory_path() / "op-training-run-cfg";
    std::error_code ec;
    fs::remove_all(cfg, ec);
    _putenv_s("XDG_CONFIG_HOME", cfg.string().c_str());
    check(fs::path(stop_file()).filename() == "training.stop" &&
          fs::path(stop_file()).parent_path() == fs::path(training_file()).parent_path());
    request_stop();
    check(fs::exists(stop_file()));
    _putenv_s("XDG_CONFIG_HOME", "");
    fs::remove_all(cfg, ec);

    // own_run: only a run whose out is the work folder's vsr.pt (any case, either slash)
    TrainingStatus t;
    t.out = training_work_dir() + "\\vsr.pt";
    check(own_run(t));
    for (auto& c : t.out) c = c == '\\' ? '/' : (char)toupper((unsigned char)c);
    check(own_run(t));
    t.out = "D:\\X\\vsr.pt";
    check(!own_run(t));
    t.out = "";
    check(!own_run(t));

    // installing an exported model: both files replace the old ones, no temp file left
    fs::path tw = fs::temp_directory_path() / fs::u8path("op-zo\xc3\xab-install"), wk = tw / "work", ud = tw / "data";
    fs::remove_all(tw, ec);
    fs::create_directories(wk), fs::create_directories(ud);
    std::ofstream(wk / "vsr_best.onnx") << "ONNX";
    std::ofstream(wk / "vsr_best.ref") << "REF";
    std::ofstream(ud / "vsr.onnx") << "old";
    std::ofstream(ud / "vsr.ref") << "oldref";
    std::string err;
    check(install_model(wk.u8string(), ud.u8string(), err));
    check(slurp(ud / "vsr.onnx") == "ONNX" && slurp(ud / "vsr.ref") == "REF");
    // the model it replaced is kept, never lost (it may be one trained for days)
    check(slurp(ud / "vsr.prev.onnx") == "old" && slurp(ud / "vsr.prev.ref") == "oldref");
    check(std::distance(fs::directory_iterator(ud), {}) == 4);  // no .tmp left
    // vsr.ref held open without sharing (antivirus, another player): its swap fails after vsr.onnx's, and the
    // pair is put back as it was, never a new .onnx with an old .ref
    std::ofstream(wk / "vsr_best.onnx") << "NEW";
    HANDLE lock = CreateFileW((ud / "vsr.ref").wstring().c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    check(lock != INVALID_HANDLE_VALUE);
    check(!install_model(wk.u8string(), ud.u8string(), err));
    CloseHandle(lock);
    check(slurp(ud / "vsr.onnx") == "ONNX" && slurp(ud / "vsr.ref") == "REF");
    check(!fs::exists(ud / "vsr.onnx.tmp") && !fs::exists(ud / "vsr.ref.tmp"));

    // run.txt: written, read back, cleared (in a temp data folder)
    _putenv_s("XDG_DATA_HOME", (fs::temp_directory_path() / "op-training-run-ud").string().c_str());
    RunInfo ri{1234, 1795300000}, back;
    check(!read_run(back));
    write_run(ri);
    check(read_run(back) && back.pid == 1234 && back.started == 1795300000);
    check(!training_alive());  // 1234 isn't the add-on's python.exe
    clear_run();
    check(!read_run(back));
    fs::remove_all(fs::temp_directory_path() / "op-training-run-ud", ec);
    _putenv_s("XDG_DATA_HOME", "");
    fs::remove(wk / "vsr_best.ref");
    check(!install_model(wk.u8string(), ud.u8string(), err) && slurp(ud / "vsr.onnx") == "ONNX");  // half export: refused
    fs::remove_all(tw, ec);
#endif
#undef check
    return fails ? 1 : 0;
}
