#include "trt_loader.h"
#include "rtx_backend.h"

#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "settings.h"
#include "ui.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#ifdef HAVE_RTX_VSR
#include "nvsdk_ngx_defs.h"
#endif
#else
#include <NvInfer.h>
#endif

namespace fs = std::filesystem;

static fs::path exe_dir() {
#ifdef _WIN32
    wchar_t p[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(nullptr, p, (DWORD)std::size(p));
    return fs::path(std::wstring(p, n)).parent_path();
#else
    std::error_code ec;
    return fs::read_symlink("/proc/self/exe", ec).parent_path();
#endif
}

std::vector<std::string> tensorrt_candidates(const std::string& user_folder, const std::string& addon,
                                             const std::string& exe) {
    std::vector<std::string> out;
    for (const auto& d : {user_folder, addon, exe})
        if (!d.empty()) out.push_back(d);
    return out;
}

std::string user_data_dir() {
    const char* xdg = getenv("XDG_DATA_HOME");  // also on Windows: the tests point it at a temp folder
#ifdef _WIN32
    if (xdg && *xdg) return (fs::u8path(xdg) / "Open Player").u8string();
    // the real Local AppData folder from Windows, never a relative "." (that could be a video's folder)
    PWSTR local = nullptr;
    fs::path base = SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &local)) ? fs::path(local)
                                                                                                : fs::temp_directory_path();
    CoTaskMemFree(local);
    return (base / "Open Player").u8string();
#else
    const char* home = getenv("HOME");
    return ((xdg && *xdg ? fs::path(xdg) : fs::path(home ? home : ".") / ".local" / "share") / "open-player").string();
#endif
}

std::string addon_dir() { return (fs::u8path(user_data_dir()) / "tensorrt").u8string(); }
std::string rtx_addon_dir() { return (fs::u8path(user_data_dir()) / "tensorrt-rtx").u8string(); }
std::string training_addon_dir() { return (fs::u8path(user_data_dir()) / "training").u8string(); }
std::string training_work_dir() { return (fs::u8path(user_data_dir()) / "training-work").u8string(); }
std::string program_dir() { return exe_dir().u8string(); }

bool installed_copy() {
    std::error_code ec;
    return fs::exists(exe_dir() / "installed.txt", ec);
}

std::string trt_line(TrtFrom from, const std::string& version, const std::string& why_not) {
    if (from == TrtFrom::None)
        return "TensorRT: " + (why_not.empty() ? std::string("not installed") : why_not) +
               " (Pixel Flow needs the TensorRT add-on)";
    const char* where = from == TrtFrom::AddOn ? "add-on" : from == TrtFrom::UserFolder ? "your folder"
                                                                                         : "next to the program";
    return "TensorRT: " + version + " (" + where + ")";
}

static std::string dotted(int v) {  // major*10000 + minor*100 + patch
    return std::to_string(v / 10000) + "." + std::to_string(v / 100 % 100) + "." + std::to_string(v % 100);
}

TrtFrom load_tensorrt(const std::string& user_folder, std::string& version, std::string& why_not) {
#ifdef _WIN32
    const TrtFrom kinds[] = {TrtFrom::UserFolder, TrtFrom::AddOn, TrtFrom::NextToExe};
    const std::string dirs[] = {user_folder, addon_dir(), exe_dir().u8string()};
    why_not = "not installed";
    for (int i = 0; i < 3; i++) {
        if (dirs[i].empty()) continue;
        fs::path d = fs::u8path(dirs[i]);
        std::error_code ec;
        if (!fs::exists(d / "nvinfer_11.dll", ec)) {
            if (i == 0) why_not = "not found in " + dirs[i];
            continue;
        }
        // nvinfer loads its builder DLLs by name: from this folder, not the current one or PATH
        SetDllDirectoryW(d.wstring().c_str());
        HMODULE inf = LoadLibraryExW((d / "nvinfer_11.dll").wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        HMODULE onnx = inf ? LoadLibraryExW((d / "nvonnxparser_11.dll").wstring().c_str(), nullptr,
                                            LOAD_WITH_ALTERED_SEARCH_PATH)
                           : nullptr;
        auto ver = inf ? (int32_t(*)())GetProcAddress(inf, "getInferLibVersion") : nullptr;
        int v = ver ? ver() : 0;
        if (!inf || !onnx || !ver || v / 10000 != 11) {
            why_not = v && v / 10000 != 11 ? "TensorRT " + dotted(v) + " in " + dirs[i] + " (needs 11.x)"
                                            : "can't load TensorRT from " + dirs[i];
            // unloaded again, so a broken add-on can be replaced and the next folder's DLLs can load
            if (onnx) FreeLibrary(onnx);
            if (inf) FreeLibrary(inf);
            SetDllDirectoryW(L"");  // back to: never the current folder
            continue;
        }
        version = dotted(v);
        return kinds[i];
    }
    return TrtFrom::None;
#else
    (void)user_folder, (void)why_not;
    version = dotted(getInferLibVersion());  // linked as usual on Linux
    return TrtFrom::NextToExe;
#endif
}

std::string rtx_vsr_line(bool tried, bool available) {
#if defined(_WIN32) && defined(HAVE_RTX_VSR)
    std::string api = std::to_string((NVSDK_NGX_VERSION_API_MACRO >> 4) & 0xf) + "." +
                      std::to_string(NVSDK_NGX_VERSION_API_MACRO & 0xf) + ".0";
    if (tried && !available) return "NVIDIA RTX VSR: not available on this GPU or driver (SDK API " + api + ")";
    // the version Windows shows in the file's Properties > Details: nothing inside the DLL is inspected
    wchar_t path[MAX_PATH * 4];
    HMODULE m = GetModuleHandleW(L"nvngx_vsr.dll");
    DWORD n = m ? GetModuleFileNameW(m, path, (DWORD)std::size(path)) : 0;
    std::wstring file = n ? std::wstring(path, n) : (exe_dir() / "nvngx_vsr.dll").wstring();
    std::string ver = "unknown version";
    DWORD dummy = 0, size = GetFileVersionInfoSizeW(file.c_str(), &dummy);
    std::vector<char> buf(size);
    VS_FIXEDFILEINFO* fi = nullptr;
    UINT len = 0;
    if (size && GetFileVersionInfoW(file.c_str(), 0, size, buf.data()) &&
        VerQueryValueW(buf.data(), L"\\", (void**)&fi, &len) && fi)
        ver = std::to_string(HIWORD(fi->dwFileVersionMS)) + "." + std::to_string(LOWORD(fi->dwFileVersionMS)) + "." +
              std::to_string(HIWORD(fi->dwFileVersionLS)) + "." + std::to_string(LOWORD(fi->dwFileVersionLS));
    return "NVIDIA RTX VSR: " + ver + " (SDK API " + api + ")" + (tried ? "" : ", not in use");
#else
    (void)tried, (void)available;
    return "NVIDIA RTX VSR: not in this build";
#endif
}

std::string driver_line() {
    int v = 0;
    if (cuDriverGetVersion(&v) != CUDA_SUCCESS || !v) return "NVIDIA driver: not found";
    return "NVIDIA driver: CUDA " + std::to_string(v / 1000) + "." + std::to_string(v % 1000 / 10);
}

Engine choose_engine(bool full_ok, bool rtx_ok) { return full_ok ? Engine::Full : rtx_ok ? Engine::Rtx : Engine::None; }
std::string rtx_dll_name(int minor) { return "tensorrt_rtx_1_" + std::to_string(minor) + ".dll"; }
std::string rtx_parser_name(int minor) { return "tensorrt_onnxparser_rtx_1_" + std::to_string(minor) + ".dll"; }

const std::vector<int>& rtx_minors() {
    static const std::vector<int> v = {
#ifdef HAVE_TRT_RTX_1_8
        8,
#endif
#ifdef HAVE_TRT_RTX_1_7
        7,
#endif
    };
    return v;
}

#ifdef _WIN32
static HMODULE g_rtx = nullptr, g_rtx_parser = nullptr;  // kept loaded: the engine code takes its entry points
#endif
void* rtx_module() {
#ifdef _WIN32
    return g_rtx;
#else
    return nullptr;
#endif
}
void* rtx_parser_module() {
#ifdef _WIN32
    return g_rtx_parser;
#else
    return nullptr;
#endif
}

bool load_trt_rtx(const std::string& user_folder, std::string& version, int& minor, std::string& why_not) {
    minor = -1;
    why_not = rtx_minors().empty() ? "this build has no TensorRT for RTX" : "not installed";
#ifdef _WIN32
    for (int m : rtx_minors())  // newest first: a 1.8 you chose wins over the 1.7 add-on
        for (const std::string& dir : {user_folder, rtx_addon_dir(), exe_dir().u8string()}) {
            if (dir.empty()) continue;
            const fs::path d = fs::u8path(dir);
            std::error_code ec;
            if (!fs::exists(d / rtx_dll_name(m), ec)) continue;
            SetDllDirectoryW(d.wstring().c_str());
            HMODULE lib = LoadLibraryExW((d / rtx_dll_name(m)).wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            HMODULE parser = lib ? LoadLibraryExW((d / rtx_parser_name(m)).wstring().c_str(), nullptr,
                                                  LOAD_WITH_ALTERED_SEARCH_PATH)
                                 : nullptr;
            SetDllDirectoryW(L"");  // back to: never the current folder
            auto ver = lib ? (int32_t(*)())GetProcAddress(lib, "getInferLibVersion") : nullptr;
            const int v = ver ? ver() : 0;
            if (!lib || !parser || !ver || v / 10000 != 1 || v / 100 % 100 != m) {  // the DLL must be what its name says
                why_not = "can't load TensorRT for RTX from " + dir;
                if (parser) FreeLibrary(parser);
                if (lib) FreeLibrary(lib);
                continue;
            }
            g_rtx = lib, g_rtx_parser = parser, version = dotted(v), minor = m;
            return true;
        }
#else
    (void)user_folder, (void)version;
#endif
    return false;
}

std::string engine_line(Engine e, const std::string& full_line, const std::string& rtx_version, const std::string& rtx_why) {
    (void)rtx_why;
    if (e == Engine::Full) return "Pixel Flow engine: TensorRT " + full_line.substr(full_line.find(':') + 2);
    if (e == Engine::Rtx) return "Pixel Flow engine: TensorRT for RTX " + rtx_version;
    return "Pixel Flow engine: none (download TensorRT for RTX in Settings > Upscaling)";
}

int trt_loader_selftest() {
    int fails = 0;
#define check(cond) \
    if (!(cond)) printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond), fails++

    check(tensorrt_candidates("D:/my trt", "C:/a", "C:/b") == (std::vector<std::string>{"D:/my trt", "C:/a", "C:/b"}));
    check(tensorrt_candidates("", "C:/a", "C:/b") == (std::vector<std::string>{"C:/a", "C:/b"}));
    check(trt_line(TrtFrom::AddOn, "11.5.0", "") == "TensorRT: 11.5.0 (add-on)");
    check(trt_line(TrtFrom::UserFolder, "11.7.1", "") == "TensorRT: 11.7.1 (your folder)");
    check(trt_line(TrtFrom::None, "", "not installed") ==
          "TensorRT: not installed (Pixel Flow needs the TensorRT add-on)");
    std::string about = about_text("build 7 (abc1234)", "", {"TensorRT: 11.5.0 (add-on)", "NVIDIA driver: CUDA 13.5"});
    check(about.find("TensorRT: 11.5.0 (add-on)\nNVIDIA driver: CUDA 13.5") != std::string::npos);
    Settings s;
    s.trt_folder = "C:/Users/Zo\xc3\xab/op-zo\xc3\xab/TensorRT-11.7";
    check(settings_parse(settings_format(s)).trt_folder == s.trt_folder);
    check(settings_parse("trt_folder=a\nb\n").trt_folder == "a");
    check(!user_data_dir().empty() && addon_dir().find(user_data_dir()) == 0);

    // the engine: the full TensorRT when it's there (faster), else TensorRT for RTX
    check(choose_engine(true, true) == Engine::Full && choose_engine(false, true) == Engine::Rtx);
    check(choose_engine(true, false) == Engine::Full && choose_engine(false, false) == Engine::None);
    check(engine_line(Engine::Full, "TensorRT: 11.5.0 (add-on)", "", "") == "Pixel Flow engine: TensorRT 11.5.0 (add-on)");
    check(engine_line(Engine::Rtx, "TensorRT: not installed", "1.7.1", "") ==
          "Pixel Flow engine: TensorRT for RTX 1.7.1");
    check(engine_line(Engine::None, "TensorRT: not installed", "", "not installed") ==
          "Pixel Flow engine: none (download TensorRT for RTX in Settings > Upscaling)");
    check(rtx_dll_name(8) == "tensorrt_rtx_1_8.dll" && rtx_parser_name(7) == "tensorrt_onnxparser_rtx_1_7.dll");
    // each compiled-in TensorRT for RTX copy was built with its own version's headers (1.<minor>.x)
    for (int m : rtx_minors()) check(trt_rtx_header_version(m) / 10000 == 1 && trt_rtx_header_version(m) / 100 % 100 == m);
#ifdef _WIN32
    check(rtx_minors().size() >= 1);  // the build machine has an SDK: the release must carry TensorRT for RTX
#endif
    {  // nothing installed (a temp data folder): not loaded, and says why
        _putenv_s("XDG_DATA_HOME", (fs::temp_directory_path() / "op-rtx-none").string().c_str());
        std::string ver, why;
        int minor = -1;
        check(!load_trt_rtx("", ver, minor, why) && !why.empty() && minor == -1);
        _putenv_s("XDG_DATA_HOME", "");
    }
#undef check
    return fails ? 1 : 0;
}
