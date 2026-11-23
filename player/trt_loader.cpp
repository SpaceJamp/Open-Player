#include "trt_loader.h"

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
#undef check
    return fails ? 1 : 0;
}
