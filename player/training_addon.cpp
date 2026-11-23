#include "training_addon.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "addon.h"
#include "trt_loader.h"
#include "update.h"

namespace fs = std::filesystem;

// Python 3.13.17 (the newest 3.13 with an embeddable build), PyTorch 2.15 for CUDA 13.2 and what it, numpy
// and onnx need (pip's resolution for cp313 win_amd64, hashes checked against PyPI's), FFmpeg 9.1 (Gyan's
// essentials build, GPLv3; only ffmpeg.exe and ffprobe.exe are kept). Proven 2026-11-23: tests/train_check.py
// passes and exits cleanly on exactly this set, DataLoader workers and export.py included.
const std::vector<TrainingPin> TRAINING_PINS = {
    {"https://www.python.org/ftp/python/3.13.17/python-3.13.17-embed-amd64.zip",
     "9e35319bffedab8b36ad44d8f377cb4f1733bcec257a0b6d04d94f01486ddb38", 11441267, "python"},
    {"https://download-r2.pytorch.org/whl/cu132/torch-2.15.0%2Bcu132-cp313-cp313-win_amd64.whl",
     "7b54774dedca2a1147f1ae47451d9a8a7e88b80665fc505bf6a4f1f427d21e97", 2108256835, "wheel"},
    {"https://files.pythonhosted.org/packages/05/57/f26eb770c89dd56fe29d6f4458c1c8e496afde034fe74fda403183f11dbc/ml_dtypes-0.7.0-cp313-cp313-win_amd64.whl",
     "8251e1bd8f68ebf9be6b0cad5ff6acee0e1768f2f45d55addda1a727e81503bf", 463435, "wheel"},
    {"https://files.pythonhosted.org/packages/38/ce/102fd4a0b2a6d111a9c86745e084c4c68c0ee020eaa359a03a8d43e4646f/onnx-1.23.2-cp312-abi3-win_amd64.whl",
     "9b382ba898a7c142a0801d03cf04ecabced96c1543c7b643a86f0928143802de", 7872489, "wheel"},
    {"https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl",
     "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c", 536198, "wheel"},
    {"https://files.pythonhosted.org/packages/45/74/bb9161943fb65caf24e4bb04c788f5c7815e6cb78ac595e7f46781033c82/numpy-2.5.5-cp313-cp313-win_amd64.whl",
     "e0608d1585557b68dcb6a70a41654f7536567960afad79d91752d2863b242b68", 12573354, "wheel"},
    {"https://files.pythonhosted.org/packages/49/d3/b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/typing_extensions-4.16.0-py3-none-any.whl",
     "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8", 45571, "wheel"},
    {"https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl",
     "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67", 134899, "wheel"},
    {"https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl",
     "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c", 129956, "wheel"},
    {"https://files.pythonhosted.org/packages/68/66/870b527908a06c7ec0e91239bd2c4d0bb22921efd4f915428b4e47d461a6/filelock-4.3.0-py3-none-any.whl",
     "f933f39f7e01249a4a49b9abab30edc6d4b874624c3de5dd0b0d137aa8adb979", 244553, "wheel"},
    {"https://files.pythonhosted.org/packages/6c/c0/a98505f18594f1bce828bb159cec0fcf9860562f1a2c85913409fc8f3d9e/fsspec-2026.9.0-py3-none-any.whl",
     "8dd6e646e99ea382bd85f97a45e6b526a442d79423a7dc673f1e2756d05fcb5f", 221738, "wheel"},
    {"https://files.pythonhosted.org/packages/6f/36/163da64de88a13db79214ef75fa041be7fa13bdb42261cf5b7484de14bfb/markupsafe-3.0.4-cp313-cp313-win_amd64.whl",
     "1f1f9477e174582b0a1b583d60b66e1f2cf5d3fe12cee985e4aedf44766600e5", 14313, "wheel"},
    {"https://files.pythonhosted.org/packages/7e/cd/fe58041e9011f307c490e3e17dd48cc516448f7c698a3f2d9d9d65d7e6a8/networkx-3.7-py3-none-any.whl",
     "e3fd2c13a7814cee3746340d8d7f8598a67f16a58bf47fb7f8793fab6efca1b0", 2142205, "wheel"},
    {"https://files.pythonhosted.org/packages/8a/55/b77bda4e5e5f5971fb51b07663694690e9afdb9402136c16a522bd621cad/protobuf-7.36.2-cp310-abi3-win_amd64.whl",
     "a300819d441e078a5608c0d3c709796bb548136058fda017ae51d425b44fd353", 456514, "wheel"},
    {"https://files.pythonhosted.org/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl",
     "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5", 6299353, "wheel"},
    {"https://files.pythonhosted.org/packages/b9/3b/06260997cdc41138e58718588a6c87d0eb342bbe0dda8a6aae91d163c384/onnxscript-0.7.2-py3-none-any.whl",
     "d0e7121c6a1eefd608058928e111cbdb76709f70d269ff0d07aee493bd1d13c9", 754215, "wheel"},
    {"https://files.pythonhosted.org/packages/eb/80/1a63440081d94e25b96be6b29119f41119fe5c9521063edbdb7baeeb9589/onnx_ir-1.1.0-py3-none-any.whl",
     "76cc313a788c073b75271c1b6abb172aaf90fcc6335ed40f9f3238b2064a2fa3", 192241, "wheel"},
    {"https://files.pythonhosted.org/packages/ef/55/6dc6d6bb953bdfc52519a0059592c52b6e3b8ad51cf8256dffceb55b0610/setuptools-84.2.1-py3-none-any.whl",
     "ca140b45d5325a9478f880e351484f62fef15f336462e449ca0c03c873d589c2", 876263, "wheel"},
    {"https://github.com/GyanD/codexffmpeg/releases/download/9.1/ffmpeg-9.1-essentials_build.zip",
     "b9bbbb3acece000cbeec994f856649354e52c773807841e78908374b24d658fd", 119467219, "ffmpeg"},
};

// every host a pin's download may touch, redirects included (GitHub hands releases to its asset host)
static const std::vector<std::string> HOSTS = {"www.python.org", "files.pythonhosted.org", "download.pytorch.org",
                                               "download-r2.pytorch.org", "github.com",
                                               "release-assets.githubusercontent.com", "objects.githubusercontent.com"};

uint64_t training_addon_size() {
    uint64_t n = 0;
    for (auto& p : TRAINING_PINS) n += p.size;
    return n;
}

int training_addon_installed() {
    std::ifstream f(fs::u8path(training_addon_dir()) / "version.txt");
    int v = -1;
    if (!(f >> v)) return 0;
    return v == TRAINING_ADDON_VERSION ? 1 : 2;
}

std::string training_python() { return (fs::u8path(training_addon_dir()) / "python" / "python.exe").u8string(); }
std::string training_ffmpeg_dir() { return (fs::u8path(training_addon_dir()) / "ffmpeg").u8string(); }

std::string training_scripts_dir() {
    fs::path exe = fs::u8path(program_dir());
    std::error_code ec;
    if (!fs::exists(exe / "training" / "train.py", ec) && fs::exists(exe.parent_path() / "train.py", ec))
        return exe.parent_path().u8string();  // a developer build: build-win\ in the repo
    return (exe / "training").u8string();
}

bool write_python_pth(const std::string& python_dir, const std::string& scripts_dir) {
    std::error_code ec;
    for (auto& e : fs::directory_iterator(fs::u8path(python_dir), ec)) {
        if (e.path().extension() != "._pth" && e.path().extension() != ".pth") continue;
        if (e.path().filename().u8string().find("._pth") == std::string::npos) continue;
        // embeddable Python reads only this file for sys.path (no PYTHONPATH, not even the script's folder)
        std::string zip = e.path().stem().u8string() + ".zip";
        fs::path tmp = e.path();
        tmp += ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary);
            f << zip << "\n.\nLib\\site-packages\n" << scripts_dir << "\nimport site\n";
            if (!f) return false;
        }
        fs::rename(tmp, e.path(), ec);
        return !ec;
    }
    return false;
}

bool unpack_training_file(const TrainingPin& pin, const std::string& file, const std::string& into,
                          std::string& error) {
    std::error_code ec;
    fs::path in = fs::u8path(into), zip = fs::u8path(file);
    std::string kind = pin.kind;
    auto tar = [&](const fs::path& dest, const std::vector<std::wstring>& patterns) {
        fs::create_directories(dest, ec);
        return !ec && tar_extract(zip, dest, patterns) ? true : (error = "couldn't unpack " + file, false);
    };
    if (kind == "python") {
        if (!tar(in / "python", {})) return false;
        return fs::exists(in / "python" / "python.exe", ec) ? true : (error = "no python.exe in the download", false);
    }
    if (kind == "wheel") return tar(in / "python" / "Lib" / "site-packages", {});
    if (kind != "ffmpeg") return error = "unknown kind " + kind, false;
    fs::path raw = in / "ffmpeg.raw", out = in / "ffmpeg";
    fs::remove_all(raw, ec);
    if (!tar(raw, {L"*/bin/ffmpeg.exe", L"*/bin/ffprobe.exe", L"*/LICENSE"})) return false;
    fs::create_directories(out, ec);
    for (auto& e : fs::recursive_directory_iterator(raw, ec))
        if (e.is_regular_file(ec)) fs::rename(e.path(), out / e.path().filename(), ec);
    fs::remove_all(raw, ec);
    if (!fs::exists(out / "ffmpeg.exe", ec) || !fs::exists(out / "ffprobe.exe", ec))
        return error = "no ffmpeg.exe/ffprobe.exe in the download", false;
    return true;
}

bool install_training_addon(const std::function<void(uint64_t, uint64_t)>& progress, const std::atomic<bool>& cancel,
                            std::string& error) {
#ifdef _WIN32
    fs::path base = fs::u8path(user_data_dir()), dir = fs::u8path(training_addon_dir()), stage = dir,
             dl = base / "training-download";
    stage += ".new";
    std::error_code ec;
    auto fail = [&](const std::string& why) {
        fs::remove_all(stage, ec), fs::remove_all(dl, ec);
        return error = why, false;
    };
    fs::create_directories(base, ec);
    // the unpacked add-on (3.3 GB) and the largest download (2.1 GB) are both on disk for a moment
    if (fs::space(base, ec).available < 6000ull << 20 && !ec) return error = "needs 6 GB of free disk space", false;
    fs::remove_all(stage, ec), fs::remove_all(dl, ec);
    if (!fs::create_directories(stage, ec) || !fs::create_directories(dl, ec)) return fail("can't create " + stage.u8string());
    const uint64_t total = training_addon_size();
    uint64_t done = 0;
    for (size_t i = 0; i < TRAINING_PINS.size(); i++) {
        const TrainingPin& pin = TRAINING_PINS[i];
        fs::path file = dl / ("pin-" + std::to_string(i) + ".zip");
        if (!https_download(pin.url, file.u8string(), pin.size, HOSTS,
                            [&](uint64_t got, uint64_t) { progress(done + got, total); }, cancel, error))
            return fail(error);
        if (cancel) return fail("cancelled");
        if (!verify_file_sha256(file.u8string(), pin.sha256)) return fail(std::string("a download doesn't match its "
                                                                                      "published file: ") + pin.url);
        if (!unpack_training_file(pin, file.u8string(), stage.u8string(), error)) return fail(error);
        fs::remove(file, ec);
        done += pin.size;
        if (cancel) return fail("cancelled");
    }
    // a cut-off unpacking (disk full) is refused here, not crashed on at the first Start
    for (const char* f : {"python/python.exe", "python/Lib/site-packages/torch/__init__.py",
                          "python/Lib/site-packages/numpy/__init__.py", "python/Lib/site-packages/onnx/__init__.py",
                          "ffmpeg/ffmpeg.exe", "ffmpeg/ffprobe.exe"})
        if (!fs::exists(stage / fs::u8path(f), ec)) return fail(std::string("the add-on is incomplete (no ") + f + ")");
    if (!write_python_pth((stage / "python").u8string(), training_scripts_dir())) return fail("can't set up Python");
    std::ofstream(stage / "version.txt") << TRAINING_ADDON_VERSION << "\n";
    fs::remove_all(dl, ec);
    fs::remove_all(dir, ec);  // the old add-on: fails while its python.exe runs
    if (ec) return fail("stop training first: the current Training add-on is in use");
    fs::rename(stage, dir, ec);
    return ec ? fail("can't move the add-on into place") : true;
#else
    (void)progress, (void)cancel;
    return error = "the Training add-on is Windows-only", false;
#endif
}

void remove_training_addon(bool in_use) {
    std::error_code ec;
    if (!in_use) fs::remove_all(fs::u8path(training_addon_dir()), ec);
    if (in_use || ec) std::ofstream(fs::u8path(user_data_dir()) / "remove-training.txt") << "remove at next start\n";
}

void finish_pending_training_removal() {
    std::error_code ec;
    fs::path marker = fs::u8path(user_data_dir()) / "remove-training.txt";
    if (!fs::exists(marker, ec)) return;
    fs::remove_all(fs::u8path(training_addon_dir()), ec);
    if (!ec) fs::remove(marker, ec);
}

static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

int training_addon_selftest() {
    int fails = 0;
#define check(cond) \
    if (!(cond)) printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond), fails++

    // the pins: every one https, a 64-hex SHA-256, a size, a known kind; exactly one python and one ffmpeg
    int py = 0, ff = 0;
    for (auto& p : TRAINING_PINS) {
        std::string k = p.kind, u = p.url, h = p.sha256;
        check(u.rfind("https://", 0) == 0 && h.size() == 64 && p.size > 0);
        check(k == "python" || k == "wheel" || k == "ffmpeg");
        py += k == "python", ff += k == "ffmpeg";
    }
    check(py == 1 && ff == 1 && TRAINING_PINS.size() > 10);
    check(training_addon_size() > (2000ull << 20));  // the add-on is about 2.1 GB
#ifdef _WIN32
    fs::path base = fs::temp_directory_path() / fs::u8path("op-zo\xc3\xab-training-test");
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base);
    fs::path into = base / "training.new";
    std::string err;
    TrainingPin wheel{"https://x/w.whl", "", 1, "wheel"}, python{"https://x/p.zip", "", 1, "python"},
        ffmpeg{"https://x/f.zip", "", 1, "ffmpeg"};

    // a wheel unpacks with its folders into python\Lib\site-packages; ".." entries land nowhere
    write_zip(base / "w.whl", {{"pkg/__init__.py", "A"}, {"pkg/sub/m.py", "B"}, {"pkg/../../../evil.py", "E"}});
    check(unpack_training_file(wheel, (base / "w.whl").u8string(), into.u8string(), err));
    fs::path sp = into / "python" / "Lib" / "site-packages";
    check(slurp(sp / "pkg" / "__init__.py") == "A" && slurp(sp / "pkg" / "sub" / "m.py") == "B");
    check(!fs::exists(base / "evil.py") && !fs::exists(into / "evil.py") && !fs::exists(sp / "evil.py"));

    // the embeddable zip unpacks into python\, and its ._pth is pointed at the packages and the scripts
    write_zip(base / "p.zip", {{"python.exe", "X"}, {"python313._pth", "python313.zip\n.\n#import site\n"}});
    check(unpack_training_file(python, (base / "p.zip").u8string(), into.u8string(), err));
    check(slurp(into / "python" / "python.exe") == "X");
    check(write_python_pth((into / "python").u8string(), "D:\\Open Player\\training"));
    std::string pth = slurp(into / "python" / "python313._pth");
    check(pth.find("python313.zip\n") != std::string::npos && pth.find("Lib\\site-packages\n") != std::string::npos);
    check(pth.find("D:\\Open Player\\training\n") != std::string::npos && pth.find("\nimport site") != std::string::npos);
    check(pth.find("#import site") == std::string::npos);

    // FFmpeg's zip: only ffmpeg.exe, ffprobe.exe and the licence, flattened into the ffmpeg folder
    write_zip(base / "f.zip", {{"ffmpeg-9.1/bin/ffmpeg.exe", "F"}, {"ffmpeg-9.1/bin/ffprobe.exe", "P"},
                               {"ffmpeg-9.1/bin/ffplay.exe", "Y"}, {"ffmpeg-9.1/LICENSE", "L"},
                               {"ffmpeg-9.1/doc/x.html", "D"}});
    check(unpack_training_file(ffmpeg, (base / "f.zip").u8string(), into.u8string(), err));
    check(slurp(into / "ffmpeg" / "ffmpeg.exe") == "F" && slurp(into / "ffmpeg" / "ffprobe.exe") == "P");
    check(fs::exists(into / "ffmpeg" / "LICENSE") && !fs::exists(into / "ffmpeg" / "ffplay.exe"));
    check(!fs::exists(into / "ffmpeg" / "x.html") && !fs::exists(into / "ffmpeg" / "ffmpeg-9.1"));

    // an ffmpeg zip without ffmpeg.exe is refused
    write_zip(base / "bad.zip", {{"ffmpeg-9.1/LICENSE", "L"}});
    check(!unpack_training_file(ffmpeg, (base / "bad.zip").u8string(), (base / "b2").u8string(), err));

    // version.txt says which pins the installed add-on has (training_addon_installed reads user_data_dir())
    fs::path ud = fs::temp_directory_path() / "op-training-ud";  // ASCII: getenv hands back narrow strings
    fs::remove_all(ud, ec);
    _putenv_s("XDG_DATA_HOME", ud.string().c_str());
    check(training_addon_installed() == 0);
    fs::create_directories(fs::u8path(training_addon_dir()));
    check(training_addon_installed() == 0);  // a folder without version.txt is not an add-on
    std::ofstream(fs::u8path(training_addon_dir()) / "version.txt") << TRAINING_ADDON_VERSION << "\n";
    check(training_addon_installed() == 1);
    std::ofstream(fs::u8path(training_addon_dir()) / "version.txt") << TRAINING_ADDON_VERSION - 1 << "\n";
    check(training_addon_installed() == 2);
    check(training_python() == (fs::u8path(training_addon_dir()) / "python" / "python.exe").u8string());
    // a cancelled install (the same path as a failed one) leaves no training.new or download, the old add-on intact
    {
        std::atomic<bool> cancel{true};
        std::string e;
        check(!install_training_addon([](uint64_t, uint64_t) {}, cancel, e));
        check(!fs::exists(fs::u8path(training_addon_dir() + ".new")) && !fs::exists(ud / "Open Player" / "training-download"));
        check(training_addon_installed() == 2);  // the version.txt written above is still there
    }
    // removing while not in use deletes it; in use marks it, and the next start finishes the job
    remove_training_addon(true);
    check(fs::exists(fs::u8path(training_addon_dir())));
    finish_pending_training_removal();
    check(!fs::exists(fs::u8path(training_addon_dir())));
    fs::create_directories(fs::u8path(training_addon_dir()));
    remove_training_addon(false);
    check(!fs::exists(fs::u8path(training_addon_dir())));
    _putenv_s("XDG_DATA_HOME", "");
    fs::remove_all(ud, ec);
    fs::remove_all(base, ec);
#endif
#undef check
    return fails ? 1 : 0;
}
