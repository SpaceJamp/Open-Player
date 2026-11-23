#pragma once
// Training from the player: train.py on the Training add-on's Python, in its own process (no window, none of the
// player's handles), so closing or restarting the player doesn't stop it. The run is recorded on disk (run.txt: its
// process id and start time), so any later player knows it's running, can stop it (a training.stop file next to
// training.json) and, once it has finished, exports its best checkpoint as the Pixel Flow the player loads.
#include <string>

#include "settings.h"

// New: no checkpoint yet. Resume: continue the checkpoint, or another 100000 steps if it has finished.
enum class StartMode { New, Resume };

// "<python>" -u "<scripts>\train.py" "<videos>" --out "<work>\vsr.pt" [--resume "<work>\vsr.pt" --more-if-done 100000]
std::wstring train_command(const std::string& python, const std::string& scripts, const std::string& videos,
                           const std::string& work, StartMode m);
// Starts train.py on videos in training_work_dir(); its output goes to train.log there. Refused while one runs.
bool start_training(const std::string& videos, StartMode m, std::string& error);

struct RunInfo {
    unsigned long pid = 0;
    long long started = 0;  // unix time
};
bool read_run(RunInfo& r);                          // training_work_dir()\run.txt; false: none
void write_run(const RunInfo& r);
void clear_run();
bool process_alive(unsigned long pid, const std::string& exe);  // pid is running and is exe
bool training_alive();  // the player's run (run.txt) is still going
// training.json was written by the run that started at started (not left over from an earlier one)
bool status_current(const TrainingStatus& t, long long started);

void request_stop();                    // writes training.stop; train.py saves and exits at its next step
std::string stop_file();                // training.stop, next to training.json
bool own_run(const TrainingStatus& t);  // t is the player's own run (its out is training_work_dir()\vsr.pt)
std::string train_log();                // training_work_dir()\train.log
bool has_checkpoint();                  // training_work_dir()\vsr.pt exists
bool has_best();                        // training_work_dir()\vsr_best.pt exists
// export.py on a copy of vsr_best.pt, then installed as user_data_dir()\vsr.onnx/vsr.ref (the old pair kept as
// vsr.prev.onnx/vsr.prev.ref)
bool export_and_install(std::string& error);
// Copies an exported vsr_best.onnx/.ref from work into data_dir as vsr.onnx/vsr.ref: the old pair becomes
// vsr.prev.*, and a failure on the way puts it back.
bool install_model(const std::string& work, const std::string& data_dir, std::string& error);
bool start_over(std::string& error);  // deletes training_work_dir(); refused while the run is going
int training_run_selftest();          // part of --uitest; 0 on pass
