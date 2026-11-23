#pragma once
// Training from the player: train.py on the Training add-on's Python, in its own process (no window), so closing
// the player doesn't stop it. Stop is a file (training.stop next to training.json), so it works from any later
// player too. When a run is done, export.py turns its best checkpoint into the Pixel Flow the player loads.
#include <string>

#include "settings.h"

enum class StartMode { New, Resume, More };  // More: another 100000 steps on a finished model

// "<python>" -u "<scripts>\train.py" "<videos>" --out "<work>\vsr.pt" [--resume "<work>\vsr.pt" [--more 100000]]
std::wstring train_command(const std::string& python, const std::string& scripts, const std::string& videos,
                           const std::string& work, StartMode m);
// Starts train.py on videos in training_work_dir(); its output goes to train.log there.
bool start_training(const std::string& videos, StartMode m, std::string& error);
bool training_child_exited(int& code);  // the run this player started has ended (false: running, or none)
void request_stop();                    // writes training.stop; train.py saves and exits at its next step
std::string stop_file();                // training.stop, next to training.json
bool own_run(const TrainingStatus& t);  // t is the player's own run (its out is training_work_dir()\vsr.pt)
std::string train_log();                // training_work_dir()\train.log
bool has_checkpoint();                  // training_work_dir()\vsr.pt exists
bool has_best();                        // training_work_dir()\vsr_best.pt exists
// export.py on vsr_best.pt, then vsr_best.onnx/.ref installed as user_data_dir()\vsr.onnx/vsr.ref
bool export_and_install(std::string& error);
// Copies an exported vsr_best.onnx/.ref from work into data_dir as vsr.onnx/vsr.ref (temp + rename).
bool install_model(const std::string& work, const std::string& data_dir, std::string& error);
bool start_over(std::string& error);  // deletes training_work_dir(); refuses while this player's run is going
int training_run_selftest();          // part of --uitest; 0 on pass
