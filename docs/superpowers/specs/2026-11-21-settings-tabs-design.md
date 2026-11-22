# Settings tabs, resume position, training status

Date: 2026-11-21. Status: approved by the user (short design, in chat).

## Intent

The user asked for the settings card to have tabs (General, Training status, About), for the player to remember
where they left off in each video (a switch in General), and for a tab showing a running `train.py` live.

## Design

- **Tabs on the left side of the card** (the user asked for the side, not the top): a column of rows, each an
  icon and a label, the selected row filled violet like the segments, hover in the card's hover colour.
  General (Upscaling, Decoding, Display, Playback & controls + "Remember where I left off"), Training, About
  (build, versions, updates, log, licences). The card grows about 130 px wider for the column; when the window is
  too narrow the column shows icons only. The last tab is remembered (`settings_tab=`).
- **Resume:** `positions.txt` next to `settings.txt`: `<seconds>\t<absolute path>` per line, newest first, at most
  200 videos. Saved every 10 s while playing, on quit, and before any open/reopen. On open, when the switch is on
  (`remember_position=1`, default) and the saved time is at least 10 s in and more than 30 s before the end, the
  player seeks there with a notice "Resumed at 12:34". `--start` wins over it.
- **Training status:** `train.py` writes `training.json` next to `settings.txt` (same folder rules as the player:
  `XDG_CONFIG_HOME`, else `%APPDATA%\open-player`, else `~/.config/open-player`) every 100 steps and after each
  validation, atomically (temp + replace): `{"state": "training"|"finished"|"stopped", "step", "steps", "loss",
  "val_db", "best_db", "s_per_step", "eta_s", "suggest", "out", "updated": unix time}`. The Training tab reads it
  once a second while open and shows state, a progress bar, loss, validation (best), speed, time left, the latest
  `suggest:` line and the output path; "not running" when `updated` is over 5 minutes old and the state is still
  "training"; "No training yet" when there is no file.

## Approaches considered (training status)

1. `train.py` writes `training.json` in the player's settings folder (chosen): no setup, one small writer.
2. The player reads `<out>.diag.jsonl`: needs the user to point the player at their training folder.
3. A local socket/HTTP feed from `train.py`: more code and a listening port for no gain.

## Non-goals

Starting or stopping training from the player (panel.py does that); several runs at once (the last writer wins).

## Left to the builder

Exact layout of the Training tab; tab widths; wording.
