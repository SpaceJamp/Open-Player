#!/usr/bin/env bash
# Playback smoke test. Needs a built player and ffmpeg. Runs offscreen and silent; to watch it instead:
#   SDL_VIDEO_DRIVER=wayland SDL_AUDIO_DRIVER=pipewire bash tests/smoke.sh
#   bash tests/smoke.sh              playback checks
#   bash tests/smoke.sh vsr.onnx     also the model self-test and playback with VSR on
set -u
cd "$(dirname "$0")/.."
export SDL_VIDEO_DRIVER=${SDL_VIDEO_DRIVER:-offscreen} SDL_AUDIO_DRIVER=${SDL_AUDIO_DRIVER:-dummy}
SIG=INT SCREEN=()  # Windows: NVIDIA's Vulkan driver can't render offscreen, so tests play on a monitor (default: the 2nd)
# (and Git Bash can't send Ctrl-C to a native program, so it is killed instead)
case "$(uname -s)" in MINGW*|MSYS*) unset SDL_VIDEO_DRIVER; SIG=KILL; SCREEN=(--screen "${OPEN_PLAYER_SCREEN:-2}") ;; esac
P=${OPEN_PLAYER:-./build/open-player}  # Windows: OPEN_PLAYER=build-win/open-player.exe
T=$(mktemp -d)
export XDG_CONFIG_HOME="$T"  # the player records recent files; keep test clips out of the real list
trap 'rm -rf "$T"' EXIT
fail=0
check() { if [ "$2" = yes ]; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
# furthest playback position (--stats "t") within $2 seconds; stops at the end of the clip, where the player
# pauses on the last frame. Further arguments go to the player.
reached() {
  timeout -s "$SIG" "$2" "$P" "$1" "${@:3}" "${SCREEN[@]}" --stats 2>/dev/null |
    awk -F'"t":' '/^stats/ { split($2, a, ","); t = a[1] } /"paused":true/ { exit } END { print t + 0 }'
}
at_least() { awk -v a="${1:-0}" -v b="$2" 'BEGIN { print (a >= b) ? "yes" : "no" }'; }

# 6 s of video whose audio stops at 3 s: playback used to freeze where the audio ended
ffmpeg -v error -f lavfi -i testsrc2=size=640x360:rate=30 -f lavfi -t 3 -i sine=frequency=440 -t 6 \
  -c:v libx264 -pix_fmt yuv420p -c:a aac "$T/short_audio.mp4"
t=$(reached "$T/short_audio.mp4" 12 --no-vsr)
check "plays past where the audio ends (reached ${t:-nothing} of 6 s)" "$(at_least "$t" 5.5)"

# --stats reports what the on-screen controls cost
timeout -s "$SIG" 4 "$P" "$T/short_audio.mp4" --no-vsr "${SCREEN[@]}" --stats 2>/dev/null | grep -q '"ui_ms":[0-9.]*,"ui_max_ms":[0-9.]*,"ui_vtx"' && ok=yes || ok=no
check "stats include UI timings" "$ok"

# raw H.264 stream: frames carry no timestamps and used to be dropped, leaving a black window
ffmpeg -v error -f lavfi -i testsrc2=size=640x360:rate=25 -t 4 -c:v libx264 -pix_fmt yuv420p -f h264 "$T/raw.h264"
t=$(reached "$T/raw.h264" 10 --no-vsr)
check "plays a stream without timestamps (reached ${t:-nothing} of 4 s)" "$(at_least "$t" 3.5)"

"$P" --uitest >/dev/null 2>&1 && ok=yes || ok=no
check "UI logic self-test" "$ok"

if [ $# -ge 1 ]; then
  "$P" --selftest "$1" >/dev/null 2>&1 && ok=yes || ok=no
  check "model self-test ($1)" "$ok"
  # VSR on: network, render-ahead and shrink-to-screen (the offscreen screen is 1024x768). The first run may
  # spend up to a minute building the engine for this size.
  t=$(reached "$T/short_audio.mp4" 120 --model "$1")
  check "plays with VSR on (reached ${t:-nothing} of 6 s)" "$(at_least "$t" 5.5)"
fi
exit $fail
