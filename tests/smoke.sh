#!/usr/bin/env bash
# Playback smoke test. Needs a built player and ffmpeg. Runs offscreen and silent; to watch it instead:
#   SDL_VIDEO_DRIVER=wayland SDL_AUDIO_DRIVER=pipewire bash tests/smoke.sh
#   bash tests/smoke.sh              playback checks
#   bash tests/smoke.sh vsr.onnx     also the model self-test and playback with VSR on
set -u
cd "$(dirname "$0")/.."
export SDL_VIDEO_DRIVER=${SDL_VIDEO_DRIVER:-offscreen} SDL_AUDIO_DRIVER=${SDL_AUDIO_DRIVER:-dummy}
P=./build/open-player
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0
check() { if [ "$2" = yes ]; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
# furthest playback position (--stats "t") within $2 seconds; stops at the end of the clip, where the player
# pauses on the last frame. Further arguments go to the player.
reached() {
  timeout -s INT "$2" "$P" "$1" "${@:3}" --stats 2>/dev/null |
    awk -F'"t":' '/^stats/ { split($2, a, ","); t = a[1] } /"paused":true/ { exit } END { print t + 0 }'
}
at_least() { awk -v a="${1:-0}" -v b="$2" 'BEGIN { print (a >= b) ? "yes" : "no" }'; }

# 6 s of video whose audio stops at 3 s: playback used to freeze where the audio ended
ffmpeg -v error -f lavfi -i testsrc2=size=640x360:rate=30 -f lavfi -t 3 -i sine=frequency=440 -t 6 \
  -c:v libx264 -pix_fmt yuv420p -c:a aac "$T/short_audio.mp4"
t=$(reached "$T/short_audio.mp4" 12 --no-vsr)
check "plays past where the audio ends (reached ${t:-nothing} of 6 s)" "$(at_least "$t" 5.5)"

# raw H.264 stream: frames carry no timestamps and used to be dropped, leaving a black window
ffmpeg -v error -f lavfi -i testsrc2=size=640x360:rate=25 -t 4 -c:v libx264 -pix_fmt yuv420p -f h264 "$T/raw.h264"
t=$(reached "$T/raw.h264" 10 --no-vsr)
check "plays a stream without timestamps (reached ${t:-nothing} of 4 s)" "$(at_least "$t" 3.5)"

if [ $# -ge 1 ]; then
  "$P" --selftest "$1" >/dev/null 2>&1 && ok=yes || ok=no
  check "model self-test ($1)" "$ok"
  # VSR on: network, render-ahead and shrink-to-screen (the offscreen screen is 1024x768). The first run may
  # spend up to a minute building the engine for this size.
  t=$(reached "$T/short_audio.mp4" 120 --model "$1")
  check "plays with VSR on (reached ${t:-nothing} of 6 s)" "$(at_least "$t" 5.5)"
fi
exit $fail
