#!/usr/bin/env bash
# Playback smoke test. Needs a built player, ffmpeg, and a display (desktop or WSLg).
#   bash tests/smoke.sh              playback checks
#   bash tests/smoke.sh vsr.onnx     also the model self-test
set -u
cd "$(dirname "$0")/.."
P=./build/open-player
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0
check() { if [ "$2" = yes ]; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
# furthest playback position the player reports (--stats "t") within $2 seconds
reached() { timeout -s INT "$2" "$P" "$1" --no-vsr --stats 2>/dev/null | grep -o '"t":[0-9.]*' | cut -d: -f2 | sort -n | tail -1; }
at_least() { awk -v a="${1:-0}" -v b="$2" 'BEGIN { print (a >= b) ? "yes" : "no" }'; }

# 6 s of video whose audio stops at 3 s: playback used to freeze where the audio ended
ffmpeg -v error -f lavfi -i testsrc2=size=640x360:rate=30 -f lavfi -t 3 -i sine=frequency=440 -t 6 \
  -c:v libx264 -pix_fmt yuv420p -c:a aac "$T/short_audio.mp4"
t=$(reached "$T/short_audio.mp4" 12)
check "plays past where the audio ends (reached ${t:-nothing} of 6 s)" "$(at_least "$t" 5.5)"

# raw H.264 stream: frames carry no timestamps and used to be dropped, leaving a black window
ffmpeg -v error -f lavfi -i testsrc2=size=640x360:rate=25 -t 4 -c:v libx264 -pix_fmt yuv420p -f h264 "$T/raw.h264"
t=$(reached "$T/raw.h264" 10)
check "plays a stream without timestamps (reached ${t:-nothing} of 4 s)" "$(at_least "$t" 3.5)"

if [ $# -ge 1 ]; then
  "$P" --selftest "$1" >/dev/null 2>&1 && ok=yes || ok=no
  check "model self-test ($1)" "$ok"
fi
exit $fail
