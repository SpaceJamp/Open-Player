#!/usr/bin/env bash
# Codec matrix: encodes a 3 s clip in each format this machine's ffmpeg can write, plays it, and reports
# how it was decoded and whether playback reached the end with sound. Runs offscreen and silent; to watch:
#   SDL_VIDEO_DRIVER=wayland SDL_AUDIO_DRIVER=pipewire bash tests/codecs.sh
set -u
cd "$(dirname "$0")/.."
export SDL_VIDEO_DRIVER=${SDL_VIDEO_DRIVER:-offscreen} SDL_AUDIO_DRIVER=${SDL_AUDIO_DRIVER:-dummy}
P=./build/open-player
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
SRC=(-f lavfi -i testsrc2=size=640x360:rate=25 -f lavfi -i "sine=frequency=440:sample_rate=48000" -t 3)
fail=0

# name | file extension | video encoder args | audio encoder args
cases=(
  "H.264|mp4|-c:v libx264 -pix_fmt yuv420p|-c:a aac"
  "H.264 10-bit|mkv|-c:v libx264 -pix_fmt yuv420p10le|-c:a libopus"
  "H.264 4:2:2 10-bit|mkv|-c:v libx264 -pix_fmt yuv422p10le|-c:a flac"
  "H.264 4:4:4|mkv|-c:v libx264 -pix_fmt yuv444p|-c:a libvorbis"
  "HEVC|mp4|-c:v libx265 -pix_fmt yuv420p -x265-params log-level=none|-c:a aac"
  "HEVC 10-bit|mkv|-c:v libx265 -pix_fmt yuv420p10le -x265-params log-level=none|-c:a eac3"
  "HEVC 4:2:2 10-bit|mkv|-c:v libx265 -pix_fmt yuv422p10le -x265-params log-level=none|-c:a ac3 -ac 6"
  "AV1|webm|-c:v libsvtav1 -pix_fmt yuv420p|-c:a libopus"
  "AV1 10-bit|mkv|-c:v libsvtav1 -pix_fmt yuv420p10le|-c:a libopus"
  # NVDEC can't do AV1 4:4:4: the player must notice and switch to the CPU decoder by itself
  "AV1 4:4:4|mkv|-c:v libaom-av1 -cpu-used 8 -pix_fmt yuv444p|-c:a libopus"
  "VP9|webm|-c:v libvpx-vp9 -pix_fmt yuv420p|-c:a libopus"
  "VP9 10-bit|webm|-c:v libvpx-vp9 -pix_fmt yuv420p10le|-c:a libvorbis"
  "VP8|webm|-c:v libvpx -pix_fmt yuv420p|-c:a libvorbis"
  "MPEG-2|ts|-c:v mpeg2video -pix_fmt yuv420p|-c:a mp2"
  "MPEG-2 interlaced|ts|-c:v mpeg2video -pix_fmt yuv420p -flags +ilme+ildct -field_order tt|-c:a ac3"
  "MPEG-4 Part 2|avi|-c:v mpeg4 -pix_fmt yuv420p|-c:a libmp3lame"
  "MJPEG (full range)|avi|-c:v mjpeg -pix_fmt yuvj422p|-c:a pcm_s16le"
  "ProRes 4:2:2 10-bit|mov|-c:v prores_ks -pix_fmt yuv422p10le|-c:a pcm_s24le"
  "DNxHR|mov|-c:v dnxhd -profile:v dnxhr_sq -pix_fmt yuv422p|-c:a pcm_s16le"
  "FFV1 lossless|mkv|-c:v ffv1 -pix_fmt yuv420p|-c:a flac"
  "Theora|ogv|-c:v libtheora -pix_fmt yuv420p|-c:a libvorbis"
  "WMV2|wmv|-c:v wmv2 -pix_fmt yuv420p|-c:a wmav2"
  "RGB (UT Video)|avi|-c:v utvideo -pix_fmt gbrp|-c:a pcm_s16le"
  # ALAC in MP4 stores audio in big chunks far apart; playback used to starve the audio and crawl
  "ALAC audio, MP4 chunks|mp4|-c:v libx264 -pix_fmt yuv420p|-c:a alac"
  "MP3 44.1 kHz mono|mkv|-c:v libx264 -pix_fmt yuv420p|-c:a libmp3lame -ar 44100 -ac 1"
  "FLAC 96 kHz|mkv|-c:v libx264 -pix_fmt yuv420p|-c:a flac -ar 96000"
  "7.1 audio|mkv|-c:v libx264 -pix_fmt yuv420p|-c:a flac -ac 8"
)

printf "%-22s %-34s %s\n" "FORMAT" "DECODED AS" "RESULT"
for c in "${cases[@]}"; do
  IFS='|' read -r name ext vargs aargs <<< "$c"
  f="$T/clip.$ext"
  # shellcheck disable=SC2086 # the encoder args are meant to split into words
  if ! ffmpeg -v error -y "${SRC[@]}" $vargs $aargs "$f" 2>"$T/err"; then
    printf "%-22s %-34s %s\n" "$name" "-" "SKIP (ffmpeg can't encode it: $(head -c 60 "$T/err" | tr '\n' ' '))"
    continue
  fi
  out=$(timeout -s INT 9 "$P" "$f" --no-vsr --stats 2>&1 | awk '{ print } /"paused":true/ { exit }')  # stop at the end
  decoded=$(grep -o '"decode":"[^"]*"' <<< "$out" | head -1 | cut -d'"' -f4)
  t=$(grep -o '"t":[0-9.]*' <<< "$out" | cut -d: -f2 | sort -n | tail -1)
  a=$(grep -o '"audio_s":[0-9.]*' <<< "$out" | cut -d: -f2 | sort -n | tail -1)  # audio buffered for the device
  if ! awk -v t="${t:-0}" 'BEGIN { exit !(t >= 2.5) }'; then r="FAIL (reached ${t:-nothing} of 3 s)"; fail=1
  elif ! awk -v a="${a:-0}" 'BEGIN { exit !(a > 0) }'; then r="FAIL (no audio reached the device)"; fail=1
  else r="PASS"; fi
  printf "%-22s %-34s %s\n" "$name" "${decoded:-?}" "$r"
done
exit $fail
