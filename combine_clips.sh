#!/bin/bash
# OK technically a shell script, not a Python script.
# Simple script to combine video clips into one.
# by Kevin Walker Sep 2026, created for https://the-making-of-creativity.com/
# 
# Drop this script into a folder of video clips and run it — it finds every video
# in that folder, sorts them in numerical/natural order, and joins them into one
# portrait video, normalizing resolution/codec/framerate/rotation along the way.
#
# Usage: ./combine_clips.sh [output.mp4]
#   output.mp4 defaults to "combined_output.mp4", written into the same folder
#   as this script (not the current working directory).
#
# Env: MAX_SIZE_MB=30 (default) targets a final file size via two-pass bitrate encoding.
#      Set MAX_SIZE_MB=0 to disable and use quality-based CRF 18 encoding instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_NAME="${1:-combined_output.mp4}"
case "$OUT_NAME" in
  /*) OUT="$OUT_NAME" ;;
  *) OUT="${SCRIPT_DIR}/${OUT_NAME}" ;;
esac

# Collect video files from the script's folder, case-insensitive extensions, natural order.
shopt -s nullglob nocaseglob
CANDIDATES=("$SCRIPT_DIR"/*.mp4 "$SCRIPT_DIR"/*.mov "$SCRIPT_DIR"/*.m4v "$SCRIPT_DIR"/*.avi "$SCRIPT_DIR"/*.mkv)
shopt -u nullglob nocaseglob

INPUTS=()
for f in "${CANDIDATES[@]}"; do
  [ "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" = "$OUT" ] && continue
  INPUTS+=("$f")
done
IFS=$'\n' INPUTS=($(printf '%s\n' "${INPUTS[@]}" | sort -V))
unset IFS

N=${#INPUTS[@]}
if [ "$N" -lt 2 ]; then
  echo "Found ${N} video(s) in ${SCRIPT_DIR} — need at least 2 to combine." >&2
  exit 1
fi

echo "Joining ${N} clips in order:"
printf '  %s\n' "${INPUTS[@]##*/}"

MAX_SIZE_MB="${MAX_SIZE_MB:-30}"

# Target portrait canvas (1080x1920 = vertical HD). Bump to 2160x3840 for full 4K.
W=1080
H=1920
FPS=30
AUDIO_BITRATE=128000

INPUT_ARGS=()
for f in "${INPUTS[@]}"; do
  INPUT_ARGS+=(-i "$f")
done

FILTER=""
VLABELS=""
for i in "${!INPUTS[@]}"; do
  FILTER+="[$i:v]scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=${FPS}[v$i];"
  FILTER+="[$i:a]aresample=44100,aformat=channel_layouts=stereo[a$i];"
  VLABELS+="[v$i][a$i]"
done
FILTER+="${VLABELS}concat=n=${N}:v=1:a=1[outv][outa]"

if [ "$MAX_SIZE_MB" = "0" ]; then
  ffmpeg -y "${INPUT_ARGS[@]}" -filter_complex "$FILTER" \
    -map "[outv]" -map "[outa]" \
    -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
    -c:a aac -b:a 192k \
    "$OUT"
else
  # Sum input durations to size a bitrate that lands the whole file under MAX_SIZE_MB.
  TOTAL_DURATION=0
  for f in "${INPUTS[@]}"; do
    D=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f")
    TOTAL_DURATION=$(echo "$TOTAL_DURATION + $D" | bc)
  done

  # 5% headroom for container/muxing overhead.
  TARGET_TOTAL_BITS=$(echo "$MAX_SIZE_MB * 8 * 1024 * 1024 * 0.95" | bc)
  TARGET_TOTAL_BPS=$(echo "$TARGET_TOTAL_BITS / $TOTAL_DURATION" | bc)
  VIDEO_BITRATE=$(echo "$TARGET_TOTAL_BPS - $AUDIO_BITRATE" | bc | cut -d. -f1)

  if [ "$VIDEO_BITRATE" -lt 100000 ]; then
    echo "Warning: computed video bitrate (${VIDEO_BITRATE} bps) is very low for ${TOTAL_DURATION}s of footage at ${MAX_SIZE_MB}MB. Quality will suffer." >&2
  fi

  PASSDIR=$(mktemp -d)
  PASSLOG="${PASSDIR}/ffmpeg2pass"

  ffmpeg -y "${INPUT_ARGS[@]}" -filter_complex "$FILTER" \
    -map "[outv]" -map "[outa]" \
    -c:v libx264 -b:v "${VIDEO_BITRATE}" -preset medium -pix_fmt yuv420p \
    -c:a aac -b:a "$AUDIO_BITRATE" \
    -pass 1 -passlogfile "$PASSLOG" \
    -f mp4 /dev/null

  ffmpeg -y "${INPUT_ARGS[@]}" -filter_complex "$FILTER" \
    -map "[outv]" -map "[outa]" \
    -c:v libx264 -b:v "${VIDEO_BITRATE}" -maxrate "$((VIDEO_BITRATE * 3 / 2))" -bufsize "$((VIDEO_BITRATE * 2))" \
    -preset medium -pix_fmt yuv420p \
    -pass 2 -passlogfile "$PASSLOG" \
    -c:a aac -b:a "$AUDIO_BITRATE" \
    "$OUT"

  rm -rf "$PASSDIR"

  ACTUAL_MB=$(echo "scale=1; $(stat -f%z "$OUT") / 1024 / 1024" | bc)
  echo "Target size: ${MAX_SIZE_MB}MB, actual: ${ACTUAL_MB}MB"
fi

echo "Done: $OUT"
