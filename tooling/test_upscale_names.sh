#!/usr/bin/env bash
# tooling/test_upscale_names.sh FILM.mp4 [MODEL] [GPU]: two image servers of the same model (two processes, as two
# cards' servers or a reload leave them) sharing one output folder, each sent a different clip inline in the same second
# (video.upscale, the platform's way). Each answer must be its own clip, whole, in a file of its own - no shared input,
# no result overwritten or deleted by the other. The program is the one beside this script's ../dist (NEXTCUDA_BIN to
# choose another); GPU work under ~/.nextcuda-gpu.lock. Default model real-esrgan (small: the naming is the server's,
# not the engine's).
set -uo pipefail
FILM=${1:?usage: test_upscale_names.sh FILM.mp4 [model] [gpu]}
MODEL=${2:-real-esrgan}
GPU=${3:-1}
BIN=${NEXTCUDA_BIN:-$(cd "$(dirname "$0")/.." && pwd)/dist/nextcuda}
W=$(mktemp -d)
OUT=$W/out
mkdir -p "$OUT"
P1=$((18600 + RANDOM % 200)); P2=$((P1 + 1))
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; rm -rf "$W"; }
trap cleanup EXIT
exec 9>"$HOME/.nextcuda-gpu.lock"
flock 9
for p in $P1 $P2; do
  "$BIN" image serve "$MODEL" --here --port "$p" --gpu "$GPU" --out "$OUT" > "$W/serve-$p.log" 2>&1 &
  pids+=($!)
done
for p in $P1 $P2; do
  for _ in $(seq 1 120); do curl -s -m 2 "localhost:$p/health" > /dev/null && break; sleep 1; done
done
# two different clips: 2 s from two places of the film
for i in 1 2; do
  ffmpeg -v error -y -ss $((30 * i)) -i "$FILM" -t 2 -c:v libx264 -crf 18 -c:a aac "$W/c$i.mp4"
  { printf '{"model":"%s","scale":2,"variant":"realesr-animevideov3","crf":16,"video":"data:video/mp4;base64,' "$MODEL"; base64 -w0 "$W/c$i.mp4"; printf '"}'; } > "$W/r$i.json"
done
# the same second, both servers
curl -s -X POST "localhost:$P1/v1/videos/upscale" --data-binary "@$W/r1.json" > "$W/a1.json" &
c1=$!
curl -s -X POST "localhost:$P2/v1/videos/upscale" --data-binary "@$W/r2.json" > "$W/a2.json" &
c2=$!
wait $c1 $c2
fail=0
for i in 1 2; do
  name=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['data'][0]['name'])" "$W/a$i.json" 2>/dev/null)
  want=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$W/c$i.mp4")
  have=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$OUT/$name" 2>/dev/null)
  echo "clip $i: $name, $have of $want frames"
  [ -n "$name" ] && [ "$have" = "$want" ] || { echo "FAIL clip $i: $(head -c 300 "$W/a$i.json")"; fail=1; }
  eval "n$i=\$name"
done
[ "${n1:-x}" != "${n2:-x}" ] || { echo "FAIL: one file for both"; fail=1; }
# each result is its own clip: the first frames differ (the clips are 30 s apart)
if [ $fail = 0 ]; then
  d=$(ffmpeg -i "$OUT/$n1" -i "$OUT/$n2" -frames:v 1 -lavfi psnr -f null - 2>&1 | grep -o "average:[0-9.inf]*")
  echo "first frames, result 1 against result 2: PSNR $d (low: different clips)"
  case "$d" in *inf*) echo "FAIL: the two results are the same clip"; fail=1 ;; esac
fi
ls "$OUT"
[ $fail = 0 ] && echo PASS || echo FAIL
exit $fail
