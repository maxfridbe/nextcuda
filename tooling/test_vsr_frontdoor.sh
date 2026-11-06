#!/usr/bin/env bash
# tooling/test_vsr_frontdoor.sh FILM.mp4 [MODEL] [LONG_SECONDS] [SHORT_SECONDS]: video.upscale through the front door the
# way the creative platform sends it - the clip inline as a data: URL, wait false, a reentrant_id, then image.status
# and a GET of the result's url - for two clips cut from FILM (h264 + aac, the moov atom at the end, as the platform's
# films are), submitted 0.5 s apart so they land on the same server together. Each must finish with every frame of its
# own clip, at twice the size, its file served. (Two jobs within a second once shared their files: one decoded the
# other's clip and the other none, 2026-11-05.) Run on the GPU host; it uses the GPUs through nextcuda's own queue.
set -uo pipefail
FILM=${1:?usage: test_vsr_frontdoor.sh FILM.mp4 [model] [long seconds] [short seconds]}
MODEL=${2:-flashvsr}
LONG=${3:-120}
SHORT=${4:-20}
URL=${NEXTCUDA_URL:-http://127.0.0.1:8000}
TOKEN=${NEXTCUDA_TOKEN:-$(python3 -c "import json;print(json.load(open('$HOME/nextcuda/serve.json'))['token'])")}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fail=0
rpc() { curl -s -X POST "$URL/rpc/$1" -H "Authorization: Bearer $TOKEN" --data-binary "@$2"; }
field() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }

for c in long short; do
  secs=$([ $c = long ] && echo "$LONG" || echo "$SHORT")
  start=$([ $c = long ] && echo 30 || echo 200)
  ffmpeg -v error -y -ss "$start" -i "$FILM" -t "$secs" -c:v libx264 -crf 18 -c:a aac "$W/$c.mp4" || { echo "FAIL cutting $c"; exit 1; }
  n=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$W/$c.mp4")
  echo "$n" > "$W/$c.frames"
  id="vsrtest-$c-$(date +%s)-$$"
  echo "$id" > "$W/$c.id"
  { printf '{"model":"%s","variant":"flashvsr-v1.1","scale":2,"crf":16,"wait":false,"label":"%s","reentrant_id":"%s","title":"front-door test (%s, %s frames)","video":"data:video/mp4;base64,' \
      "$MODEL" "$id" "$id" "$c" "$n"; base64 -w0 "$W/$c.mp4"; printf '"}'; } > "$W/$c.json"
  echo "$c: $n frames, $(du -h "$W/$c.json" | cut -f1) of request"
done
for c in long short; do
  rpc video.upscale "$W/$c.json" | head -c 300; echo
  sleep 0.5
done
for c in long short; do
  id=$(cat "$W/$c.id")
  printf '{"reentrant_id":"%s"}' "$id" > "$W/$c.q"
  for _ in $(seq 1 1440); do
    r=$(rpc image.status "$W/$c.q")
    st=$(echo "$r" | field "d.get('result',{}).get('state')")
    [ "$st" = done ] || [ "$st" = failed ] || [ "$st" = cancelled ] && break
    sleep 5
  done
  want=$(cat "$W/$c.frames")
  if [ "$st" != done ]; then
    echo "FAIL $c: $st: $(echo "$r" | field "d['result'].get('error')")"; fail=1; continue
  fi
  got=$(echo "$r" | field "d['result']['result']['data'][0]['frames']")
  link=$(echo "$r" | field "d['result']['result']['data'][0]['url']")
  code=$(curl -s -o "$W/$c.out.mp4" -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$URL/v1/images/files/${link##*/}")
  have=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=width,height,nb_read_packets -of csv=p=0 "$W/$c.out.mp4" 2>/dev/null)
  echo "$c: frames $got of $want, GET $code, file $have"
  [ "$got" = "$want" ] && [ "$code" = 200 ] && [[ "$have" == *",$want" ]] || { echo "FAIL $c"; fail=1; }
done
[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
