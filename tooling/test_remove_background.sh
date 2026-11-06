#!/usr/bin/env bash
# tooling/test_remove_background.sh PICTURE.png [REF_MASK_PP.png]: image.remove_background through the front door as the
# platform sends it - the picture inline as a data: URL, wait false, a reentrant_id, then image.status - for birefnet and
# isnet-general-use (mask: true, post_process_mask: true). Each must answer an RGBA PNG and a grey mask at the picture's
# size; with REF_MASK_PP (rembg's post-processed mask of the same picture: reference/rembg/ref.py's mask_pp.png) the
# isnet mask must equal it in at least 99.9% of its pixels.
set -uo pipefail
PIC=${1:?usage: test_remove_background.sh PICTURE.png [REF_MASK_PP.png]}
REF=${2:-}
URL=${NEXTCUDA_URL:-http://127.0.0.1:8000}
TOKEN=${NEXTCUDA_TOKEN:-$(python3 -c "import json;print(json.load(open('$HOME/nextcuda/serve.json'))['token'])")}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fail=0
size=$(ffprobe -v error -show_entries stream=width,height -of csv=p=0 "$PIC")
rpc() { curl -s -X POST "$URL/rpc/$1" -H "Authorization: Bearer $TOKEN" --data-binary "@$2"; }
field() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }
for model in birefnet isnet-general-use; do
  id="rbtest-$model-$(date +%s)-$$"
  { printf '{"model":"%s","mask":true,"post_process_mask":%s,"wait":false,"label":"%s","reentrant_id":"%s","image":"data:image/png;base64,' \
      "$model" "$([ $model = isnet-general-use ] && echo true || echo false)" "$id" "$id"; base64 -w0 "$PIC"; printf '"}'; } > "$W/req.json"
  rpc image.remove_background "$W/req.json" | head -c 200; echo
  printf '{"reentrant_id":"%s"}' "$id" > "$W/q.json"
  for _ in $(seq 1 120); do
    r=$(rpc image.status "$W/q.json")
    st=$(echo "$r" | field "d.get('result',{}).get('state')")
    [ "$st" = done ] || [ "$st" = failed ] || [ "$st" = cancelled ] && break
    sleep 2
  done
  if [ "$st" != done ]; then echo "FAIL $model: $st $(echo "$r" | field "d['result'].get('error')")"; fail=1; continue; fi
  echo "$r" | field "d['result']['result']['data'][0]['b64_json']" | base64 -d > "$W/$model.png"
  echo "$r" | field "d['result']['result']['data'][0]['mask_b64_json']" | base64 -d > "$W/$model-mask.png"
  secs=$(echo "$r" | field "d['result']['result']['nextsycl']['seconds']")
  cut=$(ffprobe -v error -show_entries stream=width,height,pix_fmt -of csv=p=0 "$W/$model.png")
  msk=$(ffprobe -v error -show_entries stream=width,height,pix_fmt -of csv=p=0 "$W/$model-mask.png")
  echo "$model: cut-out $cut, mask $msk, $secs s (picture $size)"
  [[ "$cut" == "$size,rgba" ]] && [[ "$msk" == "$size,gray" ]] || { echo "FAIL $model: shapes"; fail=1; }
  if [ $model = isnet-general-use ] && [ -n "$REF" ]; then
    ffmpeg -v error -i "$W/$model-mask.png" -f rawvideo -pix_fmt gray "$W/a.raw"
    ffmpeg -v error -i "$REF" -f rawvideo -pix_fmt gray "$W/b.raw"
    n=$(stat -c %s "$W/b.raw"); d=$(cmp -l "$W/a.raw" "$W/b.raw" | wc -l)
    pct=$(python3 -c "print(f'{100 * (1 - $d / $n):.4f}')")
    echo "isnet mask against rembg's: $pct% equal ($d of $n pixels differ)"
    python3 -c "import sys; sys.exit(0 if $d / $n <= 0.001 else 1)" || { echo "FAIL: isnet's mask is not rembg's"; fail=1; }
  fi
done
[ $fail = 0 ] && echo PASS || echo FAIL
exit $fail
