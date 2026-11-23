#!/usr/bin/env bash
# tooling/gate.sh <base> <head> [ENGINE...]: the merge gate (CONTRIBUTING.md "Pull requests"; nextsycl's, on CUDA). From
# the paths that changed between two commits it works out the engines affected - an engine's own crate, kernels or silo:
# that engine; a silo variant: that engine on the cards the variant matches (`nextcuda silo ARCH`); a shared part: the
# engines tooling/gate-baselines.json's "shared" maps it to - builds <head> in a build copy (never the service's files:
# the program in the trixie toolchain, the kernels where nvcc is), runs each engine's quick run on each of its GPUs (or
# GPU sets) twice - under the GPU lock (~/.nextcuda-gpu.lock) and a timeout, and only while the box is idle: no other
# process on those GPUs and nothing running or queued for the creative platform - and compares with the baselines: the
# exact output exactly (an LLM's greedy tokens and their log-probabilities, a picture's, a sound's or a mesh's bytes, a
# clip's decoded frames - for a fixed seed), each throughput within the tolerance (3%, the better of the two runs). An
# engine whose output differs between two runs of the same build (`nondeterministic` in its baselines, with why) is held
# to its throughput alone.
# Prints a pass/fail table to paste into the pull request; exits 1 on a failure.
#
# The baselines are per card, keyed by the card's id (what `nextcuda silo` prints: rtx-3060-28u-12g - its name, its
# SM count and its memory; a run over several GPUs: their ids joined by +), with what the card reports (`device`) and
# the box (`host`), so a contributor gates on hardware the project does not own: run it on your own box (NS_GATE_HOST
# empty or `local`: no ssh), record your card's baseline on the base commit (NS_GATE_RECORD=1) and commit it in the
# same pull request, then compare the head against it and paste the table.
#
#   NS_GATE_HOST=brainiac-nvidia tooling/gate.sh HEAD~1 HEAD           # what the last commit touched, on the project's box
#   NS_GATE_HOST=brainiac-nvidia tooling/gate.sh HEAD HEAD image/mossimage   # one engine, by name
#   NS_GATE_HOST=brainiac-nvidia NS_GATE_RECORD=1 tooling/gate.sh HEAD HEAD all   # every engine's baselines
#   NS_GATE_GPUS=0 NS_GATE_PATHS=/mnt/hdd/nextcuda-models=$HOME/models NS_GATE_RECORD=1 \
#     tooling/gate.sh origin/main origin/main image/mossimage          # your card's baseline, on your box
#
# NS_GATE_HOST: the GPU box over ssh (empty or `local`: this machine); NS_GATE_DIR: the build copy there (default
# nextcuda-gate under ~ over ssh, ~/.cache/nextcuda-gate here) - its own dist, nextcuda.conf and a copy of the
# registry, never the service's; NS_GATE_LOCK: the lock file every GPU run takes (default ~/.nextcuda-gpu.lock; `none`:
# no lock); NS_GATE_GPUS: the GPUs (or sets, "0,1") to run on, space separated (default: each engine's `gpus`, the
# project box's); NS_GATE_PATHS: FROM=TO[,FROM=TO] - the baselines' model paths (in `run`) where they are on this box;
# NS_GATE_RECORD=1: record instead of compare; NS_GATE_WAIT: seconds to wait for the box to be idle (default 7200);
# NS_GATE_SKIP_BUILD=1: run what is built there already.
set -euo pipefail
set -f   # no file-name globbing: the baselines' engine patterns ("image/*") are matched, not expanded
cd "$(dirname "$0")/.."
BASE=${1:?gate.sh <base> <head> [ENGINE...]}; HEAD=${2:?gate.sh <base> <head> [ENGINE...]}; shift 2
HOST=${NS_GATE_HOST:-}
[ "$HOST" = local ] && HOST=
if [ -n "$HOST" ]; then DIR=${NS_GATE_DIR:-nextcuda-gate}; else DIR=${NS_GATE_DIR:-$HOME/.cache/nextcuda-gate}; fi
LOCK=${NS_GATE_LOCK:-'$HOME/.nextcuda-gpu.lock'}
RECORD=${NS_GATE_RECORD:-0}
WAIT=${NS_GATE_WAIT:-7200}
GPUS=${NS_GATE_GPUS:-}
PATHS=${NS_GATE_PATHS:-}
B=tooling/gate-baselines.json
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST")
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
# a command in the build copy, its output through a file (an ssh session to the box prints its own noise on stdout)
on() {
  if [ -z "$HOST" ]; then mkdir -p "$DIR"; (cd "$DIR" && bash -c "$1"); return; fi
  "${SSH[@]}" "mkdir -p ~/$DIR && cd ~/$DIR && { $1 ; } > .gate-msg 2>&1; echo \$? > .gate-rc" >/dev/null 2>&1 || true
  scp -q "$HOST:$DIR/.gate-msg" "$tmp/msg" 2>/dev/null || : > "$tmp/msg"
  scp -q "$HOST:$DIR/.gate-rc" "$tmp/rc" 2>/dev/null || echo 255 > "$tmp/rc"
  cat "$tmp/msg"
  return "$(cat "$tmp/rc")"
}
# a directory of the build copy as rsync and scp name it
there() { if [ -n "$HOST" ]; then echo "$HOST:$DIR/${1:-}"; else echo "$DIR/${1:-}"; fi; }
RSYNC=(rsync -a); [ -z "$HOST" ] || RSYNC+=(-e "ssh -o BatchMode=yes")
# the baselines' paths where they are on this box (NS_GATE_PATHS)
mapped() { local out=$1 pair; IFS=, read -ra pairs <<< "$PATHS"; for pair in "${pairs[@]}"; do [ -n "$pair" ] && out=${out//"${pair%%=*}"/"${pair#*=}"}; done; printf '%s' "$out"; }

# ---- what changed, and the engines it touches
known=$(jq -r '.engines | keys[]' "$B")
declare -A want=()     # engine -> "all" | "variant:<arch>"
for e in "$@"; do
  if [ "$e" = all ]; then for k in $known; do want[$k]=all; done; else want[$e]=all; fi
done
# the engines of the baselines on a path's kind/arch (an engine's sub-runs are kind/arch#name)
engines_of() { for k in $known; do [ "${k%%#*}" = "$1" ] && echo "$k"; done; true; }
if [ $# -eq 0 ]; then
  for p in $(git diff --name-only "$BASE" "$HEAD"); do
    case "$p" in
      kernels/*/*/silo/variants/*)
        k=${p#kernels/}; kind=${k%%/*}; k=${k#*/}; arch=${k%%/*}
        for e in $(engines_of "$kind/$arch"); do [ "${want[$e]:-}" = all ] || want[$e]="variant:$arch"; done ;;
      */contract/*) ;;
      kernels/*/*/*|llm/*/*|image/*/*|video/*/*|audio/*/*|model3d/*/*)
        k=${p#kernels/}; kind=${k%%/*}; k=${k#*/}; arch=${k%%/*}
        for e in $(engines_of "$kind/$arch"); do want[$e]=all; done ;;
    esac
    # a shared part: the engines the baselines map it to ("*" every one, "image/*" a kind's)
    for prefix in $(jq -r '.shared | keys[]' "$B"); do
      [[ "$p" == "$prefix"* ]] || continue
      for pat in $(jq -r --arg k "$prefix" '.shared[$k][]' "$B"); do
        for eng in $known; do [[ "$eng" == $pat ]] && want[$eng]=all; done
      done
    done
  done
fi
if [ ${#want[@]} -eq 0 ]; then echo "gate: no engine of the baselines is affected by $BASE..$HEAD"; exit 0; fi
echo "gate: $(printf '%s\n' "${!want[@]}" | sort | tr '\n' ' ')"

# ---- the build copy: <head>'s tree (files that changed get a new time), the program built in it, the kernels built
# where the GPUs are (by content: a changed file stamped now; a changed header stamps the sources beside it)
if [ "${NS_GATE_SKIP_BUILD:-0}" != 1 ]; then
  LTREE=${NS_GATE_BUILD:-$HOME/.cache/nextcuda-gate}/tree
  mkdir -p "$LTREE"
  git archive "$HEAD" | tar -x -C "$tmp" --one-top-level=tree
  rsync -rlpc --delete --exclude /target --exclude /target-trixie --exclude /dist "$tmp/tree/" "$LTREE/"
  rm -rf "$tmp/tree"
  echo "gate: building $(git rev-parse --short "$HEAD")"
  (cd "$LTREE" && ./build.sh) > "$tmp/build.log" 2>&1 || { tail -30 "$tmp/build.log"; echo "gate: the program does not build"; exit 1; }
  on "mkdir -p src dist obj out in run" >/dev/null
  "${RSYNC[@]}" --checksum --no-times --delete --out-format='%n' "$LTREE/kernels/" "$(there src/kernels/)" > "$tmp/kchanged"
  "${RSYNC[@]}" --exclude 'lib*.so' --exclude /wfe "$LTREE/dist/" "$(there dist/)" >/dev/null
  "${RSYNC[@]}" "$LTREE/reference/meshes/" "$(there in/meshes/)" >/dev/null
  # kernels/build.sh compares a source's time with its object's: a header that changed makes its directory's sources new
  hdrs=$( { grep -E '\.(inc|h|cuh|hpp)$' "$tmp/kchanged" || true; } | xargs -r -n1 dirname | sort -u | tr '\n' ' ')
  [ -z "$hdrs" ] || on "cd src/kernels && for d in $hdrs; do find \$d -name '*.cu' -exec touch {} +; done" >/dev/null
  echo "gate: building the kernels"
  on "export PATH=/usr/local/cuda/bin:\$PATH; cd src/kernels && OUT=\$(cd ../../dist && pwd) OBJ=\$(cd ../../obj && pwd) ./build.sh 2>&1 | tail -15" > "$tmp/kbuild.log" \
    || { cat "$tmp/kbuild.log"; echo "gate: the kernels do not build"; exit 1; }
  if grep -q 'FAIL' "$tmp/kbuild.log"; then cat "$tmp/kbuild.log"; echo "gate: the kernels do not build"; exit 1; fi
  on "echo $(git rev-parse "$HEAD") > built" >/dev/null
fi

# ---- the copy's settings: its own out and sockets, and a copy of the box's registry (read there, never written)
on "[ ! -f ~/nextcuda/models.json ] || cp -f ~/nextcuda/models.json models.json; { [ ! -f models.json ] || echo NS_REGISTRY=\$PWD/models.json; [ ! -d ~/nextcuda/models ] || echo NS_MODELS=\$HOME/nextcuda/models; echo NS_SOCKET_DIR=\$PWD/run; echo NS_IMAGE_OUT=\$PWD/out; } > nextcuda.conf" >/dev/null

# ---- each GPU's card id and what it reports (`card N ID {json}`), and the box's name
cards=$(on "dist/nextcuda silo 2>/dev/null" | grep '^card ' || true)
if [ -z "$cards" ]; then
  # a program from before `nextcuda silo`: the id from `nextcuda gpus` (name, SMs, clock) and nvidia-smi's memory - which
  # can round to another GiB than cudaGetDeviceProperties does (the 5070: 12227 MiB there, 11772 here: 12g, not 11g);
  # record baselines on a build that has `nextcuda silo`
  cards=$(on "dist/nextcuda gpus 2>/dev/null | grep -E '^GPU [0-9]+: .*compute units' | head -16; nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits" | awk '
    /^GPU / { i = $2; sub(":", "", i); line = $0; sub(/^GPU [0-9]+: /, "", line); n = line; sub(/, .*/, "", n)
              match(line, /[0-9]+ compute units at [0-9]+ MHz/); s = substr(line, RSTART, RLENGTH); split(s, w, " "); name[i] = n; u[i] = w[1]; m[i] = w[5]; next }
    /^[0-9]+, [0-9]+$/ { split($0, w, ", "); mb[w[1]] = w[2] }
    END { for (i in name) { id = tolower(name[i]); gsub(/\(r\)|\(tm\)/, " ", id); gsub(/[^a-z0-9]+/, " ", id); out = ""
            k = split(id, ws, " "); for (j = 1; j <= k; j++) if (ws[j] != "" && ws[j] != "nvidia" && ws[j] != "geforce" && ws[j] != "intel" && ws[j] != "graphics") out = out ws[j] "-"
            printf "card %s %s%su-%dg {\"name\": \"%s\", \"units\": %s, \"mhz\": %s, \"vram_mb\": %s, \"cc\": null}\n", i, out, u[i], int((mb[i] + 512) / 1024), name[i], u[i], m[i], mb[i] } }')
fi
[ -n "$cards" ] || { echo "gate: no GPU found there (nextcuda silo, nextcuda gpus)"; exit 1; }
box=$(on "hostname -s" || echo unknown)
# a GPU set's id ("0,1": the ids joined by +) and devices (one: its JSON; several: a list)
set_id() { local g out=; for g in ${1//,/ }; do i=$(awk -v g="$g" '$1 == "card" && $2 == g { print $3 }' <<< "$cards"); [ -n "$i" ] || return 1; out=${out:+$out+}$i; done; echo "$out"; }
set_dev() {
  local g; local -a d=()
  for g in ${1//,/ }; do d+=("$(sed -n "s/^card $g [^ ]* //p" <<< "$cards")"); done
  if [ ${#d[@]} = 1 ]; then echo "${d[0]}"; else printf '%s\n' "${d[@]}" | jq -cs .; fi
}

# ---- the plan: each engine on each of its GPUs (or sets); a variant's change only where a GPU takes that variant
printf '[]' > "$tmp/plan.json"
rows=(); fail=0
for eng in $(printf '%s\n' "${!want[@]}" | sort); do
  if ! jq -e --arg e "$eng" '.engines[$e]' "$B" >/dev/null; then rows+=("| $eng | - | no baseline | - | skip |"); continue; fi
  list=$GPUS; [ -n "$list" ] || list=$(jq -r --arg e "$eng" '.engines[$e].gpus[]' "$B" | tr '\n' ' ')
  for g in $list; do
    if ! id=$(set_id "$g"); then rows+=("| $eng | $g | no such GPU here (nextcuda silo) | - | FAIL |"); fail=1; continue; fi
    if [[ "${want[$eng]}" == variant:* ]]; then
      arch=${want[$eng]#variant:}
      took=$(on "dist/nextcuda silo $arch 2>/dev/null" | sed -n "s/^silo $arch GPU \([0-9]*\): \([^ ]*\).*/\1 \2/p" | awk -v s=",$g," 'index(s, "," $1 ",") && $2 != "default" { print $2 }')
      if [ -z "$took" ]; then rows+=("| $eng | $g $id | its variant does not match these GPUs | - | skip |"); continue; fi
    fi
    if [ "$RECORD" != 1 ] && ! jq -e --arg e "$eng" --arg c "$id" '.engines[$e].cards[$c]' "$B" >/dev/null; then
      rows+=("| $eng | $g $id | no baseline for this card (record one on it: NS_GATE_RECORD=1 on the base commit) | - | skip |"); continue
    fi
    jq --arg e "$eng" --arg g "$g" --arg id "$id" --arg run "$(mapped "$(jq -r --arg e "$eng" '.engines[$e].run' "$B")")" \
       --arg setup "$(mapped "$(jq -r --arg e "$eng" '.engines[$e].setup // ""' "$B")")" --slurpfile b "$B" \
       '. + [{engine: $e, card: $g, id: $id, run: $run, setup: $setup, timeout_s: $b[0].engines[$e].timeout_s,
              outputs: $b[0].engines[$e].outputs, speed: ($b[0].engines[$e].speed // [])}]' "$tmp/plan.json" > "$tmp/p" && mv "$tmp/p" "$tmp/plan.json"
  done
done

# ---- the runs: a runner on the box (it outlives an ssh session), results as JSON lines
cat > "$tmp/runner.sh" <<'RUNNER'
#!/usr/bin/env bash
# the gate's runner: reads plan.json, writes results.jsonl, then touches done
set -uo pipefail
cd "$(dirname "$0")"
WAIT=$1; LOCK=$(eval echo "$2")
: > results.jsonl; rm -f done
export PATH=/usr/local/cuda/bin:$PATH
# the box idle for these GPUs: no other process holds them, nothing runs or waits for the creative platform here
idle() {
  local c n
  for c in ${1//,/ }; do
    n=$(nvidia-smi -i "$c" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${n:-0}" = 0 ] || return 1
  done
  n=$(curl -sk -m 15 -X POST https://127.0.0.1:8791/api/GetGpus -H 'Content-Type: application/json' -d '{}' 2>/dev/null \
      | jq -r '[.cards[] | select(.remote != true or .index == "remote-nextcuda") | (.running // 0) + (.queued // 0)] | add // 0' 2>/dev/null)
  [ "${n:-0}" = 0 ]
}
locked() { if [ "$LOCK" = none ]; then "$@"; else flock -w "$WAIT" "$LOCK" "$@"; fi; }
hash_outputs() {   # $1 out dir, $2 outputs JSON: "FILE" its bytes, "frames:FILE" a clip's decoded frames, "lines:FILE:REGEX" lines of it, "dir:DIR" every file in it
  local out=$1 o f
  for o in $(jq -r '.[] | @base64' <<< "$2"); do
    o=$(base64 -d <<< "$o")
    case "$o" in
      frames:*) f=${o#frames:}; ffmpeg -loglevel error -i "$out/$f" -map 0:v -f md5 - 2>&1 ;;
      lines:*) f=${o#lines:}; tr '\r' '\n' < "$out/${f%%:*}" | grep -aE "${f#*:}" ;;
      dir:*) f=${o#dir:}; (cd "$out/$f" && find . -type f | LC_ALL=C sort | xargs -r sha256sum) ;;
      *) sha256sum < "$out/$o" ;;
    esac
  done | sha256sum | cut -c1-16
}
fail() { jq -cn --arg e "$e" --arg c "$card" --arg id "$id" --arg l "$1" '{engine: $e, card: $c, id: $id, error: $l}' >> results.jsonl; }
# one run (twice) of plan item $1, its GPUs idle
run_one() {
  local i=$1
  e=$(jq -r ".[$i].engine" plan.json); card=$(jq -r ".[$i].card" plan.json); id=$(jq -r ".[$i].id" plan.json); lim=$(jq -r ".[$i].timeout_s" plan.json)
  slug=$(echo "$e-$card" | tr '/#,' '__-')
  exacts=(); metrics=()
  for r in 1 2; do
    out=$PWD/out/$slug-$r; rm -rf "$out"; mkdir -p "$out"
    run=$(jq -r ".[$i].run" plan.json); setup=$(jq -r ".[$i].setup" plan.json)
    run=${run//\{gpu\}/$card}; run=${run//\{out\}/$out}; run=${run//\{in\}/$PWD/in}; setup=${setup//\{in\}/$PWD/in}
    if [ -n "$setup" ] && ! bash -c "$setup" > "$out/setup.log" 2>&1; then fail "setup: $(tail -2 "$out/setup.log" | tr '\n' ' ')"; return; fi
    locked timeout --kill-after=60 "$lim" bash -c "$run" > "$out/stdout" 2> "$out/stderr"
    rc=$?
    [ $rc = 0 ] || { fail "rc $rc: $(tr '\r' '\n' < "$out/stderr" | grep -v '^\s*$' | tail -2 | tr '\n' ' ' | cut -c1-300)"; return; }
    exacts+=("$(hash_outputs "$out" "$(jq -c ".[$i].outputs" plan.json)")")
    m='{}'
    for s in $(jq -r ".[$i].speed[] | @base64" plan.json); do
      s=$(base64 -d <<< "$s"); name=$(jq -r .name <<< "$s"); re=$(jq -r .re <<< "$s"); from=$(jq -r '.from // "stderr"' <<< "$s")
      v=$(tr '\r' '\n' < "$out/$from" | sed -nE "s"$'\001'"$re"$'\001''\1'$'\001'"p" | tail -1)   # (a pattern may hold a /)
      m=$(jq -c --arg n "$name" --arg v "${v:-x}" '. + {($n): ($v | tonumber? // null)}' <<< "$m")
    done
    metrics+=("$m")
  done
  jq -cn --arg e "$e" --arg c "$card" --arg id "$id" --arg x "$(printf '%s\n' "${exacts[@]}" | sort -u | paste -sd/)" \
     --argjson a "${metrics[0]}" --argjson b "${metrics[1]}" '{engine: $e, card: $c, id: $id, exact: $x, runs: [$a, $b]}' >> results.jsonl
}
# the runs whose GPUs are idle, pass after pass (a run waiting for a busy card holds no other back), until each ran or
# WAIT seconds passed with the box busy for it
n=$(jq length plan.json)
pending=$(seq 0 $((n - 1)))
t0=$(date +%s)
while [ -n "$pending" ]; do
  left=
  for i in $pending; do
    if idle "$(jq -r ".[$i].card" plan.json)"; then run_one "$i"; t0=$(date +%s); else left="$left $i"; fi
  done
  pending=$left
  [ -n "$pending" ] || break
  if [ $(( $(date +%s) - t0 )) -gt "$WAIT" ]; then
    for i in $pending; do e=$(jq -r ".[$i].engine" plan.json); card=$(jq -r ".[$i].card" plan.json); id=$(jq -r ".[$i].id" plan.json); fail "the box stayed busy: not run"; done
    break
  fi
  sleep 30
done
touch done
RUNNER
if [ "$(jq length "$tmp/plan.json")" -gt 0 ]; then
  "${RSYNC[@]}" "$tmp/plan.json" "$tmp/runner.sh" "$(there)/" >/dev/null
  echo "gate: $(jq length "$tmp/plan.json") runs (each twice; waiting for the box to be idle as needed)"
  if [ -n "$HOST" ]; then
    on "chmod +x runner.sh && rm -f done && (setsid nohup ./runner.sh $WAIT '$LOCK' > runner.log 2>&1 &)" >/dev/null
    until "${SSH[@]}" "test -f ~/$DIR/done" >/dev/null 2>&1; do sleep 20; done
  else
    (cd "$DIR" && chmod +x runner.sh && ./runner.sh "$WAIT" "$LOCK" > runner.log 2>&1)
  fi
  "${RSYNC[@]}" "$(there results.jsonl)" "$tmp/results.jsonl" >/dev/null
else
  : > "$tmp/results.jsonl"
fi

# ---- compare (or record): the exact output equal, each throughput of the better run within the tolerance
tol=$(jq -r '.tolerance' "$B")
best() {   # the better of the two runs' values of metric $2 ("lower" or "higher" better: $3) in result $1
  jq -c --arg n "$2" --arg d "$3" '[.runs[][$n] | select(. != null)] | if length == 0 then null elif $d == "lower" then min else max end' <<< "$1"
}
at="$(git rev-parse --short "$HEAD") ($(date +%F))"
while read -r res; do
  [ -n "$res" ] || continue
  e=$(jq -r .engine <<< "$res"); g=$(jq -r .card <<< "$res"); id=$(jq -r .id <<< "$res")
  if jq -e .error <<< "$res" >/dev/null; then rows+=("| $e | $g $id | $(jq -r .error <<< "$res" | tr '|' '/') | - | FAIL |"); fail=1; continue; fi
  x=$(jq -r .exact <<< "$res")
  spd=(); ok=PASS
  if [ "$RECORD" = 1 ]; then
    jq --arg e "$e" --arg c "$id" --arg x "$x" --argjson d "$(set_dev "$g")" --arg h "$box" --arg at "$at" \
       '.engines[$e].cards[$c] = {exact: $x, device: $d, host: $h, recorded: $at}' "$B" > "$tmp/b" && mv "$tmp/b" "$B"
  fi
  for s in $(jq -r --arg e "$e" '.engines[$e].speed[] | @base64' "$B"); do
    s=$(base64 -d <<< "$s"); n=$(jq -r .name <<< "$s"); d=$(jq -r '.better // "higher"' <<< "$s"); v=$(best "$res" "$n" "$d")
    if [ "$RECORD" = 1 ]; then
      jq --arg e "$e" --arg c "$id" --arg n "$n" --argjson v "$v" '.engines[$e].cards[$c][$n] = $v' "$B" > "$tmp/b" && mv "$tmp/b" "$B"
      spd+=("$n $v"); continue
    fi
    bv=$(jq -c --arg e "$e" --arg c "$id" --arg n "$n" '.engines[$e].cards[$c][$n]' "$B")
    if [ "$v" = null ] || [ "$bv" = null ]; then spd+=("$n $v (no baseline)"); continue; fi
    pct=$(awk -v n="$v" -v b="$bv" 'BEGIN { printf "%+.1f%%", (n / b - 1) * 100 }')
    awk -v n="$v" -v b="$bv" -v t="$tol" -v d="$d" 'BEGIN { exit !(d == "lower" ? n <= b * (1 + t) : n >= b * (1 - t)) }' || ok=FAIL
    spd+=("$n $v ($pct)")
  done
  if [ "$RECORD" = 1 ]; then
    [[ "$x" == */* ]] && note="UNSTABLE: the two runs differ ($x)" || note="recorded $x"
    rows+=("| $e | $g $id | $note | ${spd[*]:-} | - |"); continue
  fi
  bx=$(jq -r --arg e "$e" --arg c "$id" '.engines[$e].cards[$c].exact // "none"' "$B")
  nd=$(jq -r --arg e "$e" '.engines[$e].nondeterministic // ""' "$B")
  if [ -n "$nd" ]; then xs="not compared (nondeterministic: $nd)"
  elif [ "$x" = "$bx" ] && [[ "$x" != */* ]]; then xs="same ($x)"; else xs="DIFFERENT ($x, baseline $bx)"; ok=FAIL; fi
  [ "$ok" = PASS ] || fail=1
  rows+=("| $e | $g $id | $xs | ${spd[*]:-} | $ok |")
done < "$tmp/results.jsonl"
[ "$RECORD" != 1 ] || { jq --arg r "$at" '.recorded = $r' "$B" > "$tmp/b" && mv "$tmp/b" "$B"; }
echo
echo "gate $BASE..$HEAD on $([ -n "$HOST" ] && echo "$HOST" || echo "this machine") ($box; best of two against the baseline, tolerance $(awk -v t="$tol" 'BEGIN { print t * 100 }')%)"
echo "| engine | GPUs | exact output | throughput | result |"
echo "|---|---|---|---|---|"
printf '%s\n' "${rows[@]}"
exit $fail
