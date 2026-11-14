#!/usr/bin/env bash
# kernels/build.sh: the kernel libraries, one a kind (libnextcuda-image.so ...): each the shared part (kernels/ns: GPUs,
# memory, copies, the C ABI's core) and that kind's engines (kernels/<kind>/<arch>), built with nvcc; and the silos,
# a model's kernels tuned for it alone (kernels/<kind>/<arch>/silo/*.cu -> dist/silo/libnextcuda-<arch>.so), which its
# engine opens at load. A silo with a `per-card` file is built once a card instead (-> libnextcuda-<arch>.sm<NN>.so,
# compiled for that card alone with NS_SILO_SM=NN): the 3060's kernels and the 5070's can differ (tile sizes, FP8
# on the card that has it), and each card's engine opens its own. Runs where nvcc is (the GPU host; ../build.sh
# copies the sources there).
#   ARCHS="86 120" (default: the cards of the box: sm_86 3060, sm_120 5070; + PTX of the newest)   JOBS=4
set -euo pipefail
cd "$(dirname "$0")"
CUDA=${CUDA_HOME:-/usr/local/cuda}
NVCC=${NVCC:-$CUDA/bin/nvcc}
ARCHS=${ARCHS:-"86 120"}
JOBS=${JOBS:-4}
OUT=${OUT:-../dist}
OBJ=${OBJ:-../target/kernels}
mkdir -p "$OUT/silo" "$OBJ"

GEN=()
last=
for a in $ARCHS; do GEN+=(-gencode "arch=compute_$a,code=sm_$a"); last=$a; done
GEN+=(-gencode "arch=compute_$last,code=compute_$last")      # PTX of the newest: a newer card JITs it
FLAGS=(-std=c++17 -O3 --use_fast_math -Xcompiler -fPIC -lineinfo "${GEN[@]}" -Ins -Ins -Idiffusion -Illm/nemotronh -Illm/qwen35 -Illm/nemotronh/third_party -Illm/nemotronh/third_party/ggml)
# --use_fast_math is not on for the numerics that must match a reference: a source names its own flags below
precise() { case "$1" in diffusion/*|llm/*|audio/*|model3d/skintokens/*|model3d/unimate/*|video/ltx2/*|image/rembg/*) return 0 ;; esac; return 1; }

KINDS="llm image video audio model3d"
sources() {
  ls ns/*.cu diffusion/*.cu 2>/dev/null || true
  for k in $KINDS; do ls $k/*/*.cu 2>/dev/null || true; done
}
silo_sources() { for k in $KINDS; do for d in $(ls -d $k/*/silo 2>/dev/null); do [ -f "$d/per-card" ] || ls $d/*.cu 2>/dev/null || true; done; done; }
card_silos() { for k in $KINDS; do for d in $(ls -d $k/*/silo 2>/dev/null); do [ -f "$d/per-card" ] && echo "$d"; done; done; true; }

compile() {
  f=$1; o="$OBJ/$(echo "$f" | tr / _).o"
  fl=()
  for a in "${FLAGS[@]}"; do
    precise "$f" && [ "$a" = "--use_fast_math" ] && continue
    fl+=("$a")
  done
  if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || [ ns/ns_internal.hpp -nt "$o" ]; then
    "$NVCC" "${fl[@]}" -c "$f" -o "$o" && echo "ok   $f" || { echo "FAIL $f"; return 1; }
  else echo "cached $f"; fi
}
export -f compile precise
export NVCC OBJ
export FLAGS_STR="${FLAGS[*]}"
fail=0
for f in $(sources) $(silo_sources); do compile "$f" || fail=1; done
# a per-card silo's sources once a card, for that card alone
for d in $(card_silos); do
  for a in $ARCHS; do
    for f in $d/*.cu; do
      o="$OBJ/card$a/$(echo "$f" | tr / _).o"
      mkdir -p "$OBJ/card$a"
      if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || [ ns/ns_internal.hpp -nt "$o" ] || [ -n "$(find "$d" -name '*.h' -newer "$o" 2>/dev/null)" ]; then
        "$NVCC" -std=c++17 -O3 -Xcompiler -fPIC -lineinfo -gencode "arch=compute_$a,code=sm_$a" -DNS_SILO_SM=$a -Ins -c "$f" -o "$o" \
          && echo "ok   $f (sm_$a)" || { echo "FAIL $f (sm_$a)"; fail=1; }
      else echo "cached $f (sm_$a)"; fi
    done
  done
done
[ "$fail" = 0 ] || { echo "==> compile failed"; exit 1; }
# an object whose source is gone (a removed engine's, 2026-11-13) is not linked again
want=" $(for f in $(sources) $(silo_sources); do printf '%s ' "$OBJ/$(echo "$f" | tr / _).o"; done)"
for o in "$OBJ"/*.o; do
  [ -f "$o" ] || continue
  case "$want" in *" $o "*) ;; *) echo "gone $o"; rm -f -- "$o" ;; esac
done

for k in $KINDS; do
  objs="$(ls "$OBJ"/ns_*.o 2>/dev/null) $(ls "$OBJ"/${k}_*.o 2>/dev/null | grep -v '_silo_' || true)"
  # a kind with no kernels of its own is still a library when its engines run on the shared ones alone (video: H3
  # is all nsd)
  [ -n "$(ls "$OBJ"/${k}_*.o 2>/dev/null | grep -v '_silo_' || true)" ] || [ "$k" = video ] || continue
  # the shared diffusion kernels (nsd) are in every kind's library (llm's too: Qwen3.8's vision tower runs on them)
  objs="$objs $(ls "$OBJ"/diffusion_*.o 2>/dev/null || true)"
  extra=()
  if [ "$k" = image ]; then
    # MOSS-Image's text encoder is Qwen3.5 (gated DeltaNet): the llm library's qwen35 and nemotronh kernels in the
    # image library too. The library exports only its API (image.map: ns_*, nsd_* and ns_fail for the silos): their
    # vendored ggml-cuda copies stay inside it
    objs="$objs $(ls "$OBJ"/llm_*.o 2>/dev/null | grep -v '_silo_' || true)"
    extra+=(-Xlinker --version-script=image.map)
  fi
  echo "==> linking libnextcuda-$k.so"
  "$NVCC" -shared "${GEN[@]}" $objs "${extra[@]}" -lcublas -lcublasLt -o "$OUT/.libnextcuda-$k.so.new"
  mv -f "$OUT/.libnextcuda-$k.so.new" "$OUT/libnextcuda-$k.so"
done
for d in $(card_silos); do
  arch=$(basename "$(dirname "$d")")
  for a in $ARCHS; do
    objs=$(ls "$OBJ"/card$a/$(echo "$d" | tr / _)_*.o 2>/dev/null || true)
    [ -n "$objs" ] || continue
    echo "==> linking silo/libnextcuda-$arch.sm$a.so"
    "$NVCC" -shared -gencode "arch=compute_$a,code=sm_$a" $objs -o "$OUT/silo/.libnextcuda-$arch.sm$a.so.new"
    mv -f "$OUT/silo/.libnextcuda-$arch.sm$a.so.new" "$OUT/silo/libnextcuda-$arch.sm$a.so"
  done
done
for d in $(ls -d */*/silo 2>/dev/null); do
  [ -f "$d/per-card" ] && continue
  arch=$(basename "$(dirname "$d")")
  objs=$(ls "$OBJ"/$(echo "$d" | tr / _)_*.o 2>/dev/null || true)
  [ -n "$objs" ] || continue
  echo "==> linking silo/libnextcuda-$arch.so"
  "$NVCC" -shared "${GEN[@]}" $objs -o "$OUT/silo/.libnextcuda-$arch.so.new"
  mv -f "$OUT/silo/.libnextcuda-$arch.so.new" "$OUT/silo/libnextcuda-$arch.so"
done
ls -la "$OUT"/libnextcuda*.so "$OUT"/silo/*.so 2>/dev/null || true
