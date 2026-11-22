#!/usr/bin/env bash
# kernels/build.sh: the kernel libraries, one a kind (libnextcuda-image.so ...): each the shared part (kernels/ns: GPUs,
# memory, copies, the C ABI's core) and that kind's engines (kernels/<kind>/<arch>), built with nvcc; and the silos,
# a model's kernels tuned for it alone (kernels/<kind>/<arch>/silo/*.cu -> dist/silo/libnextcuda-<arch>.so), which its
# engine opens at load. A silo is one library for every card, per model architecture, never per card: what differs by
# card is chosen in it by what the card reports (its compute capability, its SMs), or is a variant of it - a silo's
# variants (kernels/<kind>/<arch>/silo/variants/<name>/: only the ops that differ for a class of card, variant.yml says
# which) are dist/silo/libnextcuda-<arch>@<name>.so with the manifest beside it (libnextcuda-<arch>@<name>.yml). Runs
# where nvcc is (the GPU host; tooling/deploy.sh copies the sources there).
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
FLAGS=(-std=c++17 -O3 --use_fast_math -Xcompiler -fPIC -lineinfo "${GEN[@]}" -Ins -Ins -Idiffusion -Illm/nemotronh -Illm/qwen35)
# --use_fast_math is not on for the numerics that must match a reference: a source names its own flags below
# (MOSS-Image's silo too: it was built a card at a time without it until 2026-11-22, and its outputs stay as they were)
precise() { case "$1" in diffusion/*|llm/*|audio/*|model3d/skintokens/*|model3d/unimate/*|video/ltx2/*|image/rembg/*|image/mossimage/silo/*) return 0 ;; esac; return 1; }

KINDS="llm image video audio model3d"
sources() {
  ls ns/*.cu diffusion/*.cu 2>/dev/null || true
  for k in $KINDS; do ls $k/*/*.cu 2>/dev/null || true; done
}
silo_sources() { for k in $KINDS; do ls $k/*/silo/*.cu $k/*/silo/variants/*/*.cu 2>/dev/null || true; done; }

compile() {
  f=$1; o="$OBJ/$(echo "$f" | tr / _).o"
  fl=()
  for a in "${FLAGS[@]}"; do
    precise "$f" && [ "$a" = "--use_fast_math" ] && continue
    fl+=("$a")
  done
  # an LLM source also when a header of the LLM kernels changed (nh_blocks.cuh is the block reader of both engines); any
  # source when a header or included part beside it changed (nsd.cu's .inc files: a changed one was not rebuilt before)
  hdr=
  case "$f" in llm/*) hdr=$(find llm -name '*.cuh' -newer "$o" 2>/dev/null | head -1) ;; esac
  [ -n "$hdr" ] || [ ! -f "$o" ] || hdr=$(find "$(dirname "$f")" -maxdepth 1 \( -name '*.inc' -o -name '*.h' -o -name '*.cuh' -o -name '*.hpp' \) -newer "$o" 2>/dev/null | head -1)
  if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || [ ns/ns_internal.hpp -nt "$o" ] || [ -n "$hdr" ]; then
    "$NVCC" "${fl[@]}" -c "$f" -o "$o" && echo "ok   $f" || { echo "FAIL $f"; return 1; }
  else echo "cached $f"; fi
}
export -f compile precise
export NVCC OBJ
export FLAGS_STR="${FLAGS[*]}"
fail=0
for f in $(sources) $(silo_sources); do compile "$f" || fail=1; done
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
    # image library too. The library exports only its API (image.map: ns_*, nsd_* and ns_fail for the silos)
    objs="$objs $(ls "$OBJ"/llm_*.o 2>/dev/null | grep -v '_silo_' || true)"
    extra+=(-Xlinker --version-script=image.map)
  fi
  if [ "$k" = audio ]; then
    # Haru-4B's language model is Qwen3.5: the llm library's qwen35 and nemotronh kernels in the audio library too
    objs="$objs $(ls "$OBJ"/llm_*.o 2>/dev/null | grep -v '_silo_' || true)"
  fi
  echo "==> linking libnextcuda-$k.so"
  "$NVCC" -shared "${GEN[@]}" $objs "${extra[@]}" -lcublas -lcublasLt -o "$OUT/.libnextcuda-$k.so.new"
  mv -f "$OUT/.libnextcuda-$k.so.new" "$OUT/libnextcuda-$k.so"
done
# (the libraries of silos built a card at a time, until 2026-11-22: no engine opens them now)
rm -f "$OUT"/silo/libnextcuda-*.sm[0-9]*.so
for d in $(ls -d */*/silo 2>/dev/null); do
  arch=$(basename "$(dirname "$d")")
  objs=$(ls "$OBJ"/$(echo "$d" | tr / _)_*.o 2>/dev/null | grep -v '_silo_variants_' || true)
  [ -n "$objs" ] || continue
  echo "==> linking silo/libnextcuda-$arch.so"
  "$NVCC" -shared "${GEN[@]}" $objs -o "$OUT/silo/.libnextcuda-$arch.so.new"
  mv -f "$OUT/silo/.libnextcuda-$arch.so.new" "$OUT/silo/libnextcuda-$arch.so"
done
# a silo's variants: only the ops that differ, under the silo's C names - opened locally over the default silo, and
# -Bsymbolic so that within it its ops call its own; the manifest beside it (a variant without one is never taken)
want_v=" "
for d in $(ls -d */*/silo/variants/*/ 2>/dev/null); do
  d=${d%/}; name=$(basename "$d"); arch=$(basename "$(dirname "$(dirname "$(dirname "$d")")")")
  objs=$(ls "$OBJ"/$(echo "$d" | tr / _)_*.o 2>/dev/null || true)
  [ -n "$objs" ] && [ -f "$d/variant.yml" ] || { echo "skip $d (no sources or no variant.yml)"; continue; }
  echo "==> linking silo/libnextcuda-$arch@$name.so"
  "$NVCC" -shared "${GEN[@]}" $objs -Xlinker -Bsymbolic -o "$OUT/silo/.libnextcuda-$arch@$name.so.new"
  cp "$d/variant.yml" "$OUT/silo/libnextcuda-$arch@$name.yml"
  mv -f "$OUT/silo/.libnextcuda-$arch@$name.so.new" "$OUT/silo/libnextcuda-$arch@$name.so"
  want_v="$want_v$arch@$name "
done
# a variant whose sources are gone is not left to be taken
for f in "$OUT"/silo/libnextcuda-*@*.so; do
  [ -f "$f" ] || continue
  v=${f##*/libnextcuda-}; v=${v%.so}
  case "$want_v" in *" $v "*) ;; *) echo "gone $f"; rm -f -- "$f" "${f%.so}.yml" ;; esac
done
ls -la "$OUT"/libnextcuda*.so "$OUT"/silo/*.so "$OUT"/silo/*.yml 2>/dev/null || true
