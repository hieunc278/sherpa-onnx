#!/usr/bin/env bash
# Phase 10 evaluation: generate fp32 and int8 outputs for a fixed test set
# (sentences x voices), for later scoring (quality/WER/RTF/size) --
# see evals/zerotts-int8-baseline.md.
set -e

REPO=/mnt/disk2/projects/sherpa-onnx-zero-tts
BUILD=$REPO/build_shared
MODEL=$REPO/assets/zerotts/model
OUT=$REPO/assets/zerotts/phase10_eval
LD_LIB=$(find "$BUILD" -iname "libonnxruntime.so*" -exec dirname {} \; | head -1)
mkdir -p "$OUT"

declare -a SENTENCES=(
  "Xin chào."
  "Xin chào Việt Nam."
  "Cảm ơn bạn đã sử dụng ZeroTTS, một mô hình chạy trên CPU, không cần GPU."
)
declare -a LABELS=("short" "medium" "long")
declare -a VOICES=("maichi" "giahuy")

run_one() {
  local variant=$1 text_encoder=$2 prefix_step=$3 local_frame_decode=$4
  local codec_full=$5 codec_step=$6 voice=$7 out=$8 text=$9

  t0=$(date +%s.%N)
  LD_LIBRARY_PATH=$LD_LIB "$BUILD/bin/sherpa-onnx-offline-tts" \
    --zerotts-text-encoder="$text_encoder" \
    --zerotts-prefix-step="$prefix_step" \
    --zerotts-local-frame-decode="$local_frame_decode" \
    --zerotts-codec-decode-full="$codec_full" \
    --zerotts-codec-decode-step="$codec_step" \
    --zerotts-codec-meta="$MODEL/onnx/codec/codec_browser_onnx_meta.json" \
    --zerotts-tokenizer="$MODEL/tokenizer.json" \
    --zerotts-null-voice-emb="$MODEL/null_voice_emb.npy" \
    --zerotts-silence-frame="$MODEL/silence_frame.npy" \
    --zerotts-voice="$MODEL/voices/$voice" \
    --output-filename="$out" \
    "$text" > "${out%.wav}.log" 2>&1
  t1=$(date +%s.%N)
  echo "$variant $voice $out $(echo "$t1 - $t0" | bc)" >> "$OUT/timings.txt"
}

rm -f "$OUT/timings.txt"

for i in "${!SENTENCES[@]}"; do
  text="${SENTENCES[$i]}"
  label="${LABELS[$i]}"
  for voice in "${VOICES[@]}"; do
    echo "=== fp32 $label $voice ==="
    run_one fp32 \
      "$MODEL/onnx/text_encoder.onnx" \
      "$MODEL/onnx/prefix_step.onnx" \
      "$MODEL/onnx/local_frame_decode.onnx" \
      "$MODEL/onnx/codec/moss_audio_tokenizer_decode_full.onnx" \
      "$MODEL/onnx/codec/moss_audio_tokenizer_decode_step.onnx" \
      "$voice" "$OUT/fp32_${label}_${voice}.wav" "$text"

    echo "=== int8 $label $voice ==="
    run_one int8 \
      "$MODEL/onnx/text_encoder.int8.onnx" \
      "$MODEL/onnx/prefix_step.int8.onnx" \
      "$MODEL/onnx/local_frame_decode.int8.onnx" \
      "$MODEL/onnx/codec/moss_audio_tokenizer_decode_full.int8.onnx" \
      "$MODEL/onnx/codec/moss_audio_tokenizer_decode_step.int8.onnx" \
      "$voice" "$OUT/int8_${label}_${voice}.wav" "$text"
  done
done

echo "Done. See $OUT/timings.txt"
