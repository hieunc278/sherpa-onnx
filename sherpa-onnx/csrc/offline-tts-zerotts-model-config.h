// sherpa-onnx/csrc/offline-tts-zerotts-model-config.h
//
// Copyright (c)  2026
//
// Config for ZeroTTS (Vietnamese zero-shot TTS). See spec.md §4 for the
// field-by-field provenance of the generation-hyperparameter defaults
// (confirmed against synthesizer.py during the Phase 0 spike).
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_MODEL_CONFIG_H_

#include <cstdint>
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsZeroTtsModelConfig {
  std::string text_encoder;        // text_encoder.onnx
  std::string prefix_step;         // prefix_step.onnx
  std::string local_frame_decode;  // local_frame_decode.onnx
  std::string codec_decode_full;   // onnx/codec/moss_audio_tokenizer_decode_full.onnx
  std::string codec_decode_step;   // onnx/codec/moss_audio_tokenizer_decode_step.onnx
  std::string codec_meta;          // onnx/codec/codec_browser_onnx_meta.json
  std::string tokenizer;           // tokenizer.json
  std::string null_voice_emb;      // null_voice_emb.npy
  std::string silence_frame;       // silence_frame.npy (currently unused by
                                   // the offline/streaming loop itself, but
                                   // part of the model release; kept as a
                                   // config field for parity/future use)
  std::string voice;               // path to a voices/<name>/ dir

  // Model hyperparameters (from config.json, not user-configurable -- the
  // release is pinned to these).
  int32_t d_model = 768;
  int32_t n_layers = 9;
  int32_t n_heads = 12;
  int32_t num_codebooks = 16;
  int32_t codebook_size = 1024;
  int32_t sample_rate = 48000;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  // Generation defaults -- see spec.md §4/§12.2 for source-line provenance
  // against synthesizer.py.
  int32_t min_frames = 4;
  int32_t max_frames = 1500;
  int32_t eoa_extra_frames = 1;
  float text_temperature = 1.0f;
  int32_t text_topk = 50;
  float audio_temperature = 0.8f;
  int32_t audio_topk = 25;
  float audio_topp = 0.95f;
  float audio_repetition_penalty = 1.2f;
  float cfg_scale = 1.0f;

  // Streaming-specific.
  int32_t first_chunk_frames = 1;
  int32_t max_chunk_frames = 16;

  OfflineTtsZeroTtsModelConfig() = default;

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_MODEL_CONFIG_H_
