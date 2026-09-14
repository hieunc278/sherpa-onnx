// sherpa-onnx/csrc/offline-tts-zerotts-voice.h
//
// Copyright (c)  2026
//
// Loads a ZeroTTS voice pack: voices/<name>/{voice.bin,meta.json}. See
// spec.md §6 / notes/phase0-voice-bin-format.md for why voice.bin (not the
// reference Python's voice.npz) is used here: raw float32, little-endian,
// row-major, no header -- confirmed against a real voice pack.
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_VOICE_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_VOICE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace sherpa_onnx {

struct OfflineTtsZeroTtsVoice {
  // Flat (1, n_voice_queries, d_model) float32, row-major.
  std::vector<float> emb;
  int32_t n_voice_queries = 0;
  int32_t d_model = 0;

  std::string name;
  std::string display_name;
};

// `voice_dir` is a path to voices/<name>/ (must contain voice.bin; meta.json
// is optional but required for n_voice_queries unless `expected_d_model` * 4
// evenly divides the file size in exactly one way -- see .cc). Returns true
// and fills `out` on success.
//
// `expected_d_model`: the model's d_model (from config.json, fixed at 768
// for this ZeroTTS release) -- used both to recover n_voice_queries when
// meta.json doesn't have it and to validate consistency when it does.
bool LoadOfflineTtsZeroTtsVoice(const std::string &voice_dir,
                                int32_t expected_d_model,
                                OfflineTtsZeroTtsVoice *out);

template <typename Manager>
bool LoadOfflineTtsZeroTtsVoice(Manager *mgr, const std::string &voice_dir,
                                int32_t expected_d_model,
                                OfflineTtsZeroTtsVoice *out);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_VOICE_H_
