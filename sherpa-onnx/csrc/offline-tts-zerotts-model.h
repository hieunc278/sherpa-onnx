// sherpa-onnx/csrc/offline-tts-zerotts-model.h
//
// Copyright (c)  2026
//
// Owns the 5 Ort::Session objects that make up ZeroTTS's pipeline (see
// spec.md §2/§3): text_encoder, prefix_step, local_frame_decode,
// codec_decode_full, codec_decode_step. One Run* method per graph;
// AR-loop orchestration (the actual generation loop) lives in
// OfflineTtsZeroTtsImpl, not here -- this class is purely the ONNX I/O
// binding layer, kept separately testable per plan.md Phase 3.
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-zerotts-model-config.h"

namespace sherpa_onnx {

// One entry of codec_browser_onnx_meta.json's streaming_decode.
// transformer_offsets (see notes/phase0-codec-decode-step-io.md).
struct ZeroTtsTransformerOffsetSpec {
  std::string input_name;
  std::string output_name;
  std::vector<int64_t> shape;
};

// One entry of codec_browser_onnx_meta.json's streaming_decode.
// attention_caches.
struct ZeroTtsAttentionCacheSpec {
  std::string offset_input_name;
  std::string offset_output_name;
  std::string cached_keys_input_name;
  std::string cached_keys_output_name;
  std::string cached_values_input_name;
  std::string cached_values_output_name;
  std::string cached_positions_input_name;
  std::string cached_positions_output_name;
  std::vector<int64_t> offset_shape;
  std::vector<int64_t> cache_shape;
  std::vector<int64_t> positions_shape;
};

// Parsed codec_browser_onnx_meta.json (see notes/phase0-codec-decode-step-io.md
// for the full field-by-field breakdown and why this is data-driven rather
// than hardcoded).
struct ZeroTtsCodecStreamingMeta {
  int32_t sample_rate = 48000;
  int32_t num_channels = 2;
  int32_t downsample_rate = 3840;
  int32_t num_quantizers = 16;
  std::vector<ZeroTtsTransformerOffsetSpec> transformer_offsets;
  std::vector<ZeroTtsAttentionCacheSpec> attention_caches;
};

class OfflineTtsZeroTtsModel {
 public:
  ~OfflineTtsZeroTtsModel();

  explicit OfflineTtsZeroTtsModel(const OfflineTtsZeroTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsZeroTtsModel(Manager *mgr,
                        const OfflineTtsZeroTtsModelConfig &config);

  // text_ids: int64 (B, L). txt_lengths: int64 (B,).
  // Returns {text_states, text_valid, soa_embed, cross_kv}, in the graph's
  // own output order (see notes/ -- confirmed via onnxruntime introspection
  // during Phase 0/3): text_states (B,L,768) f32 [returned for inspection
  // only, per synthesizer.py], text_valid (B,L) bool, soa_embed
  // (B,Gsoa,768) f32, cross_kv (9,2,B,12,L,64) f32.
  std::vector<Ort::Value> RunTextEncoder(Ort::Value text_ids,
                                         Ort::Value txt_lengths) const;

  // Shared session for both the cold-start "[voice|soa] prefix" call (T =
  // n_voice_queries + 1, packed_kv empty) and the per-frame advance call
  // (T = 1, packed_kv non-empty) -- same 10 named inputs either way, see
  // spec.md §2/§7 and synthesizer.py's _prefix_step_init/_prefix_step_frame.
  // Returns {hidden, new_packed_kv, full_valid}.
  std::vector<Ort::Value> RunPrefixStep(
      Ort::Value external_embed, Ort::Value use_external_embed,
      Ort::Value frame_codes, Ort::Value new_pos, Ort::Value new_valid,
      Ort::Value new_bidirectional, Ort::Value packed_kv,
      Ort::Value past_valid, Ort::Value cross_kv,
      Ort::Value text_valid) const;

  // Returns {is_eoa, codes}. `seen_mask` is taken and returned by value
  // (not mutated in place) since Ort::Value tensors are most simply treated
  // as immutable data holders here; the caller (OfflineTtsZeroTtsImpl) owns
  // updating the seen-codes bookkeeping between calls (mirrors
  // synthesizer.py's _local_decode_frame, which mutates its numpy array
  // in place -- the C++ orchestration layer does the equivalent update on
  // its own raw buffer before rebuilding the next seen_mask tensor).
  std::vector<Ort::Value> RunLocalFrameDecode(
      Ort::Value global_hidden, Ort::Value forbid_eoa,
      Ort::Value text_temperature, Ort::Value text_topk,
      Ort::Value audio_temperature, Ort::Value audio_topk,
      Ort::Value audio_topp, Ort::Value audio_repetition_penalty,
      Ort::Value seen_mask, Ort::Value ctrl_random_u,
      Ort::Value audio_random_u, Ort::Value cfg_scale) const;

  // audio_codes: int32 (B, T, K). audio_code_lengths: int32 (B,).
  // Returns {audio, audio_lengths}.
  std::vector<Ort::Value> RunCodecDecodeFull(
      Ort::Value audio_codes, Ort::Value audio_code_lengths) const;

  // Generic name->value state map, keyed by the *input* names from
  // GetCodecStreamingMeta() -- see notes/phase0-codec-decode-step-io.md.
  // Returns a map keyed by the same *input* names (already remapped from
  // the graph's *_out_* output names) so the caller can feed the returned
  // map straight back into the next call, plus "audio" / "audio_lengths"
  // for the decoded chunk.
  std::unordered_map<std::string, Ort::Value> RunCodecDecodeStep(
      Ort::Value audio_codes, Ort::Value audio_code_lengths,
      std::unordered_map<std::string, Ort::Value> state) const;

  const ZeroTtsCodecStreamingMeta &GetCodecStreamingMeta() const;

  int32_t GetSampleRate() const;
  int32_t GetDModel() const;
  int32_t GetNumCodebooks() const;
  int32_t GetCodebookSize() const;
  int32_t GetNumLayers() const;
  int32_t GetNumHeads() const;
  int32_t GetHeadDim() const;

  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_MODEL_H_
