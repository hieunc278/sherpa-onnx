// sherpa-onnx/csrc/offline-tts-zerotts-tokenizer.h
//
// Copyright (c)  2026
//
// A self-contained tokenizer.json (HF `tokenizers`-format) BPE loader for
// ZeroTTS. See spec.md §5 for why this is a new loader rather than reusing
// an existing sherpa-onnx tokenizer: ZeroTTS's tokenizer.json embeds its own
// vocab/merges/normalizer/pre_tokenizer, unlike the vocab.json+merges.txt
// pattern the other BPE-based models in this codebase expect.
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_TOKENIZER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_TOKENIZER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_onnx {

// Implements, in order (see spec.md §5, confirmed against the real
// tokenizer.json's normalizer/pre_tokenizer/model sections during Phase 1):
//   1. Normalizer: NFC, then collapse runs of whitespace to a single space.
//   2. Pre-tokenizer: split on whitespace (isolated), then punctuation
//      (isolated), then split digit runs into individual-digit tokens.
//   3. Model: BPE merge per pre-token using the embedded vocab/merges (no
//      byte-fallback, no continuing-subword-prefix).
//   4. Wrapping: [<bos>=1, ...body_ids, <eot>=2].
class OfflineTtsZeroTtsTokenizer {
 public:
  explicit OfflineTtsZeroTtsTokenizer(const std::string &tokenizer_json);

  template <typename Manager>
  OfflineTtsZeroTtsTokenizer(Manager *mgr, const std::string &tokenizer_json);

  // NFC-normalize + whitespace-collapse, then BPE-encode -- no BOS/EOT.
  // Exposed for testing / for callers that need to wrap ids themselves
  // (e.g. broadcasting/inserting <en>/<vi> tags around sub-spans).
  std::vector<int32_t> EncodeBody(const std::string &text) const;

  // [<bos>, ...EncodeBody(text), <eot>] -- matches tokenizer.py's __call__.
  std::vector<int32_t> Encode(const std::string &text) const;

  int32_t GetVocabSize() const { return static_cast<int32_t>(vocab_.size()); }

  // Pinned special-token ids (see tokenizer.py's SPECIAL_TOKENS -- "the
  // exported graphs hardcode these ids ... they are part of the weights,
  // not a preference"). Resolved from the loaded tokenizer.json's vocab
  // rather than hardcoded, so a mismatched tokenizer.json is caught instead
  // of silently producing wrong ids.
  int32_t GetSoaId() const { return soa_id_; }
  int32_t GetSlotId() const { return slot_id_; }
  int32_t GetEoaId() const { return eoa_id_; }

  static std::string NormalizeText(const std::string &text);

 private:
  void InitFromContent(const std::string &content);

  std::vector<std::string> PreTokenize(const std::string &text) const;
  std::vector<int32_t> Bpe(const std::string &pre_token) const;

  // token string -> id
  std::unordered_map<std::string, int32_t> vocab_;
  // "left\x00right" -> rank (lower rank merges first); see .cc for why a
  // literal NUL-joined key is safe here.
  std::unordered_map<std::string, int32_t> merge_rank_;

  int32_t bos_id_ = 1;
  int32_t eot_id_ = 2;
  int32_t unk_id_ = 8;
  int32_t soa_id_ = 3;
  int32_t slot_id_ = 4;
  int32_t eoa_id_ = 5;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_TOKENIZER_H_
