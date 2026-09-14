// sherpa-onnx/csrc/offline-tts-zerotts-tokenizer.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-tts-zerotts-tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-nfc-table.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-nfc.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

using json = nlohmann::json;

std::string ToString(const std::vector<char> &data) {
  if (data.empty()) {
    return "";
  }
  return std::string(data.data(), data.size());
}

std::string ReadTextFile(const std::string &path) {
  return ToString(ReadFile(path));
}

#if __ANDROID_API__ >= 9
std::string ReadTextFile(AAssetManager *mgr, const std::string &path) {
  return ToString(ReadFile(mgr, path));
}
#endif

#if __OHOS__
std::string ReadTextFile(NativeResourceManager *mgr, const std::string &path) {
  return ToString(ReadFile(mgr, path));
}
#endif

// ---------------------------------------------------------------------
// Pre-tokenizer primitives
// ---------------------------------------------------------------------

bool IsAsciiPunctuation(char32_t cp) {
  // Mirrors Rust's `char::is_ascii_punctuation()`: any of
  // !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  if (cp > 0x7F) return false;
  char c = static_cast<char>(cp);
  return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
        (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

bool IsPunctuation(char32_t cp) {
  if (cp < 0x80) return IsAsciiPunctuation(cp);
  if (cp > 0xFFFF) return false;
  const uint16_t *begin = kPunctuationCodepoints;
  const uint16_t *end = begin + kPunctuationTableSize;
  return std::binary_search(begin, end, static_cast<uint16_t>(cp));
}

bool IsDigit(char32_t cp) { return cp >= U'0' && cp <= U'9'; }

bool IsWhitespace(char32_t cp) {
  return cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r' ||
        cp == U'\f' || cp == U'\v';
}

// tokenizer.json's pre_tokenizer is Sequence[
//   Split(Regex "\s+", Isolated),
//   Punctuation(Isolated),
//   Digits(individual_digits=true)]
// i.e. split into whitespace-run / non-whitespace pieces (each kept, not
// dropped), then further split any punctuation character out on its own,
// then further split any digit out on its own. Order matters: Punctuation
// runs before Digits, both operate on the pieces the previous step produced.
std::vector<std::string> PreTokenizeImpl(const std::string &text) {
  std::u32string u32 = Utf8ToUtf32(text);

  // Step 1: split on whitespace runs, isolated (each run and each non-run
  // becomes its own piece).
  std::vector<std::u32string> pieces;
  {
    size_t i = 0;
    while (i < u32.size()) {
      bool ws = IsWhitespace(u32[i]);
      size_t j = i;
      while (j < u32.size() && IsWhitespace(u32[j]) == ws) ++j;
      pieces.emplace_back(u32.substr(i, j - i));
      i = j;
    }
  }

  // Step 2: split punctuation out, isolated.
  std::vector<std::u32string> after_punct;
  for (const auto &piece : pieces) {
    size_t i = 0;
    while (i < piece.size()) {
      if (IsPunctuation(piece[i])) {
        after_punct.emplace_back(1, piece[i]);
        ++i;
        continue;
      }
      size_t j = i;
      while (j < piece.size() && !IsPunctuation(piece[j])) ++j;
      after_punct.emplace_back(piece.substr(i, j - i));
      i = j;
    }
  }

  // Step 3: split digits out, one digit per piece.
  std::vector<std::string> out;
  for (const auto &piece : after_punct) {
    size_t i = 0;
    while (i < piece.size()) {
      if (IsDigit(piece[i])) {
        out.push_back(Utf32ToUtf8(piece.substr(i, 1)));
        ++i;
        continue;
      }
      size_t j = i;
      while (j < piece.size() && !IsDigit(piece[j])) ++j;
      out.push_back(Utf32ToUtf8(piece.substr(i, j - i)));
      i = j;
    }
  }

  // tokenizers' Split/Punctuation/Digits steps drop zero-length pieces.
  std::vector<std::string> result;
  result.reserve(out.size());
  for (auto &s : out) {
    if (!s.empty()) result.push_back(std::move(s));
  }
  return result;
}

// Explodes a UTF-8 string into a list of single-codepoint UTF-8 strings,
// with NO merging/filtering. Deliberately not sherpa-onnx's SplitUtf8(),
// which is a lexicon word-splitting helper that (a) drops whitespace
// characters outright and (b) fuses runs of ASCII letters back into whole
// "words" via MergeCharactersIntoWords() -- exactly the opposite of what a
// standard (GPT-2-style) BPE base-symbol split needs: every codepoint kept,
// nothing pre-merged, so the rank-based merge loop below is the only thing
// doing any merging.
std::vector<std::string> SplitIntoCodepoints(const std::string &text) {
  std::u32string u32 = Utf8ToUtf32(text);
  std::vector<std::string> out;
  out.reserve(u32.size());
  for (char32_t cp : u32) {
    out.push_back(Utf32ToUtf8(cp));
  }
  return out;
}

std::string MergeKey(const std::string &a, const std::string &b) {
  std::string key;
  key.reserve(a.size() + b.size() + 1);
  key.append(a);
  key.push_back('\0');  // std::string may contain embedded NULs safely
  key.append(b);
  return key;
}

}  // namespace

std::string OfflineTtsZeroTtsTokenizer::NormalizeText(
    const std::string &text) {
  std::string nfc = ZeroTtsToNfc(text);

  // Collapse runs of whitespace to a single ASCII space (tokenizer.json:
  // Replace(Regex "\s+", " ")).
  std::u32string u32 = Utf8ToUtf32(nfc);
  std::u32string collapsed;
  collapsed.reserve(u32.size());
  size_t i = 0;
  while (i < u32.size()) {
    if (IsWhitespace(u32[i])) {
      collapsed.push_back(U' ');
      while (i < u32.size() && IsWhitespace(u32[i])) ++i;
    } else {
      collapsed.push_back(u32[i]);
      ++i;
    }
  }
  return Utf32ToUtf8(collapsed);
}

std::vector<std::string> OfflineTtsZeroTtsTokenizer::PreTokenize(
    const std::string &text) const {
  return PreTokenizeImpl(text);
}

std::vector<int32_t> OfflineTtsZeroTtsTokenizer::Bpe(
    const std::string &pre_token) const {
  std::vector<std::string> symbols = SplitIntoCodepoints(pre_token);
  if (symbols.empty()) return {};

  while (symbols.size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_pos = symbols.size();  // sentinel: none found
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = merge_rank_.find(MergeKey(symbols[i], symbols[i + 1]));
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_pos = i;
      }
    }
    if (best_pos == symbols.size()) break;  // no mergeable pair left

    // Merge every non-overlapping occurrence of this exact (left, right)
    // pair in this pass -- matches the standard BPE merge loop (GPT-2 /
    // HF tokenizers): find the single best-ranked pair, merge all of its
    // occurrences, then re-scan from scratch.
    const std::string &left = symbols[best_pos];
    const std::string &right = symbols[best_pos + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    size_t i = 0;
    while (i < symbols.size()) {
      if (i + 1 < symbols.size() && symbols[i] == left &&
          symbols[i + 1] == right) {
        merged.push_back(left + right);
        i += 2;
      } else {
        merged.push_back(symbols[i]);
        ++i;
      }
    }
    symbols.swap(merged);
  }

  std::vector<int32_t> ids;
  ids.reserve(symbols.size());
  for (const auto &s : symbols) {
    auto it = vocab_.find(s);
    ids.push_back(it != vocab_.end() ? it->second : unk_id_);
  }
  return ids;
}

std::vector<int32_t> OfflineTtsZeroTtsTokenizer::EncodeBody(
    const std::string &text) const {
  std::string normalized = NormalizeText(text);
  std::vector<int32_t> ids;
  for (const auto &pre_token : PreTokenize(normalized)) {
    std::vector<int32_t> piece_ids = Bpe(pre_token);
    ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
  }
  return ids;
}

std::vector<int32_t> OfflineTtsZeroTtsTokenizer::Encode(
    const std::string &text) const {
  std::vector<int32_t> ids;
  ids.reserve(2);
  ids.push_back(bos_id_);
  std::vector<int32_t> body = EncodeBody(text);
  ids.insert(ids.end(), body.begin(), body.end());
  ids.push_back(eot_id_);
  return ids;
}

void OfflineTtsZeroTtsTokenizer::InitFromContent(const std::string &content) {
  if (content.empty()) {
    SHERPA_ONNX_LOGE("Failed to read ZeroTTS tokenizer.json (empty content)");
    SHERPA_ONNX_EXIT(-1);
  }

  json tok;
  try {
    tok = json::parse(content);
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("Failed to parse ZeroTTS tokenizer.json: %s", e.what());
    SHERPA_ONNX_EXIT(-1);
  }

  if (!tok.contains("model") || !tok["model"].contains("vocab")) {
    SHERPA_ONNX_LOGE(
        "ZeroTTS tokenizer.json has no model.vocab -- not a valid "
        "self-contained tokenizer.json");
    SHERPA_ONNX_EXIT(-1);
  }

  vocab_.clear();
  for (auto it = tok["model"]["vocab"].begin();
      it != tok["model"]["vocab"].end(); ++it) {
    vocab_[it.key()] = it.value().get<int32_t>();
  }

  merge_rank_.clear();
  if (tok["model"].contains("merges")) {
    const auto &merges = tok["model"]["merges"];
    int32_t rank = 0;
    for (const auto &m : merges) {
      // tokenizer.json merges are a list of [left, right] pairs (this
      // release's format; the older "left right" single-string format is
      // not used here, confirmed against the real tokenizer.json).
      std::string left = m.at(0).get<std::string>();
      std::string right = m.at(1).get<std::string>();
      merge_rank_[MergeKey(left, right)] = rank++;
    }
  }

  auto find_special = [&](const char *name, int32_t fallback) -> int32_t {
    auto it = vocab_.find(name);
    return it != vocab_.end() ? it->second : fallback;
  };
  bos_id_ = find_special("<bos>", 1);
  eot_id_ = find_special("<eot>", 2);
  unk_id_ = find_special("<unk>", 8);
  soa_id_ = find_special("<soa>", 3);
  slot_id_ = find_special("<slot>", 4);
  eoa_id_ = find_special("<eoa>", 5);
}

OfflineTtsZeroTtsTokenizer::OfflineTtsZeroTtsTokenizer(
    const std::string &tokenizer_json) {
  InitFromContent(ReadTextFile(tokenizer_json));
}

template <typename Manager>
OfflineTtsZeroTtsTokenizer::OfflineTtsZeroTtsTokenizer(
    Manager *mgr, const std::string &tokenizer_json) {
  InitFromContent(ReadTextFile(mgr, tokenizer_json));
}

#if __ANDROID_API__ >= 9
template OfflineTtsZeroTtsTokenizer::OfflineTtsZeroTtsTokenizer(
    AAssetManager *mgr, const std::string &tokenizer_json);
#endif

#if __OHOS__
template OfflineTtsZeroTtsTokenizer::OfflineTtsZeroTtsTokenizer(
    NativeResourceManager *mgr, const std::string &tokenizer_json);
#endif

}  // namespace sherpa_onnx
