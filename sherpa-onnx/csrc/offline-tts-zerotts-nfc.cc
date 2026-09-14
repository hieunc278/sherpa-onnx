// sherpa-onnx/csrc/offline-tts-zerotts-nfc.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-tts-zerotts-nfc.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts-zerotts-nfc-table.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

uint8_t CombiningClassOf(char32_t cp) {
  if (cp > 0xFFFF) return 0;
  const uint16_t *begin = kCombiningClassCodepoints;
  const uint16_t *end = begin + kCombiningClassTableSize;
  const uint16_t *it = std::lower_bound(begin, end, static_cast<uint16_t>(cp));
  if (it == end || *it != cp) return 0;
  return kCombiningClassValues[it - begin];
}

// Fully expands cp into its canonical-decomposition sequence (recursively
// resolved already by the table generator), or leaves it as-is if it has no
// canonical decomposition.
void DecomposeCodepoint(char32_t cp, std::vector<char32_t> *out) {
  if (cp <= 0xFFFF) {
    const uint16_t *begin = kNfdCodepoints;
    const uint16_t *end = begin + kNfcNfdTableSize;
    const uint16_t *it =
        std::lower_bound(begin, end, static_cast<uint16_t>(cp));
    if (it != end && *it == cp) {
      int32_t idx = static_cast<int32_t>(it - begin);
      for (int32_t i = kNfdOffsets[idx]; i < kNfdOffsets[idx + 1]; ++i) {
        out->push_back(kNfdPool[i]);
      }
      return;
    }
  }
  out->push_back(cp);
}

// Returns the composed codepoint for (base, combining), or 0 if there is no
// canonical composition pair for them.
char32_t ComposePair(char32_t base, char32_t combining) {
  if (base > 0xFFFF || combining > 0xFFFF) return 0;
  const uint16_t *begin = kComposeBase;
  const uint16_t *end = begin + kComposeTableSize;
  auto range = std::equal_range(begin, end, static_cast<uint16_t>(base));
  for (const uint16_t *it = range.first; it != range.second; ++it) {
    int32_t idx = static_cast<int32_t>(it - kComposeBase);
    if (kComposeCombining[idx] == combining) {
      return kComposeComposed[idx];
    }
  }
  return 0;
}

// Canonically reorders a run of combining marks (stable sort by combining
// class, only ever swapping adjacent characters that are both marks -- i.e.
// both have a non-zero combining class); starters (class 0) are left as
// fixed boundaries. Standard Unicode canonical ordering algorithm.
void CanonicalOrder(std::vector<char32_t> *seq) {
  auto &s = *seq;
  for (size_t i = 1; i < s.size(); ++i) {
    uint8_t ci = CombiningClassOf(s[i]);
    if (ci == 0) continue;
    size_t j = i;
    while (j > 0) {
      uint8_t cj = CombiningClassOf(s[j - 1]);
      if (cj == 0 || cj <= ci) break;
      std::swap(s[j - 1], s[j]);
      --j;
    }
  }
}

// Standard Unicode NFC composition algorithm (UAX #15), restricted to the
// canonical (non-compatibility) composition pairs in our generated table.
std::vector<char32_t> Compose(const std::vector<char32_t> &decomposed) {
  std::vector<char32_t> out;
  out.reserve(decomposed.size());
  int32_t starter_idx = -1;
  int32_t max_class_since_starter = -1;

  for (char32_t cp : decomposed) {
    uint8_t cls = CombiningClassOf(cp);
    bool composed = false;
    if (starter_idx >= 0 && cls != 0 &&
        static_cast<int32_t>(cls) > max_class_since_starter) {
      char32_t merged = ComposePair(out[starter_idx], cp);
      if (merged != 0) {
        out[starter_idx] = merged;
        composed = true;
      }
    }
    if (!composed) {
      out.push_back(cp);
      if (cls == 0) {
        starter_idx = static_cast<int32_t>(out.size()) - 1;
        max_class_since_starter = -1;
      } else {
        max_class_since_starter = std::max(max_class_since_starter,
                                           static_cast<int32_t>(cls));
      }
    }
  }
  return out;
}

}  // namespace

std::string ZeroTtsToNfc(const std::string &text) {
  std::u32string u32 = Utf8ToUtf32(text);

  std::vector<char32_t> decomposed;
  decomposed.reserve(u32.size() * 2);
  for (char32_t cp : u32) {
    DecomposeCodepoint(cp, &decomposed);
  }

  CanonicalOrder(&decomposed);
  std::vector<char32_t> composed = Compose(decomposed);

  std::u32string result(composed.begin(), composed.end());
  return Utf32ToUtf8(result);
}

}  // namespace sherpa_onnx
