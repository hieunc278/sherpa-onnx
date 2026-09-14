// sherpa-onnx/csrc/offline-tts-zerotts-tokenizer-test.cc
//
// Copyright (c)  2026
//
// Proof for plan.md Phase 1: encode a fixed set of Vietnamese/English/mixed
// strings with the C++ loader and assert identical token ID sequences
// against the real Python `tokenizers`-backed reference (fixture generated
// once, offline, via `python3 -c "from zerotts.tokenizer import
// BPEProcessor; ..."` against the actual downloaded tokenizer.json -- see
// notes/phase0-voice-bin-format.md sibling spikes for how the model was
// fetched). The fixture values are pinned as literals below so this test
// doesn't depend on Python being available at C++ test time -- only the
// tokenizer.json asset itself.

#include "sherpa-onnx/csrc/offline-tts-zerotts-tokenizer.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {
struct Case {
  const char *text;
  std::vector<int32_t> expected_ids;
};
}  // namespace

TEST(OfflineTtsZeroTtsTokenizer, TestEncodeAgainstPythonReference) {
  // Matches assets/zerotts/model/ (zeroweight-ai/ZeroTTS), downloaded during
  // Phase 0. This test is a no-op skip (not a failure) if the model assets
  // aren't present on this machine -- consistent with how
  // sentence-piece-tokenizer-test.cc handles optional external test data.
  std::string tokenizer_json;
  static const char *candidates[] = {
      "assets/zerotts/model/tokenizer.json",
      "../assets/zerotts/model/tokenizer.json",
      "../../assets/zerotts/model/tokenizer.json",
      "/mnt/disk2/projects/sherpa-onnx-zero-tts/assets/zerotts/model/"
      "tokenizer.json",
  };
  for (const char *c : candidates) {
    if (FileExists(c)) {
      tokenizer_json = c;
      break;
    }
  }
  if (tokenizer_json.empty()) {
    SHERPA_ONNX_LOGE(
        "No ZeroTTS tokenizer.json found, skipping "
        "TestEncodeAgainstPythonReference(). Download the model into "
        "assets/zerotts/model/ (see CLAUDE.md) to run this test.");
    return;
  }

  OfflineTtsZeroTtsTokenizer tokenizer(tokenizer_json);

  // Fixture generated via:
  //   python3 -c "
  //   from zerotts.tokenizer import BPEProcessor
  //   tok = BPEProcessor('assets/zerotts/model/tokenizer.json')
  //   print([int(x) for x in tok(TEXT)])"
  // against tokenizers==0.23.2 and the real tokenizer.json.
  std::vector<Case> cases = {
      {"Xin chào Việt Nam", {1, 3095, 9, 2903, 9, 1320, 9, 1194, 2}},
      {"Hello world, this is a test.",
      {1, 551, 3576, 9, 470, 334, 21, 9, 365, 9, 282, 9, 74, 9, 1593, 23, 2}},
      {"ZeroTTS chạy trên ONNX Runtime, không cần PyTorch.",
      {1, 7437, 61, 4860, 9, 1912, 9, 683, 9, 5285, 7759, 9, 5915, 583, 21,
        9, 301, 314, 9, 748, 9, 6469, 5132, 276, 23, 2}},
      {"Tôi có 123 con mèo và 45 con chó.",
      {1, 922, 9, 340, 9, 26, 27, 28, 9, 371, 9, 4862, 9, 332, 9, 29, 30, 9,
        371, 9, 4128, 23, 2}},
      {"Ngày 23/8/2024 lúc 15h30.",
      {1, 2562, 9, 27, 28, 24, 33, 24, 27, 25, 27, 29, 9, 1063, 9, 26, 30, 81,
        28, 25, 23, 2}},
      {"  nhiều   khoảng   trắng  ",
      {1, 9, 496, 411, 9, 875, 918, 9, 288, 945, 9, 2}},
      {"Chào bạn! Bạn khỏe không? Tốt, cảm ơn.",
      {1, 5406, 9, 473, 10, 9, 1424, 9, 1974, 9, 301, 314, 40, 9, 5789, 21,
        9, 872, 9, 532, 23, 2}},
      // Decomposed input (e + combining circumflex U+0302 + combining acute
      // U+0301) must NFC-recompose to the precomposed "ế" before BPE, or it
      // tokenizes completely differently -- this is the case
      // notes/phase0-vi-normalizer-scope.md and spec.md §5 flagged as the
      // sharpest correctness risk in the tokenizer.
      {"ế test decomposed vietnamese",
      {1, 202, 9, 1593, 9, 350, 422, 825, 290, 9, 5087, 87, 1390, 78, 2}},
  };

  for (const auto &c : cases) {
    std::vector<int32_t> ids = tokenizer.Encode(c.text);
    EXPECT_EQ(ids, c.expected_ids) << "text: " << c.text;
  }
}

}  // namespace sherpa_onnx
