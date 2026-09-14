// sherpa-onnx/csrc/offline-tts-zerotts-voice-test.cc
//
// Copyright (c)  2026
//
// Proof for plan.md Phase 2: load all 8 real bundled voice packs plus
// null_voice_emb.npy/silence_frame.npy, assert expected shapes, and confirm
// OfflineTtsZeroTtsModelConfig::Validate() rejects missing/malformed paths.
// No inference happens in this test -- purely "do the files parse
// correctly," per Phase 2's scope.

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-npy.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-voice.h"

namespace sherpa_onnx {

namespace {
std::string ModelDir() {
  // Tried in order: cwd-relative (ctest run from the build dir or its bin/
  // subdir), and the fixed absolute path this repo's assets are downloaded
  // to (see CLAUDE.md) -- mirrors how sentence-piece-tokenizer-test.cc
  // hardcodes /tmp/sherpa-onnx-test-data for the same "optional external
  // test data" reason.
  static const char *candidates[] = {
      "assets/zerotts/model",
      "../assets/zerotts/model",
      "../../assets/zerotts/model",
      "/mnt/disk2/projects/sherpa-onnx-zero-tts/assets/zerotts/model",
  };
  for (const char *c : candidates) {
    if (FileExists(std::string(c) + "/config.json")) {
      return c;
    }
  }
  return "";
}
}  // namespace

TEST(OfflineTtsZeroTtsVoice, TestLoadAllBundledVoices) {
  std::string dir = ModelDir();
  if (dir.empty()) {
    SHERPA_ONNX_LOGE(
        "No ZeroTTS model assets found, skipping "
        "TestLoadAllBundledVoices(). Download the model into "
        "assets/zerotts/model/ (see CLAUDE.md) to run this test.");
    return;
  }

  constexpr int32_t kDModel = 768;
  constexpr int32_t kExpectedQueries = 10;

  std::vector<std::string> voices = {"baotrang", "giahuy",   "hamy",
                                     "huuduc",   "kimoanh",  "maichi",
                                     "quangminh", "tiendat"};

  for (const auto &name : voices) {
    OfflineTtsZeroTtsVoice voice;
    bool ok = LoadOfflineTtsZeroTtsVoice(dir + "/voices/" + name, kDModel,
                                        &voice);
    EXPECT_TRUE(ok) << "failed to load voice: " << name;
    if (!ok) continue;

    EXPECT_EQ(voice.n_voice_queries, kExpectedQueries) << name;
    EXPECT_EQ(voice.d_model, kDModel) << name;
    EXPECT_EQ(static_cast<int32_t>(voice.emb.size()),
             kExpectedQueries * kDModel)
        << name;
    EXPECT_FALSE(voice.display_name.empty()) << name;
  }
}

TEST(OfflineTtsZeroTtsVoice, TestLoadNullVoiceEmbAndSilenceFrame) {
  std::string dir = ModelDir();
  if (dir.empty()) {
    SHERPA_ONNX_LOGE(
        "No ZeroTTS model assets found, skipping "
        "TestLoadNullVoiceEmbAndSilenceFrame().");
    return;
  }

  ZeroTtsNpyArray null_voice_emb;
  ASSERT_TRUE(
      ReadZeroTtsNpy(dir + "/null_voice_emb.npy", &null_voice_emb));
  EXPECT_EQ(null_voice_emb.dtype, "<f4");
  ASSERT_EQ(null_voice_emb.shape.size(), 3u);
  EXPECT_EQ(null_voice_emb.shape[0], 1);
  EXPECT_EQ(null_voice_emb.shape[1], 10);
  EXPECT_EQ(null_voice_emb.shape[2], 768);
  EXPECT_EQ(static_cast<int64_t>(null_voice_emb.f32.size()),
           null_voice_emb.NumElements());

  ZeroTtsNpyArray silence_frame;
  ASSERT_TRUE(ReadZeroTtsNpy(dir + "/silence_frame.npy", &silence_frame));
  EXPECT_EQ(silence_frame.dtype, "<i8");
  ASSERT_EQ(silence_frame.shape.size(), 2u);
  EXPECT_EQ(silence_frame.shape[0], 1);
  EXPECT_EQ(silence_frame.shape[1], 16);
  EXPECT_EQ(static_cast<int64_t>(silence_frame.i64.size()),
           silence_frame.NumElements());
}

TEST(OfflineTtsZeroTtsModelConfig, TestValidateRejectsMissingPaths) {
  OfflineTtsZeroTtsModelConfig config;
  // Completely empty config must fail.
  EXPECT_FALSE(config.Validate());

  config.text_encoder = "/nonexistent/text_encoder.onnx";
  config.prefix_step = "/nonexistent/prefix_step.onnx";
  config.local_frame_decode = "/nonexistent/local_frame_decode.onnx";
  config.codec_decode_full = "/nonexistent/decode_full.onnx";
  config.codec_decode_step = "/nonexistent/decode_step.onnx";
  config.codec_meta = "/nonexistent/meta.json";
  config.tokenizer = "/nonexistent/tokenizer.json";
  config.null_voice_emb = "/nonexistent/null_voice_emb.npy";
  // Every path is non-empty but points at a file that doesn't exist ->
  // Validate() must still fail.
  EXPECT_FALSE(config.Validate());
}

TEST(OfflineTtsZeroTtsModelConfig, TestValidateAcceptsRealAssets) {
  std::string dir = ModelDir();
  if (dir.empty()) {
    SHERPA_ONNX_LOGE(
        "No ZeroTTS model assets found, skipping "
        "TestValidateAcceptsRealAssets().");
    return;
  }

  OfflineTtsZeroTtsModelConfig config;
  config.text_encoder = dir + "/onnx/text_encoder.onnx";
  config.prefix_step = dir + "/onnx/prefix_step.onnx";
  config.local_frame_decode = dir + "/onnx/local_frame_decode.onnx";
  config.codec_decode_full =
      dir + "/onnx/codec/moss_audio_tokenizer_decode_full.onnx";
  config.codec_decode_step =
      dir + "/onnx/codec/moss_audio_tokenizer_decode_step.onnx";
  config.codec_meta = dir + "/onnx/codec/codec_browser_onnx_meta.json";
  config.tokenizer = dir + "/tokenizer.json";
  config.null_voice_emb = dir + "/null_voice_emb.npy";
  config.silence_frame = dir + "/silence_frame.npy";
  config.voice = dir + "/voices/maichi";

  EXPECT_TRUE(config.Validate());
}

}  // namespace sherpa_onnx
