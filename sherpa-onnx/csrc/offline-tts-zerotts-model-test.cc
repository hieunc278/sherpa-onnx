// sherpa-onnx/csrc/offline-tts-zerotts-model-test.cc
//
// Copyright (c)  2026
//
// Proof for plan.md Phase 3: call each OfflineTtsZeroTtsModel::Run* method
// in isolation against REAL intermediate tensors captured from a single
// instrumented run of the Python reference implementation (not synthetic
// data), and diff shapes + a numerical tolerance against the Python-side
// output for that same call. This is the check plan.md flags as "the
// highest-leverage debugging investment in the whole plan" -- it catches
// ONNX I/O binding mistakes (wrong tensor name, wrong dtype, wrong shape,
// wrong argument order) before they're buried inside a 1500-step loop.
//
// The capture was produced by a temporary, NOT-committed instrumentation
// script run once against assets/zerotts/model/ (zeroweight-ai/ZeroTTS) with
// a fixed sentence ("Xin chào Việt Nam"), voice="maichi", and a fixed numpy
// RNG seed (1234) so local_frame_decode.onnx's random draws are pinned --
// see cpp/include/zerotts.h upstream, which documents this same property
// ("the random draws are inputs ... so a run is reproducible from a seed").
// Captured tensors live in assets/zerotts/phase3_capture/ (gitignored, raw
// float32/int64/int32/uint8 .bin dumps + a meta.json recording shape/dtype
// per tensor).

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

namespace {

using json = nlohmann::json;

std::string CaptureDir() {
  static const char *candidates[] = {
      "assets/zerotts/phase3_capture",
      "../assets/zerotts/phase3_capture",
      "../../assets/zerotts/phase3_capture",
      "/mnt/disk2/projects/sherpa-onnx-zero-tts/assets/zerotts/"
      "phase3_capture",
  };
  for (const char *c : candidates) {
    if (FileExists(std::string(c) + "/meta.json")) return c;
  }
  return "";
}

std::string ModelDir() {
  static const char *candidates[] = {
      "assets/zerotts/model",
      "../assets/zerotts/model",
      "../../assets/zerotts/model",
      "/mnt/disk2/projects/sherpa-onnx-zero-tts/assets/zerotts/model",
  };
  for (const char *c : candidates) {
    if (FileExists(std::string(c) + "/config.json")) return c;
  }
  return "";
}

struct Captured {
  std::vector<int64_t> shape;
  std::string dtype;
  std::vector<char> raw;
};

Captured LoadCaptured(const std::string &dir, const json &meta,
                     const std::string &name) {
  Captured c;
  const auto &entry = meta.at(name);
  for (const auto &d : entry.at("shape")) c.shape.push_back(d.get<int64_t>());
  c.dtype = entry.at("dtype").get<std::string>();
  c.raw = ReadFile(dir + "/" + name + ".bin");
  return c;
}

Ort::Value ToInt64(Ort::MemoryInfo &mem, const Captured &c) {
  return Ort::Value::CreateTensor<int64_t>(
      mem, reinterpret_cast<int64_t *>(const_cast<char *>(c.raw.data())),
      c.raw.size() / sizeof(int64_t), c.shape.data(), c.shape.size());
}

Ort::Value ToInt32(Ort::MemoryInfo &mem, const Captured &c) {
  return Ort::Value::CreateTensor<int32_t>(
      mem, reinterpret_cast<int32_t *>(const_cast<char *>(c.raw.data())),
      c.raw.size() / sizeof(int32_t), c.shape.data(), c.shape.size());
}

Ort::Value ToFloat(Ort::MemoryInfo &mem, const Captured &c) {
  return Ort::Value::CreateTensor<float>(
      mem, reinterpret_cast<float *>(const_cast<char *>(c.raw.data())),
      c.raw.size() / sizeof(float), c.shape.data(), c.shape.size());
}

Ort::Value ToBool(Ort::MemoryInfo &mem, const Captured &c) {
  // Captured as uint8 0/1; bool is 1 byte on every platform this project
  // targets, so the bytes are already in the right layout.
  return Ort::Value::CreateTensor<bool>(
      mem, reinterpret_cast<bool *>(const_cast<char *>(c.raw.data())),
      c.raw.size(), c.shape.data(), c.shape.size());
}

void ExpectShapeEq(const Ort::Value &v, const std::vector<int64_t> &expected,
                  const char *name) {
  auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(shape, expected) << name;
}

void ExpectFloatClose(const Ort::Value &v, const Captured &expected,
                     const char *name, float atol = 2e-2f) {
  const float *actual = v.GetTensorData<float>();
  const float *want = reinterpret_cast<const float *>(expected.raw.data());
  size_t n = expected.raw.size() / sizeof(float);
  float max_abs_diff = 0.f;
  for (size_t i = 0; i < n; ++i) {
    max_abs_diff = std::max(max_abs_diff, std::fabs(actual[i] - want[i]));
  }
  EXPECT_LE(max_abs_diff, atol)
      << name << ": max abs diff " << max_abs_diff << " over " << n
      << " elements";
}

void ExpectInt32Eq(const Ort::Value &v, const Captured &expected,
                  const char *name) {
  const int32_t *actual = v.GetTensorData<int32_t>();
  const int32_t *want = reinterpret_cast<const int32_t *>(expected.raw.data());
  size_t n = expected.raw.size() / sizeof(int32_t);
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(actual[i], want[i]) << name << " index " << i;
  }
}

}  // namespace

TEST(OfflineTtsZeroTtsModel, TestAgainstPythonReferenceIntermediateTensors) {
  std::string capture_dir = CaptureDir();
  std::string model_dir = ModelDir();
  if (capture_dir.empty() || model_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "No ZeroTTS phase3 capture / model assets found, skipping "
        "TestAgainstPythonReferenceIntermediateTensors(). See "
        "sherpa-onnx/csrc/offline-tts-zerotts-model-test.cc's header "
        "comment for how the capture is regenerated.");
    return;
  }

  std::vector<char> meta_content = ReadFile(capture_dir + "/meta.json");
  json meta = json::parse(std::string(meta_content.data(), meta_content.size()));

  OfflineTtsZeroTtsModelConfig config;
  config.text_encoder = model_dir + "/onnx/text_encoder.onnx";
  config.prefix_step = model_dir + "/onnx/prefix_step.onnx";
  config.local_frame_decode = model_dir + "/onnx/local_frame_decode.onnx";
  config.codec_decode_full =
      model_dir + "/onnx/codec/moss_audio_tokenizer_decode_full.onnx";
  config.codec_decode_step =
      model_dir + "/onnx/codec/moss_audio_tokenizer_decode_step.onnx";
  config.codec_meta = model_dir + "/onnx/codec/codec_browser_onnx_meta.json";
  config.tokenizer = model_dir + "/tokenizer.json";
  config.null_voice_emb = model_dir + "/null_voice_emb.npy";
  config.num_threads = 1;

  OfflineTtsZeroTtsModel model(config);
  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  auto load = [&](const std::string &name) {
    return LoadCaptured(capture_dir, meta, name);
  };

  // ---- 1. text_encoder ----
  {
    Captured text_ids = load("text_ids");
    Captured txt_lengths = load("txt_lengths");
    auto outputs = model.RunTextEncoder(ToInt64(mem, text_ids),
                                        ToInt64(mem, txt_lengths));
    ASSERT_EQ(outputs.size(), 4u);
    Captured exp_text_valid = load("text_valid");
    Captured exp_soa_embed = load("soa_embed");
    Captured exp_cross_kv = load("cross_kv");

    ExpectShapeEq(outputs[2], exp_soa_embed.shape, "soa_embed");
    ExpectFloatClose(outputs[2], exp_soa_embed, "soa_embed");
    ExpectShapeEq(outputs[3], exp_cross_kv.shape, "cross_kv");
    ExpectFloatClose(outputs[3], exp_cross_kv, "cross_kv");
  }

  // ---- 2. prefix_step (cold start) ----
  {
    Captured external_embed = load("external_embed_init");
    Captured use_external_embed = load("use_external_embed_init");
    Captured frame_codes = load("frame_codes_init");
    Captured new_pos = load("new_pos_init");
    Captured new_valid = load("new_valid_init");
    Captured new_bidir = load("new_bidirectional_init");
    Captured packed_kv = load("packed_kv_init");
    Captured past_valid = load("past_valid_init");
    Captured cross_kv = load("cross_kv");
    Captured text_valid = load("text_valid");

    auto outputs = model.RunPrefixStep(
        ToFloat(mem, external_embed), ToBool(mem, use_external_embed),
        ToInt64(mem, frame_codes), ToInt64(mem, new_pos),
        ToBool(mem, new_valid), ToBool(mem, new_bidir),
        ToFloat(mem, packed_kv), ToBool(mem, past_valid),
        ToFloat(mem, cross_kv), ToBool(mem, text_valid));
    ASSERT_EQ(outputs.size(), 3u);

    Captured exp_hidden0 = load("hidden0");
    Captured exp_packed_kv1 = load("packed_kv1");

    ExpectShapeEq(outputs[0], exp_hidden0.shape, "hidden0");
    ExpectFloatClose(outputs[0], exp_hidden0, "hidden0");
    ExpectShapeEq(outputs[1], exp_packed_kv1.shape, "packed_kv1");
    ExpectFloatClose(outputs[1], exp_packed_kv1, "packed_kv1", 5e-2f);
  }

  // ---- 3. local_frame_decode (frame 0, fixed random draws) ----
  {
    Captured h0_last = load("h0_last");
    Captured seen_mask0 = load("seen_mask0");
    Captured ctrl_u = load("ctrl_random_u0");
    Captured audio_u = load("audio_random_u0");

    std::vector<int64_t> scalar_shape = {1};
    bool forbid_eoa_val = true;
    float text_temperature_val = 1.0f;
    int64_t text_topk_val = 50;
    float audio_temperature_val = 0.8f;
    int64_t audio_topk_val = 25;
    float audio_topp_val = 0.95f;
    float audio_rep_pen_val = 1.2f;
    float cfg_scale_val = 1.0f;

    auto outputs = model.RunLocalFrameDecode(
        ToFloat(mem, h0_last),
        Ort::Value::CreateTensor<bool>(mem, &forbid_eoa_val, 1,
                                       scalar_shape.data(), 1),
        Ort::Value::CreateTensor<float>(mem, &text_temperature_val, 1,
                                        scalar_shape.data(), 1),
        Ort::Value::CreateTensor<int64_t>(mem, &text_topk_val, 1,
                                          scalar_shape.data(), 1),
        Ort::Value::CreateTensor<float>(mem, &audio_temperature_val, 1,
                                        scalar_shape.data(), 1),
        Ort::Value::CreateTensor<int64_t>(mem, &audio_topk_val, 1,
                                          scalar_shape.data(), 1),
        Ort::Value::CreateTensor<float>(mem, &audio_topp_val, 1,
                                        scalar_shape.data(), 1),
        Ort::Value::CreateTensor<float>(mem, &audio_rep_pen_val, 1,
                                        scalar_shape.data(), 1),
        ToBool(mem, seen_mask0), ToFloat(mem, ctrl_u), ToFloat(mem, audio_u),
        Ort::Value::CreateTensor<float>(mem, &cfg_scale_val, 1,
                                        scalar_shape.data(), 1));
    ASSERT_EQ(outputs.size(), 2u);

    Captured exp_codes0 = load("codes0");
    // Integer sampling output given identical logits + identical random
    // draws must match exactly, not just "close" -- this is the sharpest
    // possible check that the ONNX I/O binding (tensor names, dtypes,
    // shapes) for this graph is exactly right.
    ExpectShapeEq(outputs[1], exp_codes0.shape, "codes0");
    ExpectInt32Eq(outputs[1], exp_codes0, "codes0");
  }

  // ---- 4. prefix_step (per-frame advance) ----
  {
    Captured frame_codes = load("frame_codes_step0");
    Captured new_pos = load("new_pos_step0");
    Captured packed_kv1 = load("packed_kv1");
    Captured full_valid1 = load("full_valid1");
    Captured cross_kv = load("cross_kv");
    Captured text_valid = load("text_valid");

    std::vector<int64_t> embed_shape = {1, 1, 768};
    std::vector<float> zero_embed(768, 0.f);
    std::vector<int64_t> flag_shape = {1, 1};
    uint8_t false_flag = 0;
    uint8_t true_flag = 1;

    auto outputs = model.RunPrefixStep(
        Ort::Value::CreateTensor<float>(mem, zero_embed.data(),
                                        zero_embed.size(), embed_shape.data(),
                                        embed_shape.size()),
        Ort::Value::CreateTensor<bool>(
            mem, reinterpret_cast<bool *>(&false_flag), 1, flag_shape.data(),
            2),
        ToInt64(mem, frame_codes), ToInt64(mem, new_pos),
        Ort::Value::CreateTensor<bool>(
            mem, reinterpret_cast<bool *>(&true_flag), 1, flag_shape.data(),
            2),
        Ort::Value::CreateTensor<bool>(
            mem, reinterpret_cast<bool *>(&false_flag), 1, flag_shape.data(),
            2),
        ToFloat(mem, packed_kv1), ToBool(mem, full_valid1),
        ToFloat(mem, cross_kv), ToBool(mem, text_valid));
    ASSERT_EQ(outputs.size(), 3u);

    Captured exp_hidden1 = load("hidden1");
    ExpectShapeEq(outputs[0], exp_hidden1.shape, "hidden1");
    ExpectFloatClose(outputs[0], exp_hidden1, "hidden1");
  }

  // ---- 5. codec_decode_full ----
  {
    Captured codes_btk = load("codes_btk");
    Captured codes_lengths = load("codes_lengths");
    auto outputs =
        model.RunCodecDecodeFull(ToInt32(mem, codes_btk),
                                 ToInt32(mem, codes_lengths));
    ASSERT_EQ(outputs.size(), 2u);

    Captured exp_audio = load("audio_full");
    ExpectShapeEq(outputs[0], exp_audio.shape, "audio_full");
    ExpectFloatClose(outputs[0], exp_audio, "audio_full", 5e-2f);
  }
}

}  // namespace sherpa_onnx
