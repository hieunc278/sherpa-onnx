// sherpa-onnx/csrc/offline-tts-zerotts-voice.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-tts-zerotts-voice.h"

#include <cstring>
#include <string>
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

namespace sherpa_onnx {

namespace {

using json = nlohmann::json;

std::string JoinPath(const std::string &dir, const std::string &name) {
  if (!dir.empty() && dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

bool FinishLoad(const std::vector<char> &bin_content,
                const std::vector<char> &meta_content,
                const std::string &voice_dir, int32_t expected_d_model,
                OfflineTtsZeroTtsVoice *out) {
  if (bin_content.empty()) {
    SHERPA_ONNX_LOGE("Failed to read voice.bin from %s", voice_dir.c_str());
    return false;
  }
  if (bin_content.size() % sizeof(float) != 0) {
    SHERPA_ONNX_LOGE(
        "%s/voice.bin size (%zu bytes) is not a multiple of 4 -- not a "
        "raw float32 buffer",
        voice_dir.c_str(), bin_content.size());
    return false;
  }

  int32_t n_voice_queries = 0;
  std::string display_name;
  if (!meta_content.empty()) {
    try {
      json meta = json::parse(
          std::string(meta_content.data(), meta_content.size()));
      if (meta.contains("n_voice_queries")) {
        n_voice_queries = meta["n_voice_queries"].get<int32_t>();
      }
      if (meta.contains("display_name")) {
        display_name = meta["display_name"].get<std::string>();
      } else if (meta.contains("name")) {
        display_name = meta["name"].get<std::string>();
      }
    } catch (const std::exception &e) {
      SHERPA_ONNX_LOGE("%s/meta.json failed to parse: %s", voice_dir.c_str(),
                       e.what());
      return false;
    }
  }

  int64_t total_floats =
      static_cast<int64_t>(bin_content.size() / sizeof(float));

  if (n_voice_queries == 0) {
    // No meta.json / no n_voice_queries field: recover it from the file
    // size and the model's known d_model (see spec.md §6 /
    // notes/phase0-voice-bin-format.md: voice.bin is exactly
    // n_voice_queries * d_model float32s, no header).
    if (expected_d_model <= 0 || total_floats % expected_d_model != 0) {
      SHERPA_ONNX_LOGE(
          "%s/voice.bin has %lld float32 values, which is not a multiple "
          "of d_model=%d, and no meta.json n_voice_queries was found -- "
          "cannot recover the voice shape",
          voice_dir.c_str(), static_cast<long long>(total_floats),
          expected_d_model);
      return false;
    }
    n_voice_queries = static_cast<int32_t>(total_floats / expected_d_model);
  }

  int64_t expected_total =
      static_cast<int64_t>(n_voice_queries) * expected_d_model;
  if (expected_total != total_floats) {
    SHERPA_ONNX_LOGE(
        "%s/voice.bin: size mismatch. meta.json says n_voice_queries=%d, "
        "model d_model=%d (expects %lld float32s), but the file has %lld",
        voice_dir.c_str(), n_voice_queries, expected_d_model,
        static_cast<long long>(expected_total),
        static_cast<long long>(total_floats));
    return false;
  }

  out->emb.resize(total_floats);
  std::memcpy(out->emb.data(), bin_content.data(), bin_content.size());
  out->n_voice_queries = n_voice_queries;
  out->d_model = expected_d_model;
  out->display_name = display_name;

  return true;
}

}  // namespace

bool LoadOfflineTtsZeroTtsVoice(const std::string &voice_dir,
                                int32_t expected_d_model,
                                OfflineTtsZeroTtsVoice *out) {
  std::vector<char> bin_content = ReadFile(JoinPath(voice_dir, "voice.bin"));
  std::vector<char> meta_content = ReadFile(JoinPath(voice_dir, "meta.json"));
  bool ok =
      FinishLoad(bin_content, meta_content, voice_dir, expected_d_model, out);
  if (ok) {
    size_t slash = voice_dir.find_last_of('/');
    out->name = (slash == std::string::npos) ? voice_dir
                                             : voice_dir.substr(slash + 1);
    if (out->name.empty() && slash != std::string::npos) {
      // trailing slash: retry without it
      std::string trimmed = voice_dir.substr(0, voice_dir.size() - 1);
      size_t s2 = trimmed.find_last_of('/');
      out->name = (s2 == std::string::npos) ? trimmed : trimmed.substr(s2 + 1);
    }
  }
  return ok;
}

template <typename Manager>
bool LoadOfflineTtsZeroTtsVoice(Manager *mgr, const std::string &voice_dir,
                                int32_t expected_d_model,
                                OfflineTtsZeroTtsVoice *out) {
  std::vector<char> bin_content =
      ReadFile(mgr, JoinPath(voice_dir, "voice.bin"));
  std::vector<char> meta_content =
      ReadFile(mgr, JoinPath(voice_dir, "meta.json"));
  return FinishLoad(bin_content, meta_content, voice_dir, expected_d_model,
                    out);
}

#if __ANDROID_API__ >= 9
template bool LoadOfflineTtsZeroTtsVoice(AAssetManager *mgr,
                                         const std::string &voice_dir,
                                         int32_t expected_d_model,
                                         OfflineTtsZeroTtsVoice *out);
#endif

#if __OHOS__
template bool LoadOfflineTtsZeroTtsVoice(NativeResourceManager *mgr,
                                         const std::string &voice_dir,
                                         int32_t expected_d_model,
                                         OfflineTtsZeroTtsVoice *out);
#endif

}  // namespace sherpa_onnx
