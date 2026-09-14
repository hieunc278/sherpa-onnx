// sherpa-onnx/csrc/offline-tts-zerotts-model.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-tts-zerotts-model.h"

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
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {
using json = nlohmann::json;

std::vector<int64_t> JsonIntArray(const json &arr) {
  std::vector<int64_t> out;
  for (const auto &v : arr) out.push_back(v.get<int64_t>());
  return out;
}

void ParseCodecMeta(const std::string &content, const std::string &path,
                   ZeroTtsCodecStreamingMeta *out) {
  json meta;
  try {
    meta = json::parse(content);
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("Failed to parse %s: %s", path.c_str(), e.what());
    SHERPA_ONNX_EXIT(-1);
  }

  if (meta.contains("codec_config")) {
    const auto &cfg = meta["codec_config"];
    if (cfg.contains("sample_rate"))
      out->sample_rate = cfg["sample_rate"].get<int32_t>();
    if (cfg.contains("channels"))
      out->num_channels = cfg["channels"].get<int32_t>();
    if (cfg.contains("downsample_rate"))
      out->downsample_rate = cfg["downsample_rate"].get<int32_t>();
    if (cfg.contains("num_quantizers"))
      out->num_quantizers = cfg["num_quantizers"].get<int32_t>();
  }

  if (!meta.contains("streaming_decode")) {
    SHERPA_ONNX_LOGE(
        "%s has no streaming_decode section -- cannot drive "
        "decode_step.onnx",
        path.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  const auto &streaming = meta["streaming_decode"];

  out->transformer_offsets.clear();
  for (const auto &spec : streaming.value("transformer_offsets", json::array())) {
    ZeroTtsTransformerOffsetSpec s;
    s.input_name = spec.at("input_name").get<std::string>();
    s.output_name = spec.at("output_name").get<std::string>();
    s.shape = JsonIntArray(spec.at("shape"));
    out->transformer_offsets.push_back(std::move(s));
  }

  out->attention_caches.clear();
  for (const auto &spec : streaming.value("attention_caches", json::array())) {
    ZeroTtsAttentionCacheSpec s;
    s.offset_input_name = spec.at("offset_input_name").get<std::string>();
    s.offset_output_name = spec.at("offset_output_name").get<std::string>();
    s.cached_keys_input_name =
        spec.at("cached_keys_input_name").get<std::string>();
    s.cached_keys_output_name =
        spec.at("cached_keys_output_name").get<std::string>();
    s.cached_values_input_name =
        spec.at("cached_values_input_name").get<std::string>();
    s.cached_values_output_name =
        spec.at("cached_values_output_name").get<std::string>();
    s.cached_positions_input_name =
        spec.at("cached_positions_input_name").get<std::string>();
    s.cached_positions_output_name =
        spec.at("cached_positions_output_name").get<std::string>();
    s.offset_shape = JsonIntArray(spec.at("offset_shape"));
    s.cache_shape = JsonIntArray(spec.at("cache_shape"));
    s.positions_shape = JsonIntArray(spec.at("positions_shape"));
    out->attention_caches.push_back(std::move(s));
  }
}

}  // namespace

class OfflineTtsZeroTtsModel::Impl {
 public:
  explicit Impl(const OfflineTtsZeroTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    text_encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.text_encoder), sess_opts_);
    InitNames(text_encoder_sess_.get(), &text_encoder_in_, &text_encoder_out_);

    prefix_step_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.prefix_step), sess_opts_);
    InitNames(prefix_step_sess_.get(), &prefix_step_in_, &prefix_step_out_);

    local_frame_decode_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.local_frame_decode),
        sess_opts_);
    InitNames(local_frame_decode_sess_.get(), &local_frame_decode_in_,
             &local_frame_decode_out_);

    codec_decode_full_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.codec_decode_full), sess_opts_);
    InitNames(codec_decode_full_sess_.get(), &codec_decode_full_in_,
             &codec_decode_full_out_);

    codec_decode_step_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.codec_decode_step), sess_opts_);
    InitNames(codec_decode_step_sess_.get(), &codec_decode_step_in_,
             &codec_decode_step_out_);

    std::string meta_content = ReadFileToString(config_.codec_meta);
    ParseCodecMeta(meta_content, config_.codec_meta, &codec_meta_);

    if (config_.debug) {
      PrintDebugInfo();
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsZeroTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto load = [&](const std::string &path) {
      auto buf = ReadFile(mgr, path);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read ZeroTTS model file: %s",
                         path.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      return std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                            sess_opts_);
    };

    text_encoder_sess_ = load(config_.text_encoder);
    InitNames(text_encoder_sess_.get(), &text_encoder_in_, &text_encoder_out_);

    prefix_step_sess_ = load(config_.prefix_step);
    InitNames(prefix_step_sess_.get(), &prefix_step_in_, &prefix_step_out_);

    local_frame_decode_sess_ = load(config_.local_frame_decode);
    InitNames(local_frame_decode_sess_.get(), &local_frame_decode_in_,
             &local_frame_decode_out_);

    // NOTE (Phase 8 follow-up): the two codec graphs reference an external
    // .data file (moss_audio_tokenizer_decode_shared.data) sitting next to
    // them on disk; loading them from an in-memory buffer (the Android
    // asset-manager path) does not resolve that automatically the way
    // constructing from a file path does. This is a known gap flagged for
    // Phase 8's Android build rather than papered over here -- see plan.md.
    codec_decode_full_sess_ = load(config_.codec_decode_full);
    InitNames(codec_decode_full_sess_.get(), &codec_decode_full_in_,
             &codec_decode_full_out_);

    codec_decode_step_sess_ = load(config_.codec_decode_step);
    InitNames(codec_decode_step_sess_.get(), &codec_decode_step_in_,
             &codec_decode_step_out_);

    auto meta_buf = ReadFile(mgr, config_.codec_meta);
    std::string meta_content(meta_buf.data(), meta_buf.size());
    ParseCodecMeta(meta_content, config_.codec_meta, &codec_meta_);

    if (config_.debug) {
      PrintDebugInfo();
    }
  }

  std::vector<Ort::Value> RunTextEncoder(Ort::Value text_ids,
                                         Ort::Value txt_lengths) {
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(text_ids));
    inputs.push_back(std::move(txt_lengths));
    return text_encoder_sess_->Run(
        Ort::RunOptions{nullptr}, text_encoder_in_.ptrs.data(), inputs.data(),
        inputs.size(), text_encoder_out_.ptrs.data(),
        text_encoder_out_.ptrs.size());
  }

  std::vector<Ort::Value> RunPrefixStep(
      Ort::Value external_embed, Ort::Value use_external_embed,
      Ort::Value frame_codes, Ort::Value new_pos, Ort::Value new_valid,
      Ort::Value new_bidirectional, Ort::Value packed_kv,
      Ort::Value past_valid, Ort::Value cross_kv, Ort::Value text_valid) {
    // Must be built in the same order as prefix_step_in_.names (confirmed
    // during Phase 0 via onnxruntime introspection): external_embed,
    // use_external_embed, frame_codes, new_pos, new_valid, packed_kv,
    // past_valid, cross_kv, text_valid, new_bidirectional.
    std::unordered_map<std::string, Ort::Value> by_name;
    by_name.emplace("external_embed", std::move(external_embed));
    by_name.emplace("use_external_embed", std::move(use_external_embed));
    by_name.emplace("frame_codes", std::move(frame_codes));
    by_name.emplace("new_pos", std::move(new_pos));
    by_name.emplace("new_valid", std::move(new_valid));
    by_name.emplace("new_bidirectional", std::move(new_bidirectional));
    by_name.emplace("packed_kv", std::move(packed_kv));
    by_name.emplace("past_valid", std::move(past_valid));
    by_name.emplace("cross_kv", std::move(cross_kv));
    by_name.emplace("text_valid", std::move(text_valid));

    std::vector<Ort::Value> inputs;
    inputs.reserve(prefix_step_in_.names.size());
    for (const auto &name : prefix_step_in_.names) {
      auto it = by_name.find(name);
      if (it == by_name.end()) {
        SHERPA_ONNX_LOGE(
            "prefix_step.onnx expects input '%s', which was not provided",
            name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      inputs.push_back(std::move(it->second));
    }

    return prefix_step_sess_->Run(
        Ort::RunOptions{nullptr}, prefix_step_in_.ptrs.data(), inputs.data(),
        inputs.size(), prefix_step_out_.ptrs.data(),
        prefix_step_out_.ptrs.size());
  }

  std::vector<Ort::Value> RunLocalFrameDecode(
      Ort::Value global_hidden, Ort::Value forbid_eoa,
      Ort::Value text_temperature, Ort::Value text_topk,
      Ort::Value audio_temperature, Ort::Value audio_topk,
      Ort::Value audio_topp, Ort::Value audio_repetition_penalty,
      Ort::Value seen_mask, Ort::Value ctrl_random_u,
      Ort::Value audio_random_u, Ort::Value cfg_scale) {
    std::unordered_map<std::string, Ort::Value> by_name;
    by_name.emplace("global_hidden", std::move(global_hidden));
    by_name.emplace("forbid_eoa", std::move(forbid_eoa));
    by_name.emplace("text_temperature", std::move(text_temperature));
    by_name.emplace("text_topk", std::move(text_topk));
    by_name.emplace("audio_temperature", std::move(audio_temperature));
    by_name.emplace("audio_topk", std::move(audio_topk));
    by_name.emplace("audio_topp", std::move(audio_topp));
    by_name.emplace("audio_repetition_penalty",
                    std::move(audio_repetition_penalty));
    by_name.emplace("seen_mask", std::move(seen_mask));
    by_name.emplace("ctrl_random_u", std::move(ctrl_random_u));
    by_name.emplace("audio_random_u", std::move(audio_random_u));
    by_name.emplace("cfg_scale", std::move(cfg_scale));

    std::vector<Ort::Value> inputs;
    inputs.reserve(local_frame_decode_in_.names.size());
    for (const auto &name : local_frame_decode_in_.names) {
      auto it = by_name.find(name);
      if (it == by_name.end()) {
        SHERPA_ONNX_LOGE(
            "local_frame_decode.onnx expects input '%s', which was not "
            "provided",
            name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      inputs.push_back(std::move(it->second));
    }

    return local_frame_decode_sess_->Run(
        Ort::RunOptions{nullptr}, local_frame_decode_in_.ptrs.data(),
        inputs.data(), inputs.size(), local_frame_decode_out_.ptrs.data(),
        local_frame_decode_out_.ptrs.size());
  }

  std::vector<Ort::Value> RunCodecDecodeFull(Ort::Value audio_codes,
                                             Ort::Value audio_code_lengths) {
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(audio_codes));
    inputs.push_back(std::move(audio_code_lengths));
    return codec_decode_full_sess_->Run(
        Ort::RunOptions{nullptr}, codec_decode_full_in_.ptrs.data(),
        inputs.data(), inputs.size(), codec_decode_full_out_.ptrs.data(),
        codec_decode_full_out_.ptrs.size());
  }

  std::unordered_map<std::string, Ort::Value> RunCodecDecodeStep(
      Ort::Value audio_codes, Ort::Value audio_code_lengths,
      std::unordered_map<std::string, Ort::Value> state) {
    state.emplace("audio_codes", std::move(audio_codes));
    state.emplace("audio_code_lengths", std::move(audio_code_lengths));

    std::vector<Ort::Value> inputs;
    inputs.reserve(codec_decode_step_in_.names.size());
    for (const auto &name : codec_decode_step_in_.names) {
      auto it = state.find(name);
      if (it == state.end()) {
        SHERPA_ONNX_LOGE(
            "decode_step.onnx expects input '%s', which was not provided",
            name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      inputs.push_back(std::move(it->second));
    }

    std::vector<Ort::Value> outputs = codec_decode_step_sess_->Run(
        Ort::RunOptions{nullptr}, codec_decode_step_in_.ptrs.data(),
        inputs.data(), inputs.size(), codec_decode_step_out_.ptrs.data(),
        codec_decode_step_out_.ptrs.size());

    std::unordered_map<std::string, Ort::Value> named_outputs;
    for (size_t i = 0; i < codec_decode_step_out_.names.size(); ++i) {
      named_outputs.emplace(codec_decode_step_out_.names[i],
                           std::move(outputs[i]));
    }

    // Rebuild the next-call state map, keyed by *input* names, from the
    // *_out_* output names (mirrors codec.py's MossStreamingDecoder.
    // decode_chunk output->input rebinding loop).
    std::unordered_map<std::string, Ort::Value> next_state;
    for (const auto &spec : codec_meta_.transformer_offsets) {
      next_state.emplace(spec.input_name,
                        std::move(named_outputs.at(spec.output_name)));
    }
    for (const auto &spec : codec_meta_.attention_caches) {
      next_state.emplace(spec.offset_input_name,
                        std::move(named_outputs.at(spec.offset_output_name)));
      next_state.emplace(
          spec.cached_keys_input_name,
          std::move(named_outputs.at(spec.cached_keys_output_name)));
      next_state.emplace(
          spec.cached_values_input_name,
          std::move(named_outputs.at(spec.cached_values_output_name)));
      next_state.emplace(
          spec.cached_positions_input_name,
          std::move(named_outputs.at(spec.cached_positions_output_name)));
    }
    next_state.emplace("audio", std::move(named_outputs.at("audio")));
    next_state.emplace("audio_lengths",
                      std::move(named_outputs.at("audio_lengths")));

    return next_state;
  }

  const ZeroTtsCodecStreamingMeta &GetCodecStreamingMeta() const {
    return codec_meta_;
  }

  int32_t GetSampleRate() const { return config_.sample_rate; }
  int32_t GetDModel() const { return config_.d_model; }
  int32_t GetNumCodebooks() const { return config_.num_codebooks; }
  int32_t GetCodebookSize() const { return config_.codebook_size; }
  int32_t GetNumLayers() const { return config_.n_layers; }
  int32_t GetNumHeads() const { return config_.n_heads; }
  int32_t GetHeadDim() const { return config_.d_model / config_.n_heads; }

  OrtAllocator *Allocator() const { return allocator_; }

 private:
  struct NameSet {
    std::vector<std::string> names;
    std::vector<const char *> ptrs;
  };

  static void InitNames(Ort::Session *sess, NameSet *in, NameSet *out) {
    std::vector<std::string> in_names, out_names;
    std::vector<const char *> in_ptrs, out_ptrs;
    GetInputNames(sess, &in_names, &in_ptrs);
    GetOutputNames(sess, &out_names, &out_ptrs);
    in->names = std::move(in_names);
    in->ptrs = std::move(in_ptrs);
    out->names = std::move(out_names);
    out->ptrs = std::move(out_ptrs);
  }

  static std::string ReadFileToString(const std::string &path) {
    std::vector<char> buf = ReadFile(path);
    return std::string(buf.data(), buf.size());
  }

  void PrintDebugInfo() const {
    std::ostringstream os;
    os << "---zerotts model---\n";
    os << "d_model=" << config_.d_model << " n_layers=" << config_.n_layers
       << " n_heads=" << config_.n_heads
       << " num_codebooks=" << config_.num_codebooks
       << " codebook_size=" << config_.codebook_size
       << " sample_rate=" << config_.sample_rate << "\n";
    os << "codec streaming: " << codec_meta_.transformer_offsets.size()
       << " transformer offsets, " << codec_meta_.attention_caches.size()
       << " attention caches\n";
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
  }

  OfflineTtsZeroTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> text_encoder_sess_;
  std::unique_ptr<Ort::Session> prefix_step_sess_;
  std::unique_ptr<Ort::Session> local_frame_decode_sess_;
  std::unique_ptr<Ort::Session> codec_decode_full_sess_;
  std::unique_ptr<Ort::Session> codec_decode_step_sess_;

  NameSet text_encoder_in_, text_encoder_out_;
  NameSet prefix_step_in_, prefix_step_out_;
  NameSet local_frame_decode_in_, local_frame_decode_out_;
  NameSet codec_decode_full_in_, codec_decode_full_out_;
  NameSet codec_decode_step_in_, codec_decode_step_out_;

  ZeroTtsCodecStreamingMeta codec_meta_;
};

OfflineTtsZeroTtsModel::~OfflineTtsZeroTtsModel() = default;

OfflineTtsZeroTtsModel::OfflineTtsZeroTtsModel(
    const OfflineTtsZeroTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsZeroTtsModel::OfflineTtsZeroTtsModel(
    Manager *mgr, const OfflineTtsZeroTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<Ort::Value> OfflineTtsZeroTtsModel::RunTextEncoder(
    Ort::Value text_ids, Ort::Value txt_lengths) const {
  return impl_->RunTextEncoder(std::move(text_ids), std::move(txt_lengths));
}

std::vector<Ort::Value> OfflineTtsZeroTtsModel::RunPrefixStep(
    Ort::Value external_embed, Ort::Value use_external_embed,
    Ort::Value frame_codes, Ort::Value new_pos, Ort::Value new_valid,
    Ort::Value new_bidirectional, Ort::Value packed_kv, Ort::Value past_valid,
    Ort::Value cross_kv, Ort::Value text_valid) const {
  return impl_->RunPrefixStep(
      std::move(external_embed), std::move(use_external_embed),
      std::move(frame_codes), std::move(new_pos), std::move(new_valid),
      std::move(new_bidirectional), std::move(packed_kv),
      std::move(past_valid), std::move(cross_kv), std::move(text_valid));
}

std::vector<Ort::Value> OfflineTtsZeroTtsModel::RunLocalFrameDecode(
    Ort::Value global_hidden, Ort::Value forbid_eoa,
    Ort::Value text_temperature, Ort::Value text_topk,
    Ort::Value audio_temperature, Ort::Value audio_topk,
    Ort::Value audio_topp, Ort::Value audio_repetition_penalty,
    Ort::Value seen_mask, Ort::Value ctrl_random_u, Ort::Value audio_random_u,
    Ort::Value cfg_scale) const {
  return impl_->RunLocalFrameDecode(
      std::move(global_hidden), std::move(forbid_eoa),
      std::move(text_temperature), std::move(text_topk),
      std::move(audio_temperature), std::move(audio_topk),
      std::move(audio_topp), std::move(audio_repetition_penalty),
      std::move(seen_mask), std::move(ctrl_random_u),
      std::move(audio_random_u), std::move(cfg_scale));
}

std::vector<Ort::Value> OfflineTtsZeroTtsModel::RunCodecDecodeFull(
    Ort::Value audio_codes, Ort::Value audio_code_lengths) const {
  return impl_->RunCodecDecodeFull(std::move(audio_codes),
                                   std::move(audio_code_lengths));
}

std::unordered_map<std::string, Ort::Value>
OfflineTtsZeroTtsModel::RunCodecDecodeStep(
    Ort::Value audio_codes, Ort::Value audio_code_lengths,
    std::unordered_map<std::string, Ort::Value> state) const {
  return impl_->RunCodecDecodeStep(
      std::move(audio_codes), std::move(audio_code_lengths),
      std::move(state));
}

const ZeroTtsCodecStreamingMeta &OfflineTtsZeroTtsModel::GetCodecStreamingMeta()
    const {
  return impl_->GetCodecStreamingMeta();
}

int32_t OfflineTtsZeroTtsModel::GetSampleRate() const {
  return impl_->GetSampleRate();
}
int32_t OfflineTtsZeroTtsModel::GetDModel() const {
  return impl_->GetDModel();
}
int32_t OfflineTtsZeroTtsModel::GetNumCodebooks() const {
  return impl_->GetNumCodebooks();
}
int32_t OfflineTtsZeroTtsModel::GetCodebookSize() const {
  return impl_->GetCodebookSize();
}
int32_t OfflineTtsZeroTtsModel::GetNumLayers() const {
  return impl_->GetNumLayers();
}
int32_t OfflineTtsZeroTtsModel::GetNumHeads() const {
  return impl_->GetNumHeads();
}
int32_t OfflineTtsZeroTtsModel::GetHeadDim() const {
  return impl_->GetHeadDim();
}
OrtAllocator *OfflineTtsZeroTtsModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineTtsZeroTtsModel::OfflineTtsZeroTtsModel(
    AAssetManager *mgr, const OfflineTtsZeroTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsZeroTtsModel::OfflineTtsZeroTtsModel(
    NativeResourceManager *mgr, const OfflineTtsZeroTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
