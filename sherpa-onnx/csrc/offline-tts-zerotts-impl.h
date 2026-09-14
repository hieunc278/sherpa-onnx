// sherpa-onnx/csrc/offline-tts-zerotts-impl.h
//
// Copyright (c)  2026
//
// OfflineTtsZeroTtsImpl -- orchestrates the ZeroTTS pipeline (spec.md §2/§7/
// §8): text -> tokenizer -> text_encoder (once) -> prefix_step (cold
// start) -> loop(local_frame_decode -> prefix_step) -> stop condition ->
// codec_decode_full (offline) or codec_decode_step per chunk (streaming) ->
// waveform. This file owns the actual autoregressive loop's control flow
// (RunArLoop, shared by both Generate and GenerateStreaming so the
// highest-risk piece of this feature -- the AR loop's stop condition -- is
// implemented exactly once); OfflineTtsZeroTtsModel
// (offline-tts-zerotts-model.h) is purely the ONNX I/O binding layer
// underneath it.
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_IMPL_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-model.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-npy.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-tokenizer.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-vi-normalizer.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-voice.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

class OfflineTtsZeroTtsImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsZeroTtsImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsZeroTtsModel>(config.model.zerotts)),
        tokenizer_(config.model.zerotts.tokenizer),
        rng_(std::random_device{}()) {
    PostInit();
  }

  template <typename Manager>
  OfflineTtsZeroTtsImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(
            std::make_unique<OfflineTtsZeroTtsModel>(mgr, config.model.zerotts)),
        tokenizer_(mgr, config.model.zerotts.tokenizer),
        rng_(std::random_device{}()) {
    PostInit(mgr);
  }

  int32_t SampleRate() const override { return model_->GetSampleRate(); }

  bool SupportsStreaming() const override { return true; }

  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const override {
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    int32_t K = model_->GetNumCodebooks();

    std::vector<int32_t> codes_flat;  // (T_gen, K), row-major -- matches
                                      // codec_decode_full's (B,T,K) layout
    RunArLoop(text, mem, [&](const int32_t *codes, int32_t k) {
      codes_flat.insert(codes_flat.end(), codes, codes + k);
    });

    int32_t n_frames = static_cast<int32_t>(codes_flat.size()) / K;
    if (n_frames == 0) {
      return GeneratedAudio{{}, model_->GetSampleRate()};
    }

    std::vector<int64_t> audio_codes_shape = {1, n_frames, K};
    std::vector<int32_t> audio_code_lengths = {n_frames};
    std::vector<int64_t> lengths_shape = {1};

    std::vector<Ort::Value> codec_out = model_->RunCodecDecodeFull(
        Ort::Value::CreateTensor<int32_t>(mem, codes_flat.data(),
                                          codes_flat.size(),
                                          audio_codes_shape.data(), 3),
        Ort::Value::CreateTensor<int32_t>(mem, audio_code_lengths.data(), 1,
                                          lengths_shape.data(), 1));

    GeneratedAudio result;
    result.sample_rate = model_->GetSampleRate();
    result.samples = ExtractMonoAudio(codec_out[0], codec_out[1]);

    if (callback) {
      callback(result.samples.data(),
               static_cast<int32_t>(result.samples.size()), 1.0f);
    }

    return result;
  }

  // Streaming synthesis (spec.md §8): reuses the identical AR loop as
  // Generate() above (RunArLoop), but instead of accumulating all frames
  // and calling codec_decode_full once, buffers frames per the doubling
  // schedule (first_chunk_frames, then x2 up to max_chunk_frames) and
  // decodes each buffered chunk via codec_decode_step, carrying the
  // codec's ring-buffer/KV-cache state forward between calls -- mirrors
  // codec.py's streaming_decoder()/decode_chunk().
  void GenerateStreaming(const std::string &text,
                        const GenerationConfig & /*config*/,
                        StreamingAudioCallback callback) const override {
    const auto &zc = config_.model.zerotts;
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    int32_t K = model_->GetNumCodebooks();
    int32_t sample_rate = model_->GetSampleRate();

    std::unordered_map<std::string, Ort::Value> state =
        InitCodecStreamingState(mem);

    std::vector<int32_t> buf_codes;  // (n_buffered, K), row-major
    int32_t buf_frames = 0;
    int32_t target = std::max(1, zc.first_chunk_frames);
    int32_t cap = std::max(target, zc.max_chunk_frames);
    bool stopped_early = false;

    auto flush = [&]() {
      if (buf_frames == 0 || stopped_early) return;
      std::vector<int64_t> shape = {1, buf_frames, K};
      std::vector<int32_t> lengths = {buf_frames};
      std::vector<int64_t> len_shape = {1};

      Ort::Value audio_codes = Ort::Value::CreateTensor<int32_t>(
          mem, buf_codes.data(), buf_codes.size(), shape.data(), 3);
      Ort::Value audio_lengths = Ort::Value::CreateTensor<int32_t>(
          mem, lengths.data(), 1, len_shape.data(), 1);

      std::unordered_map<std::string, Ort::Value> result =
          model_->RunCodecDecodeStep(std::move(audio_codes),
                                     std::move(audio_lengths),
                                     std::move(state));
      Ort::Value audio_chunk = std::move(result.at("audio"));
      Ort::Value audio_len = std::move(result.at("audio_lengths"));
      result.erase("audio");
      result.erase("audio_lengths");
      state = std::move(result);

      std::vector<float> mono = ExtractMonoAudio(audio_chunk, audio_len);
      if (callback && !mono.empty()) {
        int32_t keep_going =
            callback(mono.data(), static_cast<int32_t>(mono.size()),
                    sample_rate);
        if (keep_going == 0) stopped_early = true;
      }

      buf_codes.clear();
      buf_frames = 0;
      target = std::min(cap, target * 2);
    };

    RunArLoop(text, mem, [&](const int32_t *codes, int32_t k) {
      if (stopped_early) return;
      buf_codes.insert(buf_codes.end(), codes, codes + k);
      ++buf_frames;
      if (buf_frames >= target) flush();
    });
    flush();
  }

 private:
  // Runs text_encoder once, then prefix_step (cold start) + the
  // local_frame_decode/prefix_step AR loop (spec.md §7's stop condition,
  // ported 1:1 from synthesizer.py's _generate_frames), invoking
  // `on_frame(codes_ptr, K)` once per *kept* frame (i.e. after the
  // eoa/tail-frame/max_frames stop-condition checks -- matches
  // synthesizer.py's `yield codes` placement exactly). Does not touch the
  // codec at all; that's the caller's job (offline: decode_full once at
  // the end; streaming: decode_step per buffered chunk).
  void RunArLoop(
      const std::string &text, Ort::MemoryInfo &mem,
      const std::function<void(const int32_t *, int32_t)> &on_frame) const {
    const auto &zc = config_.model.zerotts;
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("zerotts input text: %s", text.c_str());
    }

    int32_t d_model = model_->GetDModel();
    int32_t n_layers = model_->GetNumLayers();
    int32_t n_heads = model_->GetNumHeads();
    int32_t d_head = model_->GetHeadDim();
    int32_t K = model_->GetNumCodebooks();
    int32_t codebook_size = model_->GetCodebookSize();

    // ---- 1. normalize (numbers/dates/times/abbreviations -> spoken
    //         Vietnamese, see spec.md §5/Phase 5) then tokenize ------------
    std::string normalized_text = NormalizeViText(text);
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("zerotts normalized text: %s", normalized_text.c_str());
    }
    std::vector<int32_t> ids32 = tokenizer_.Encode(normalized_text);
    std::vector<int64_t> text_ids(ids32.begin(), ids32.end());
    int32_t L = static_cast<int32_t>(text_ids.size());
    std::vector<int64_t> txt_lengths = {L};

    std::vector<int64_t> text_ids_shape = {1, L};
    std::vector<int64_t> len_shape = {1};
    Ort::Value text_ids_t = Ort::Value::CreateTensor<int64_t>(
        mem, text_ids.data(), text_ids.size(), text_ids_shape.data(), 2);
    Ort::Value txt_lengths_t = Ort::Value::CreateTensor<int64_t>(
        mem, txt_lengths.data(), txt_lengths.size(), len_shape.data(), 1);

    // ---- 2. text_encoder (once) ----------------------------------------
    std::vector<Ort::Value> enc_out =
        model_->RunTextEncoder(std::move(text_ids_t), std::move(txt_lengths_t));
    // order confirmed via onnxruntime introspection (Phase 0/3):
    // [text_states, text_valid, soa_embed, cross_kv]
    Ort::Value text_valid = std::move(enc_out[1]);
    Ort::Value soa_embed = std::move(enc_out[2]);
    Ort::Value cross_kv = std::move(enc_out[3]);

    auto soa_shape = soa_embed.GetTensorTypeAndShapeInfo().GetShape();
    int32_t Gsoa = static_cast<int32_t>(soa_shape[1]);

    // ---- 3. resolve voice, build external_embed = concat(voice, soa) ---
    const std::vector<float> *voice_emb_ptr = nullptr;
    int32_t V = 0;
    if (!zc.voice.empty()) {
      V = loaded_voice_.n_voice_queries;
      voice_emb_ptr = &loaded_voice_.emb;
    } else {
      V = static_cast<int32_t>(null_voice_emb_.shape[1]);
      voice_emb_ptr = &null_voice_emb_.f32;
    }

    int32_t T0 = V + Gsoa;
    std::vector<float> external_embed(static_cast<size_t>(T0) * d_model);
    std::copy(voice_emb_ptr->begin(), voice_emb_ptr->end(),
             external_embed.begin());
    const float *soa_data = soa_embed.GetTensorData<float>();
    std::copy(soa_data, soa_data + static_cast<size_t>(Gsoa) * d_model,
             external_embed.begin() + static_cast<size_t>(V) * d_model);

    std::vector<uint8_t> use_external_embed(T0, 1);
    std::vector<int64_t> frame_codes_init(static_cast<size_t>(T0) * K, 0);
    std::vector<int64_t> new_pos_init(T0);
    for (int32_t i = 0; i < T0; ++i) new_pos_init[i] = i;
    std::vector<uint8_t> new_valid_init(T0, 1);
    std::vector<uint8_t> new_bidir_init(T0, 0);
    for (int32_t i = 0; i < V; ++i) new_bidir_init[i] = 1;
    std::vector<float> packed_kv_dummy(1, 0.f);
    std::vector<uint8_t> past_valid_dummy(1, 0);

    std::vector<int64_t> embed_shape = {1, T0, d_model};
    std::vector<int64_t> flag_shape_t0 = {1, T0};
    std::vector<int64_t> frame_codes_shape = {1, T0, K};
    std::vector<int64_t> packed_kv_shape = {n_layers, 2, 1, n_heads, 0, d_head};
    std::vector<int64_t> past_valid_shape = {1, 0};

    std::vector<Ort::Value> step0 = model_->RunPrefixStep(
        Ort::Value::CreateTensor<float>(mem, external_embed.data(),
                                        external_embed.size(),
                                        embed_shape.data(), 3),
        Ort::Value::CreateTensor<bool>(
            mem, reinterpret_cast<bool *>(use_external_embed.data()),
            use_external_embed.size(), flag_shape_t0.data(), 2),
        Ort::Value::CreateTensor<int64_t>(mem, frame_codes_init.data(),
                                          frame_codes_init.size(),
                                          frame_codes_shape.data(), 3),
        Ort::Value::CreateTensor<int64_t>(mem, new_pos_init.data(),
                                          new_pos_init.size(),
                                          flag_shape_t0.data(), 2),
        Ort::Value::CreateTensor<bool>(
            mem, reinterpret_cast<bool *>(new_valid_init.data()),
            new_valid_init.size(), flag_shape_t0.data(), 2),
        Ort::Value::CreateTensor<bool>(
            mem, reinterpret_cast<bool *>(new_bidir_init.data()),
            new_bidir_init.size(), flag_shape_t0.data(), 2),
        Ort::Value::CreateTensor<float>(mem, packed_kv_dummy.data(), 0,
                                        packed_kv_shape.data(), 6),
        Ort::Value::CreateTensor<bool>(
            mem, reinterpret_cast<bool *>(past_valid_dummy.data()), 0,
            past_valid_shape.data(), 2),
        View(&cross_kv), View(&text_valid));

    Ort::Value hidden = std::move(step0[0]);
    Ort::Value packed_kv = std::move(step0[1]);
    Ort::Value full_valid = std::move(step0[2]);

    std::vector<float> h_last(d_model);
    {
      const float *h = hidden.GetTensorData<float>();
      auto hshape = hidden.GetTensorTypeAndShapeInfo().GetShape();
      int64_t Tcur = hshape[1];
      std::copy(h + (Tcur - 1) * d_model, h + Tcur * d_model, h_last.begin());
    }

    // ---- 4. AR loop: local_frame_decode -> prefix_step -----------------
    std::vector<uint8_t> seen_mask(static_cast<size_t>(K) * codebook_size, 0);
    int32_t t = 0;
    bool tail_active = false;
    int32_t tail_left = 0;

    std::vector<int64_t> scalar_shape = {1};
    std::vector<int64_t> seen_mask_shape = {1, K, codebook_size};
    std::vector<int64_t> audio_u_shape = {1, K};
    std::vector<int64_t> h_last_shape = {1, d_model};

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    while (true) {
      bool forbid_eoa = (t < zc.min_frames) || tail_active;

      bool forbid_eoa_val = forbid_eoa;
      float text_temp_val = zc.text_temperature;
      int64_t text_topk_val = zc.text_topk;
      float audio_temp_val = zc.audio_temperature;
      int64_t audio_topk_val = zc.audio_topk;
      float audio_topp_val = zc.audio_topp;
      float audio_rep_pen_val = zc.audio_repetition_penalty;
      float cfg_scale_val = zc.cfg_scale;

      std::vector<float> ctrl_u = {uniform(rng_)};
      std::vector<float> audio_u(K);
      for (auto &v : audio_u) v = uniform(rng_);

      std::vector<Ort::Value> frame_out = model_->RunLocalFrameDecode(
          Ort::Value::CreateTensor<float>(mem, h_last.data(), h_last.size(),
                                          h_last_shape.data(), 2),
          Ort::Value::CreateTensor<bool>(
              mem, reinterpret_cast<bool *>(&forbid_eoa_val), 1,
              scalar_shape.data(), 1),
          Ort::Value::CreateTensor<float>(mem, &text_temp_val, 1,
                                          scalar_shape.data(), 1),
          Ort::Value::CreateTensor<int64_t>(mem, &text_topk_val, 1,
                                            scalar_shape.data(), 1),
          Ort::Value::CreateTensor<float>(mem, &audio_temp_val, 1,
                                          scalar_shape.data(), 1),
          Ort::Value::CreateTensor<int64_t>(mem, &audio_topk_val, 1,
                                            scalar_shape.data(), 1),
          Ort::Value::CreateTensor<float>(mem, &audio_topp_val, 1,
                                          scalar_shape.data(), 1),
          Ort::Value::CreateTensor<float>(mem, &audio_rep_pen_val, 1,
                                          scalar_shape.data(), 1),
          Ort::Value::CreateTensor<bool>(
              mem, reinterpret_cast<bool *>(seen_mask.data()),
              seen_mask.size(), seen_mask_shape.data(), 3),
          Ort::Value::CreateTensor<float>(mem, ctrl_u.data(), 1,
                                          scalar_shape.data(), 1),
          Ort::Value::CreateTensor<float>(mem, audio_u.data(), audio_u.size(),
                                          audio_u_shape.data(), 2),
          Ort::Value::CreateTensor<float>(mem, &cfg_scale_val, 1,
                                          scalar_shape.data(), 1));

      int32_t is_eoa = frame_out[0].GetTensorData<int32_t>()[0];
      const int32_t *codes_ptr = frame_out[1].GetTensorData<int32_t>();
      std::vector<int32_t> codes(codes_ptr, codes_ptr + K);
      for (int32_t c = 0; c < K; ++c) {
        seen_mask[static_cast<size_t>(c) * codebook_size + codes[c]] = 1;
      }

      if (!tail_active && is_eoa != 0) {
        tail_active = true;
        tail_left = std::max(0, zc.eoa_extra_frames);
      }

      if ((tail_active && tail_left <= 0) || t >= zc.max_frames) {
        break;
      }

      on_frame(codes.data(), K);

      if (tail_active) {
        tail_left -= 1;
        if (tail_left <= 0) {
          break;
        }
      }

      // advance the global transformer by this frame
      std::vector<int64_t> frame_codes_step(codes.begin(), codes.end());
      std::vector<int64_t> new_pos_step = {V + 1 + t};
      std::vector<float> zero_embed(d_model, 0.f);
      uint8_t zero_flag = 0;
      uint8_t one_flag = 1;
      std::vector<int64_t> one_embed_shape = {1, 1, d_model};
      std::vector<int64_t> one_flag_shape = {1, 1};
      std::vector<int64_t> frame_codes_step_shape = {1, 1, K};

      std::vector<Ort::Value> step = model_->RunPrefixStep(
          Ort::Value::CreateTensor<float>(mem, zero_embed.data(),
                                          zero_embed.size(),
                                          one_embed_shape.data(), 3),
          Ort::Value::CreateTensor<bool>(
              mem, reinterpret_cast<bool *>(&zero_flag), 1,
              one_flag_shape.data(), 2),
          Ort::Value::CreateTensor<int64_t>(mem, frame_codes_step.data(),
                                            frame_codes_step.size(),
                                            frame_codes_step_shape.data(), 3),
          Ort::Value::CreateTensor<int64_t>(mem, new_pos_step.data(), 1,
                                            one_flag_shape.data(), 2),
          Ort::Value::CreateTensor<bool>(
              mem, reinterpret_cast<bool *>(&one_flag), 1,
              one_flag_shape.data(), 2),
          Ort::Value::CreateTensor<bool>(
              mem, reinterpret_cast<bool *>(&zero_flag), 1,
              one_flag_shape.data(), 2),
          std::move(packed_kv), std::move(full_valid), View(&cross_kv),
          View(&text_valid));

      hidden = std::move(step[0]);
      packed_kv = std::move(step[1]);
      full_valid = std::move(step[2]);

      const float *h = hidden.GetTensorData<float>();
      std::copy(h, h + d_model, h_last.begin());  // T=1 here, so h[:, -1, :]
                                                  // is just the only row

      t += 1;
    }
  }

  // Averages a (1, C, audio_length) float tensor down to mono, trimmed to
  // `audio_lengths`'s reported valid length. Shared by Generate() and
  // GenerateStreaming()'s chunk flushing.
  static std::vector<float> ExtractMonoAudio(const Ort::Value &audio,
                                             const Ort::Value &audio_lengths) {
    int32_t n = audio_lengths.GetTensorData<int32_t>()[0];
    auto shape = audio.GetTensorTypeAndShapeInfo().GetShape();
    int64_t C = shape[1];
    int64_t audio_length = shape[2];
    const float *data = audio.GetTensorData<float>();

    std::vector<float> mono(n);
    for (int32_t i = 0; i < n; ++i) {
      float sum = 0.f;
      for (int64_t c = 0; c < C; ++c) sum += data[c * audio_length + i];
      mono[i] = sum / static_cast<float>(C);
    }
    return mono;
  }

  // Zero-initializes the decode_step state map from
  // codec_browser_onnx_meta.json's declared shapes (see
  // notes/phase0-codec-decode-step-io.md) -- mirrors codec.py's
  // MossStreamingDecoder._reset_state(), including the -1 (not 0) fill for
  // attn_cached_positions_* ("position 0 is a real position").
  std::unordered_map<std::string, Ort::Value> InitCodecStreamingState(
      Ort::MemoryInfo &mem) const {
    const auto &meta = model_->GetCodecStreamingMeta();
    std::unordered_map<std::string, Ort::Value> state;

    streaming_state_bufs_i32_.clear();
    streaming_state_bufs_f32_.clear();

    for (const auto &spec : meta.transformer_offsets) {
      auto buf = std::make_shared<std::vector<int32_t>>(
          ShapeNumElements(spec.shape), 0);
      state.emplace(spec.input_name,
                    Ort::Value::CreateTensor<int32_t>(
                        mem, buf->data(), buf->size(), spec.shape.data(),
                        spec.shape.size()));
      streaming_state_bufs_i32_.push_back(buf);
    }

    for (const auto &spec : meta.attention_caches) {
      auto off = std::make_shared<std::vector<int32_t>>(
          ShapeNumElements(spec.offset_shape), 0);
      state.emplace(spec.offset_input_name,
                    Ort::Value::CreateTensor<int32_t>(
                        mem, off->data(), off->size(), spec.offset_shape.data(),
                        spec.offset_shape.size()));
      streaming_state_bufs_i32_.push_back(off);

      auto keys = std::make_shared<std::vector<float>>(
          ShapeNumElements(spec.cache_shape), 0.f);
      state.emplace(spec.cached_keys_input_name,
                    Ort::Value::CreateTensor<float>(
                        mem, keys->data(), keys->size(),
                        spec.cache_shape.data(), spec.cache_shape.size()));
      streaming_state_bufs_f32_.push_back(keys);

      auto values = std::make_shared<std::vector<float>>(
          ShapeNumElements(spec.cache_shape), 0.f);
      state.emplace(spec.cached_values_input_name,
                    Ort::Value::CreateTensor<float>(
                        mem, values->data(), values->size(),
                        spec.cache_shape.data(), spec.cache_shape.size()));
      streaming_state_bufs_f32_.push_back(values);

      auto positions = std::make_shared<std::vector<int32_t>>(
          ShapeNumElements(spec.positions_shape), -1);
      state.emplace(spec.cached_positions_input_name,
                    Ort::Value::CreateTensor<int32_t>(
                        mem, positions->data(), positions->size(),
                        spec.positions_shape.data(),
                        spec.positions_shape.size()));
      streaming_state_bufs_i32_.push_back(positions);
    }

    return state;
  }

  static int64_t ShapeNumElements(const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }

  void PostInit() {
    const auto &zc = config_.model.zerotts;
    if (!ReadZeroTtsNpy(zc.null_voice_emb, &null_voice_emb_)) {
      SHERPA_ONNX_LOGE("Failed to load %s", zc.null_voice_emb.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    if (!zc.voice.empty()) {
      if (!LoadOfflineTtsZeroTtsVoice(zc.voice, zc.d_model, &loaded_voice_)) {
        SHERPA_ONNX_LOGE("Failed to load voice from %s", zc.voice.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  template <typename Manager>
  void PostInit(Manager * /*mgr*/) {
    // Voice/npy loading currently goes through plain file paths even on
    // Android (see offline-tts-zerotts-model.cc's Manager-constructor
    // comment about external-data codec graphs -- same Phase 8 follow-up
    // applies to these small asset files).
    PostInit();
  }

  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsZeroTtsModel> model_;
  OfflineTtsZeroTtsTokenizer tokenizer_;
  ZeroTtsNpyArray null_voice_emb_;
  OfflineTtsZeroTtsVoice loaded_voice_;
  mutable std::mt19937 rng_;

  // InitCodecStreamingState()'s zero-buffers must outlive the Ort::Value
  // tensors that view them (CreateTensor doesn't copy); GenerateStreaming()
  // is const, so these are mutable scratch state, cleared at the start of
  // each GenerateStreaming() call.
  mutable std::vector<std::shared_ptr<std::vector<int32_t>>>
      streaming_state_bufs_i32_;
  mutable std::vector<std::shared_ptr<std::vector<float>>>
      streaming_state_bufs_f32_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_IMPL_H_
