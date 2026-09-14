// sherpa-onnx/csrc/offline-tts-zerotts-model-config.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-tts-zerotts-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsZeroTtsModelConfig::Register(ParseOptions *po) {
  po->Register("zerotts-text-encoder", &text_encoder,
               "Path to ZeroTTS text_encoder.onnx");
  po->Register("zerotts-prefix-step", &prefix_step,
               "Path to ZeroTTS prefix_step.onnx");
  po->Register("zerotts-local-frame-decode", &local_frame_decode,
               "Path to ZeroTTS local_frame_decode.onnx");
  po->Register("zerotts-codec-decode-full", &codec_decode_full,
               "Path to ZeroTTS moss_audio_tokenizer_decode_full.onnx");
  po->Register("zerotts-codec-decode-step", &codec_decode_step,
               "Path to ZeroTTS moss_audio_tokenizer_decode_step.onnx");
  po->Register("zerotts-codec-meta", &codec_meta,
               "Path to ZeroTTS codec_browser_onnx_meta.json");
  po->Register("zerotts-tokenizer", &tokenizer,
               "Path to ZeroTTS tokenizer.json");
  po->Register("zerotts-null-voice-emb", &null_voice_emb,
               "Path to ZeroTTS null_voice_emb.npy");
  po->Register("zerotts-silence-frame", &silence_frame,
               "Path to ZeroTTS silence_frame.npy");
  po->Register("zerotts-voice", &voice,
               "Path to a ZeroTTS voices/<name>/ directory");

  po->Register("zerotts-min-frames", &min_frames,
               "Minimum number of generated audio frames before <eoa> is "
               "allowed to fire");
  po->Register("zerotts-max-frames", &max_frames,
               "Maximum number of generated audio frames (hard stop)");
  po->Register("zerotts-eoa-extra-frames", &eoa_extra_frames,
               "Number of trailing frames to keep after <eoa> fires");
  po->Register("zerotts-text-temperature", &text_temperature,
               "Sampling temperature for the control channel");
  po->Register("zerotts-text-topk", &text_topk,
               "Top-k for the control channel");
  po->Register("zerotts-audio-temperature", &audio_temperature,
               "Sampling temperature for the audio codebooks");
  po->Register("zerotts-audio-topk", &audio_topk,
               "Top-k for the audio codebooks");
  po->Register("zerotts-audio-topp", &audio_topp,
               "Top-p (nucleus) for the audio codebooks");
  po->Register("zerotts-audio-repetition-penalty", &audio_repetition_penalty,
               "Repetition penalty for the audio codebooks");
  po->Register("zerotts-cfg-scale", &cfg_scale,
               "Classifier-free-guidance scale (>1.0 doubles per-frame cost)");
  po->Register("zerotts-first-chunk-frames", &first_chunk_frames,
               "Number of frames in the first streaming chunk");
  po->Register("zerotts-max-chunk-frames", &max_chunk_frames,
               "Maximum number of frames per streaming chunk");
}

bool OfflineTtsZeroTtsModelConfig::Validate() const {
  if (text_encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zerotts-text-encoder");
    return false;
  }
  if (!FileExists(text_encoder)) {
    SHERPA_ONNX_LOGE("--zerotts-text-encoder: '%s' does not exist",
                     text_encoder.c_str());
    return false;
  }

  if (prefix_step.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zerotts-prefix-step");
    return false;
  }
  if (!FileExists(prefix_step)) {
    SHERPA_ONNX_LOGE("--zerotts-prefix-step: '%s' does not exist",
                     prefix_step.c_str());
    return false;
  }

  if (local_frame_decode.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zerotts-local-frame-decode");
    return false;
  }
  if (!FileExists(local_frame_decode)) {
    SHERPA_ONNX_LOGE("--zerotts-local-frame-decode: '%s' does not exist",
                     local_frame_decode.c_str());
    return false;
  }

  if (codec_decode_full.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zerotts-codec-decode-full");
    return false;
  }
  if (!FileExists(codec_decode_full)) {
    SHERPA_ONNX_LOGE("--zerotts-codec-decode-full: '%s' does not exist",
                     codec_decode_full.c_str());
    return false;
  }

  if (codec_decode_step.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zerotts-codec-decode-step");
    return false;
  }
  if (!FileExists(codec_decode_step)) {
    SHERPA_ONNX_LOGE("--zerotts-codec-decode-step: '%s' does not exist",
                     codec_decode_step.c_str());
    return false;
  }

  if (codec_meta.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zerotts-codec-meta");
    return false;
  }
  if (!FileExists(codec_meta)) {
    SHERPA_ONNX_LOGE("--zerotts-codec-meta: '%s' does not exist",
                     codec_meta.c_str());
    return false;
  }

  if (tokenizer.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zerotts-tokenizer");
    return false;
  }
  if (!FileExists(tokenizer)) {
    SHERPA_ONNX_LOGE("--zerotts-tokenizer: '%s' does not exist",
                     tokenizer.c_str());
    return false;
  }

  if (null_voice_emb.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zerotts-null-voice-emb");
    return false;
  }
  if (!FileExists(null_voice_emb)) {
    SHERPA_ONNX_LOGE("--zerotts-null-voice-emb: '%s' does not exist",
                     null_voice_emb.c_str());
    return false;
  }

  if (!silence_frame.empty() && !FileExists(silence_frame)) {
    SHERPA_ONNX_LOGE("--zerotts-silence-frame: '%s' does not exist",
                     silence_frame.c_str());
    return false;
  }

  // `voice` may be empty (falls back to the unconditional null_voice_emb,
  // matching synthesizer.py's resolve_voice(None)), but if given, it must
  // point at a real voice pack directory.
  if (!voice.empty()) {
    std::string voice_bin = voice;
    if (!voice_bin.empty() && voice_bin.back() != '/') voice_bin += "/";
    voice_bin += "voice.bin";
    if (!FileExists(voice_bin)) {
      SHERPA_ONNX_LOGE(
          "--zerotts-voice: '%s' does not exist (expected a "
          "voices/<name>/ directory containing voice.bin)",
          voice_bin.c_str());
      return false;
    }
  }

  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("--zerotts num_threads should be > 0. Given %d",
                     num_threads);
    return false;
  }

  if (min_frames < 0 || max_frames <= 0 || min_frames > max_frames) {
    SHERPA_ONNX_LOGE(
        "Invalid zerotts min_frames/max_frames: %d/%d", min_frames,
        max_frames);
    return false;
  }

  return true;
}

std::string OfflineTtsZeroTtsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsZeroTtsModelConfig(";
  os << "text_encoder=\"" << text_encoder << "\", ";
  os << "prefix_step=\"" << prefix_step << "\", ";
  os << "local_frame_decode=\"" << local_frame_decode << "\", ";
  os << "codec_decode_full=\"" << codec_decode_full << "\", ";
  os << "codec_decode_step=\"" << codec_decode_step << "\", ";
  os << "codec_meta=\"" << codec_meta << "\", ";
  os << "tokenizer=\"" << tokenizer << "\", ";
  os << "null_voice_emb=\"" << null_voice_emb << "\", ";
  os << "silence_frame=\"" << silence_frame << "\", ";
  os << "voice=\"" << voice << "\", ";
  os << "d_model=" << d_model << ", ";
  os << "n_layers=" << n_layers << ", ";
  os << "n_heads=" << n_heads << ", ";
  os << "num_codebooks=" << num_codebooks << ", ";
  os << "codebook_size=" << codebook_size << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "min_frames=" << min_frames << ", ";
  os << "max_frames=" << max_frames << ", ";
  os << "eoa_extra_frames=" << eoa_extra_frames << ", ";
  os << "text_temperature=" << text_temperature << ", ";
  os << "text_topk=" << text_topk << ", ";
  os << "audio_temperature=" << audio_temperature << ", ";
  os << "audio_topk=" << audio_topk << ", ";
  os << "audio_topp=" << audio_topp << ", ";
  os << "audio_repetition_penalty=" << audio_repetition_penalty << ", ";
  os << "cfg_scale=" << cfg_scale << ", ";
  os << "first_chunk_frames=" << first_chunk_frames << ", ";
  os << "max_chunk_frames=" << max_chunk_frames << ")";

  return os.str();
}

}  // namespace sherpa_onnx
