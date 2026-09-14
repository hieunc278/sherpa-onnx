// sherpa-onnx/csrc/offline-tts-zerotts-npy.h
//
// Copyright (c)  2026
//
// A minimal NumPy .npy (not .npz) reader, scoped to exactly the two dtypes
// ZeroTTS ships (see spec.md §12.4 / plan.md Phase 2): float32 ("<f4", for
// null_voice_emb.npy) and int64 ("<i8", for silence_frame.npy). Not a
// general-purpose NumPy format reader -- deliberately narrow, per the
// decision recorded in spec.md to avoid depending on a full NumPy-format
// library for two small, fixed-shape files.
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_NPY_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_NPY_H_

#include <cstdint>
#include <string>
#include <vector>

namespace sherpa_onnx {

struct ZeroTtsNpyArray {
  std::vector<int64_t> shape;
  std::string dtype;         // "<f4" or "<i8"
  std::vector<float> f32;    // populated iff dtype == "<f4"
  std::vector<int64_t> i64;  // populated iff dtype == "<i8"

  int64_t NumElements() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return shape.empty() ? 0 : n;
  }
};

// Returns true and fills `out` on success; returns false (and logs) on any
// parse failure -- malformed magic/version, unsupported dtype, Fortran
// order, or truncated file.
bool ReadZeroTtsNpy(const std::string &path, ZeroTtsNpyArray *out);

template <typename Manager>
bool ReadZeroTtsNpy(Manager *mgr, const std::string &path,
                    ZeroTtsNpyArray *out);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_NPY_H_
