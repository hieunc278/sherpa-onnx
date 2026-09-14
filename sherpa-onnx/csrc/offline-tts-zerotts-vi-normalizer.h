// sherpa-onnx/csrc/offline-tts-zerotts-vi-normalizer.h
//
// Copyright (c)  2026
//
// Port of ZeroTTS's vi_normalizer.py, scoped per
// notes/phase0-vi-normalizer-scope.md (Phase 0.4): 8 of 17 categories found
// in the Python source are ported for v1 (numbers, percent, time, date,
// month-year, abbreviations, fractions, the "@" sign, plus URL/email
// protection as a prerequisite guard), 9 lower-value/niche categories are
// deferred, each with a documented reason in that note.
//
// Key difference from the Python original: `vi_normalizer.py`'s scanner is
// one big regex relying on negative lookbehind assertions, which C++
// `std::regex` does not support at all. This port is a hand-rolled
// left-to-right scanner instead of a regex translation -- see the note for
// the full rationale.
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_VI_NORMALIZER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_VI_NORMALIZER_H_

#include <string>

namespace sherpa_onnx {

// Vietnamese text normalization for TTS input: numbers, dates, times,
// percentages, fractions, abbreviations/acronyms -> spoken Vietnamese words.
// Safe to call on text with none of these forms (a no-op beyond NFC
// normalization + the space-cleanup pass at the very end), and on
// already-normalized text (idempotent for the categories it covers).
std::string NormalizeViText(const std::string &text);

// Speaks an integer / decimal / signed number ("1.250.000" -> "một triệu
// hai trăm năm mươi nghìn", "12,5" -> "mười hai phẩy năm"). Exposed for
// testing; also used internally by NormalizeViText's date/time/percent/
// fraction handlers.
//
// v1 simplification vs. vi_normalizer.py's expand_number(): does not parse
// multi-number arithmetic expressions ("2+3", "5 * 4") -- see
// notes/phase0-vi-normalizer-scope.md category #1's port note. A bare
// integer/decimal/signed number is fully supported.
std::string ZeroTtsExpandNumber(const std::string &number);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_VI_NORMALIZER_H_
