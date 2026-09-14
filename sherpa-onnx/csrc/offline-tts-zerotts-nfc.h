// sherpa-onnx/csrc/offline-tts-zerotts-nfc.h
//
// Copyright (c)  2026
//
// Shared Unicode NFC (canonical) normalization, table-driven -- extracted
// out of the tokenizer so both OfflineTtsZeroTtsTokenizer (tokenizer.json's
// normalizer step) and OfflineTtsZeroTtsViNormalizer (vi_normalizer.py's
// own unicodedata.normalize("NFC", ...) call) share one implementation
// instead of two copies drifting apart. See
// offline-tts-zerotts-nfc-table.h / scripts/zerotts/generate_nfc_table.py.
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_NFC_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_NFC_H_

#include <string>

namespace sherpa_onnx {

// NFC-normalizes `text` (canonical decomposition + canonical ordering +
// canonical composition -- no compatibility decomposition, i.e. NOT NFKC).
std::string ZeroTtsToNfc(const std::string &text);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_NFC_H_
