// sherpa-onnx/csrc/offline-tts-zerotts-npy.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-tts-zerotts-npy.h"

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

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

// Extracts the value of "'key': <value>" up to the next comma at depth 0
// (so it can contain nested () / [] without stopping early), starting the
// search at `from`. Returns "" if not found.
std::string ExtractField(const std::string &header, const std::string &key,
                         size_t from = 0) {
  std::string needle = "'" + key + "'";
  size_t pos = header.find(needle, from);
  if (pos == std::string::npos) return "";
  pos = header.find(':', pos);
  if (pos == std::string::npos) return "";
  ++pos;
  while (pos < header.size() && header[pos] == ' ') ++pos;

  size_t start = pos;
  int32_t depth = 0;
  size_t end = pos;
  for (; end < header.size(); ++end) {
    char c = header[end];
    if (c == '(' || c == '[') {
      ++depth;
    } else if (c == ')' || c == ']') {
      if (depth == 0) break;
      --depth;
    } else if (c == ',' && depth == 0) {
      break;
    }
  }
  std::string value = header.substr(start, end - start);
  // trim trailing spaces
  while (!value.empty() && value.back() == ' ') value.pop_back();
  return value;
}

std::string StripQuotes(const std::string &s) {
  if (s.size() >= 2 && (s.front() == '\'' || s.front() == '"')) {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

std::vector<int64_t> ParseShapeTuple(const std::string &s) {
  std::vector<int64_t> shape;
  std::string digits;
  for (char c : s) {
    if (c >= '0' && c <= '9') {
      digits.push_back(c);
    } else {
      if (!digits.empty()) {
        shape.push_back(std::stoll(digits));
        digits.clear();
      }
    }
  }
  if (!digits.empty()) shape.push_back(std::stoll(digits));
  return shape;
}

bool ParseNpy(const std::vector<char> &content, const std::string &path,
             ZeroTtsNpyArray *out) {
  if (content.size() < 10 || std::memcmp(content.data(), "\x93NUMPY", 6) != 0) {
    SHERPA_ONNX_LOGE("%s: not a valid .npy file (bad magic)", path.c_str());
    return false;
  }
  uint8_t major = static_cast<uint8_t>(content[6]);
  size_t header_len_field_size = (major >= 2) ? 4 : 2;
  size_t header_len_offset = 8;
  if (content.size() < header_len_offset + header_len_field_size) {
    SHERPA_ONNX_LOGE("%s: truncated .npy header", path.c_str());
    return false;
  }

  uint32_t header_len = 0;
  if (header_len_field_size == 2) {
    uint16_t v;
    std::memcpy(&v, content.data() + header_len_offset, 2);
    header_len = v;
  } else {
    uint32_t v;
    std::memcpy(&v, content.data() + header_len_offset, 4);
    header_len = v;
  }

  size_t data_offset = header_len_offset + header_len_field_size + header_len;
  if (content.size() < data_offset) {
    SHERPA_ONNX_LOGE("%s: truncated .npy header dict", path.c_str());
    return false;
  }

  std::string header(content.data() + header_len_offset + header_len_field_size,
                     header_len);

  std::string descr = StripQuotes(ExtractField(header, "descr"));
  std::string fortran = ExtractField(header, "fortran_order");
  std::string shape_str = ExtractField(header, "shape");

  if (fortran.find("True") != std::string::npos) {
    SHERPA_ONNX_LOGE(
        "%s: fortran_order=True .npy files are not supported", path.c_str());
    return false;
  }

  out->shape = ParseShapeTuple(shape_str);
  out->dtype = descr;

  int64_t n = out->NumElements();
  size_t data_bytes = content.size() - data_offset;

  if (descr == "<f4") {
    if (data_bytes < static_cast<size_t>(n) * sizeof(float)) {
      SHERPA_ONNX_LOGE("%s: truncated float32 data", path.c_str());
      return false;
    }
    out->f32.resize(n);
    std::memcpy(out->f32.data(), content.data() + data_offset,
               n * sizeof(float));
  } else if (descr == "<i8") {
    if (data_bytes < static_cast<size_t>(n) * sizeof(int64_t)) {
      SHERPA_ONNX_LOGE("%s: truncated int64 data", path.c_str());
      return false;
    }
    out->i64.resize(n);
    std::memcpy(out->i64.data(), content.data() + data_offset,
               n * sizeof(int64_t));
  } else {
    SHERPA_ONNX_LOGE(
        "%s: unsupported .npy dtype '%s' (only <f4 and <i8 are supported)",
        path.c_str(), descr.c_str());
    return false;
  }

  return true;
}

}  // namespace

bool ReadZeroTtsNpy(const std::string &path, ZeroTtsNpyArray *out) {
  std::vector<char> content = ReadFile(path);
  if (content.empty()) {
    SHERPA_ONNX_LOGE("Failed to read %s", path.c_str());
    return false;
  }
  return ParseNpy(content, path, out);
}

template <typename Manager>
bool ReadZeroTtsNpy(Manager *mgr, const std::string &path,
                    ZeroTtsNpyArray *out) {
  std::vector<char> content = ReadFile(mgr, path);
  if (content.empty()) {
    SHERPA_ONNX_LOGE("Failed to read %s", path.c_str());
    return false;
  }
  return ParseNpy(content, path, out);
}

#if __ANDROID_API__ >= 9
template bool ReadZeroTtsNpy(AAssetManager *mgr, const std::string &path,
                             ZeroTtsNpyArray *out);
#endif

#if __OHOS__
template bool ReadZeroTtsNpy(NativeResourceManager *mgr,
                             const std::string &path, ZeroTtsNpyArray *out);
#endif

}  // namespace sherpa_onnx
