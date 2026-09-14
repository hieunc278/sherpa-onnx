// sherpa-onnx/csrc/offline-tts-zerotts-vi-normalizer.cc
//
// Copyright (c)  2026

#include "sherpa-onnx/csrc/offline-tts-zerotts-vi-normalizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts-zerotts-abbreviations-table.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-letter-case-table.h"
#include "sherpa-onnx/csrc/offline-tts-zerotts-nfc.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// ---------------------------------------------------------------------
// expand_number() port -- pure string/arithmetic logic, no regex needed.
// See notes/phase0-vi-normalizer-scope.md category #1.
// ---------------------------------------------------------------------

const char *DigitWord(char c) {
  switch (c) {
    case '0': return "không";
    case '1': return "một";
    case '2': return "hai";
    case '3': return "ba";
    case '4': return "bốn";
    case '5': return "năm";
    case '6': return "sáu";
    case '7': return "bảy";
    case '8': return "tám";
    case '9': return "chín";
    case ',': return "phẩy";
    default: return nullptr;
  }
}

std::string ExpandDigitByDigit(const std::string &digits) {
  std::string out;
  bool first = true;
  for (char c : digits) {
    if (c == ' ') continue;
    const char *w = DigitWord(c);
    if (!first) out += " ";
    out += w ? w : std::string(1, c);
    first = false;
  }
  return out;
}

std::string ApplySandhi(std::string text) {
  auto replace_all = [](std::string &s, const std::string &from,
                        const std::string &to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.size(), to);
      pos += to.size();
    }
  };
  replace_all(text, "mười năm", "mười lăm");
  replace_all(text, "mươi năm", "mươi lăm");
  replace_all(text, "mươi bốn", "mươi tư");
  replace_all(text, "mươi một", "mươi mốt");
  replace_all(text, "linh bốn", "linh tư");
  return text;
}

std::vector<std::string> SplitChunks(const std::string &number) {
  // Split into 3-digit chunks, most significant first, with a short
  // leading chunk when length isn't a multiple of 3.
  std::vector<std::string> chunks;
  int32_t n = static_cast<int32_t>(number.size());
  int32_t lead = n % 3;
  int32_t i = 0;
  if (lead) {
    chunks.push_back(number.substr(0, lead));
    i = lead;
  }
  for (; i < n; i += 3) {
    chunks.push_back(number.substr(i, 3));
  }
  return chunks;
}

// Speak one 3-digit chunk plus its scale word. Throws (returns false) if
// scale_index is out of range -- caller falls back to digit-by-digit.
bool SpeakChunk(const std::string &chunk, int32_t scale_index,
                std::string *out) {
  static const char *kUnitSingle[3] = {"", "mươi", "trăm"};
  static const char *kUnitTriple[7] = {"",     "nghìn",     "triệu", "tỷ",
                                       "nghìn tỷ", "triệu tỷ", "tỷ tỷ"};

  if (chunk == "000") {
    *out = "";
    return true;
  }

  std::string result;
  int32_t pos = static_cast<int32_t>(chunk.size()) - 1;
  int32_t len = static_cast<int32_t>(chunk.size());
  while (pos >= 0) {
    char c = chunk[pos];
    if (pos == len - 1 && c == '0' && len > 1) {
      // trailing zero: "hai mươi", not "hai mươi không"
    } else if (pos == len - 2 && (c == '1' || c == '0')) {
      if (pos == 0 && c == '0') {
        // no-op
      } else if (c == '1') {
        char next = chunk[pos + 1];
        result = (next != '0')
                    ? std::string("mười ") + DigitWord(next)
                    : "mười";
      } else {
        char next = chunk[pos + 1];
        result = (next != '0') ? std::string("linh ") + DigitWord(next) : "";
      }
    } else {
      std::string unit = kUnitSingle[len - pos - 1];
      std::string piece = std::string(DigitWord(c)) +
                          (unit.empty() ? "" : " " + unit);
      result = piece + (result.empty() ? "" : " " + result);
    }
    --pos;
  }

  if (scale_index >= 7) return false;

  // strip trailing/leading whitespace from result
  size_t b = result.find_first_not_of(' ');
  size_t e = result.find_last_not_of(' ');
  std::string trimmed = (b == std::string::npos) ? "" : result.substr(b, e - b + 1);

  std::string scale = kUnitTriple[scale_index];
  if (trimmed.empty()) {
    *out = scale;
  } else if (scale.empty()) {
    *out = trimmed;
  } else {
    *out = trimmed + " " + scale;
  }
  return true;
}

}  // namespace

std::string ZeroTtsExpandNumber(const std::string &number_in) {
  std::string number = number_in;
  std::string sign;

  if (!number.empty() && (number[0] == '-' || number[0] == '+')) {
    sign = (number[0] == '+') ? "cộng" : "trừ";
    number = number.substr(1);
  }

  while (number.size() > 1 && number[0] == '0' &&
        std::isdigit(static_cast<unsigned char>(number[1]))) {
    number = number.substr(1);
  }

  // strip anything that isn't a digit, '.', or ','
  std::string cleaned;
  for (char c : number) {
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == ',') {
      cleaned.push_back(c);
    }
  }
  number = cleaned;

  std::string decimal_part;
  int32_t comma_count = static_cast<int32_t>(
      std::count(number.begin(), number.end(), ','));
  int32_t dot_count = static_cast<int32_t>(
      std::count(number.begin(), number.end(), '.'));

  if (comma_count == 1) {
    std::string no_dots;
    for (char c : number)
      if (c != '.') no_dots.push_back(c);
    size_t comma_pos = no_dots.find(',');
    decimal_part = "phẩy " + ExpandDigitByDigit(no_dots.substr(comma_pos + 1));
    number = no_dots.substr(0, comma_pos);
  } else if (dot_count == 1 &&
            (number.size() - number.find('.')) <= 3) {
    std::string no_commas;
    for (char c : number)
      if (c != ',') no_commas.push_back(c);
    size_t dot_pos = no_commas.find('.');
    decimal_part = "chấm " + ExpandDigitByDigit(no_commas.substr(dot_pos + 1));
    number = no_commas.substr(0, dot_pos);
  } else {
    std::string no_dots;
    for (char c : number)
      if (c != '.') no_dots.push_back(c);
    number = no_dots;
  }

  if (number.empty()) number = "0";

  std::vector<std::string> chunks = SplitChunks(number);
  std::vector<std::string> parts;
  bool ok = true;
  for (size_t i = 0; i < chunks.size() && ok; ++i) {
    std::string spoken;
    ok = SpeakChunk(chunks[i], static_cast<int32_t>(chunks.size() - i - 1),
                    &spoken);
    if (ok && !spoken.empty()) parts.push_back(spoken);
  }

  if (!ok) {
    // number too large for the scale table -- fall back to digit-by-digit,
    // mirroring expand_number()'s `except IndexError: return
    // expand_digit(number)`. Uses the original (unsigned, uncleaned) input.
    return ExpandDigitByDigit(number_in);
  }

  std::string joined;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) joined += " ";
    joined += parts[i];
  }
  joined = ApplySandhi(joined);

  std::string result;
  if (!sign.empty()) result += sign + " ";
  result += joined;
  if (!decimal_part.empty()) result += " " + decimal_part;

  // trim
  size_t b = result.find_first_not_of(' ');
  size_t e = result.find_last_not_of(' ');
  return (b == std::string::npos) ? "" : result.substr(b, e - b + 1);
}

namespace {

// ---------------------------------------------------------------------
// Hand-rolled left-to-right scanner (v1 categories only -- see
// notes/phase0-vi-normalizer-scope.md).
// ---------------------------------------------------------------------

bool IsAsciiDigit(char32_t c) { return c >= U'0' && c <= U'9'; }

bool IsUpperLetter(char32_t cp) {
  if (cp > 0xFFFF) return false;
  return std::binary_search(kUpperLetterCodepoints,
                            kUpperLetterCodepoints + kUpperLetterTableSize,
                            static_cast<uint16_t>(cp));
}

bool IsLowerLetter(char32_t cp) {
  if (cp > 0xFFFF) return false;
  return std::binary_search(kLowerLetterCodepoints,
                            kLowerLetterCodepoints + kLowerLetterTableSize,
                            static_cast<uint16_t>(cp));
}

bool IsWordChar(char32_t cp) {
  return IsUpperLetter(cp) || IsLowerLetter(cp) || IsAsciiDigit(cp) ||
        cp == U'_';
}

// month==4 is spoken "tư", never "bốn" (vi_normalizer.py's _month()).
std::string MonthWord(const std::string &digits) {
  std::string trimmed = digits;
  size_t nz = trimmed.find_first_not_of('0');
  std::string stripped = (nz == std::string::npos) ? "0" : trimmed.substr(nz);
  if (stripped == "4") return "tư";
  return ZeroTtsExpandNumber(digits);
}

std::string SpeakTime(const std::string &h, const std::string *m,
                     const std::string *s) {
  std::string out = ZeroTtsExpandNumber(h) + " giờ";
  bool m_has_content = m && (s != nullptr || m->find_first_not_of('0') !=
                                                  std::string::npos);
  if (m_has_content) {
    out += " " + ZeroTtsExpandNumber(*m) + " phút";
  }
  if (s) {
    out += " " + ZeroTtsExpandNumber(*s) + " giây";
  }
  return out;
}

std::unordered_map<std::string, std::string> BuildAbbrevTable() {
  std::unordered_map<std::string, std::string> table;
  for (int32_t i = 0; i < kZeroTtsAbbreviationsTableSize; ++i) {
    table.emplace(kZeroTtsAbbreviationsTable[i].abbr,
                 kZeroTtsAbbreviationsTable[i].reading);
  }
  return table;
}

const std::unordered_map<std::string, std::string> &AbbrevTable() {
  static const auto table = BuildAbbrevTable();
  return table;
}

// Digit run starting at i, up to max_len codepoints, at least 1. Returns
// consumed length (0 if none).
size_t ReadDigits(const std::u32string &s, size_t i, size_t max_len) {
  size_t n = 0;
  while (i + n < s.size() && n < max_len && IsAsciiDigit(s[i + n])) ++n;
  return n;
}

std::string U32ToAscii(const std::u32string &s, size_t start, size_t len) {
  std::string out;
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) out.push_back(static_cast<char>(s[start + i]));
  return out;
}

struct Match {
  bool ok = false;
  size_t consumed = 0;
  std::u32string replacement;
};

Match MatchDate(const std::u32string &s, size_t i) {
  size_t start = i;
  size_t dlen = ReadDigits(s, i, 2);
  if (dlen == 0) return {};
  size_t p = i + dlen;
  if (p >= s.size()) return {};
  char32_t sep = s[p];
  if (sep != U'/' && sep != U'-' && sep != U'.') return {};
  ++p;
  size_t mlen = ReadDigits(s, p, 2);
  if (mlen == 0) return {};
  size_t p2 = p + mlen;
  if (p2 >= s.size() || s[p2] != sep) return {};
  ++p2;
  size_t ylen = ReadDigits(s, p2, 4);
  if (ylen != 4) return {};
  size_t end = p2 + ylen;

  // guards: not preceded by digit/sep/dot, not followed by digit/sep, and
  // not followed by ".digit"
  if (start > 0) {
    char32_t before = s[start - 1];
    if (IsAsciiDigit(before) || before == U'/' || before == U'.' ||
        before == U'-')
      return {};
  }
  if (end < s.size()) {
    char32_t after = s[end];
    if (IsAsciiDigit(after) || after == U'/' || after == U'-') return {};
    if (after == U'.' && end + 1 < s.size() && IsAsciiDigit(s[end + 1]))
      return {};
  }

  std::string day = U32ToAscii(s, i, dlen);
  std::string month = U32ToAscii(s, i + dlen + 1, mlen);
  std::string year = U32ToAscii(s, p2, ylen);

  int32_t day_val = std::stoi(day), month_val = std::stoi(month);
  if (day_val < 1 || day_val > 31 || month_val < 1 || month_val > 12) return {};

  std::string result = ZeroTtsExpandNumber(day) + " tháng " +
                       MonthWord(month) + " năm " + ZeroTtsExpandNumber(year);
  return {true, end - start, Utf8ToUtf32(result)};
}

Match MatchMonthYear(const std::u32string &s, size_t i) {
  size_t start = i;
  size_t mlen = ReadDigits(s, i, 2);
  if (mlen == 0) return {};
  size_t p = i + mlen;
  if (p >= s.size()) return {};
  char32_t sep = s[p];
  if (sep != U'/' && sep != U'-') return {};
  ++p;
  size_t ylen = ReadDigits(s, p, 4);
  if (ylen != 4) return {};
  size_t end = p + ylen;

  if (start > 0) {
    char32_t before = s[start - 1];
    if (IsAsciiDigit(before) || before == U'/' || before == U'.' ||
        before == U'-')
      return {};
  }
  if (end < s.size()) {
    char32_t after = s[end];
    if (IsAsciiDigit(after) || after == U'/' || after == U'-') return {};
    if (after == U'.' && end + 1 < s.size() && IsAsciiDigit(s[end + 1]))
      return {};
  }

  std::string month = U32ToAscii(s, i, mlen);
  std::string year = U32ToAscii(s, p, ylen);
  int32_t month_val = std::stoi(month);
  int32_t year_val = std::stoi(year);
  if (month_val < 1 || month_val > 12) return {};
  if (year_val < 1000 || year_val > 2999) return {};

  // Look back up to 8 codepoints for a literal "tháng" right before.
  bool has_thang_before = false;
  if (start >= 5) {
    std::u32string thang = Utf8ToUtf32("tháng");
    size_t look_start = start >= 8 ? start - 8 : 0;
    std::u32string context(s.begin() + look_start, s.begin() + start);
    if (context.size() >= thang.size() &&
        context.compare(context.size() - thang.size(), thang.size(), thang) ==
            0) {
      has_thang_before = true;
    }
  }

  std::string lead = has_thang_before ? "" : "tháng ";
  std::string result =
      lead + MonthWord(month) + " năm " + ZeroTtsExpandNumber(year);
  return {true, end - start, Utf8ToUtf32(result)};
}

Match MatchTimeHms(const std::u32string &s, size_t i) {
  size_t start = i;
  size_t hlen = ReadDigits(s, i, 2);
  if (hlen == 0) return {};
  size_t p = i + hlen;
  if (p >= s.size()) return {};
  char32_t sep1 = s[p];
  if (sep1 != U':' && sep1 != U'h' && sep1 != U'g') return {};
  ++p;
  size_t mlen = ReadDigits(s, p, 2);
  if (mlen == 0) return {};
  size_t p2 = p + mlen;
  if (p2 >= s.size()) return {};
  char32_t sep2 = s[p2];
  if (sep2 != U':' && sep2 != U'm' && sep2 != U'p') return {};
  ++p2;
  size_t slen = ReadDigits(s, p2, 2);
  if (slen == 0) return {};
  size_t end = p2 + slen;

  if (start > 0) {
    char32_t before = s[start - 1];
    if (IsAsciiDigit(before) || before == U':') return {};
  }
  if (end < s.size()) {
    char32_t after = s[end];
    if (IsAsciiDigit(after) || after == U':') return {};
  }

  std::string h = U32ToAscii(s, i, hlen);
  std::string m = U32ToAscii(s, p, mlen);
  std::string sec = U32ToAscii(s, p2, slen);
  if (std::stoi(h) > 23 || std::stoi(m) > 59 || std::stoi(sec) > 59) return {};

  std::string result = SpeakTime(h, &m, &sec);
  return {true, end - start, Utf8ToUtf32(result)};
}

Match MatchTimeHm(const std::u32string &s, size_t i) {
  size_t start = i;
  size_t hlen = ReadDigits(s, i, 2);
  if (hlen == 0) return {};
  size_t p = i + hlen;
  if (p >= s.size()) return {};
  char32_t sep = s[p];
  if (sep != U':' && sep != U'h' && sep != U'g') return {};
  ++p;
  size_t mlen = ReadDigits(s, p, 2);
  if (mlen != 2) return {};  // exactly 2 digits for this shape
  size_t end = p + mlen;

  if (start > 0) {
    char32_t before = s[start - 1];
    if (IsAsciiDigit(before) || before == U':') return {};
  }
  if (end < s.size()) {
    char32_t after = s[end];
    if (sep == U':') {
      if (IsAsciiDigit(after) || after == U':') return {};
    } else {
      if (IsAsciiDigit(after) || after == U'h' || after == U'g') return {};
    }
  }

  std::string h = U32ToAscii(s, i, hlen);
  std::string m = U32ToAscii(s, p, mlen);
  if (std::stoi(h) > 23 || std::stoi(m) > 59) return {};

  std::string result = SpeakTime(h, &m, nullptr);
  return {true, end - start, Utf8ToUtf32(result)};
}

Match MatchTimeBareH(const std::u32string &s, size_t i) {
  size_t start = i;
  size_t hlen = ReadDigits(s, i, 2);
  if (hlen == 0) return {};
  size_t p = i + hlen;
  if (p >= s.size()) return {};
  char32_t sep = s[p];
  if (sep != U'h' && sep != U'g') return {};
  size_t end = p + 1;

  if (start > 0) {
    char32_t before = s[start - 1];
    if (IsAsciiDigit(before) || before == U':') return {};
  }
  if (end < s.size() && IsWordChar(s[end])) return {};

  std::string h = U32ToAscii(s, i, hlen);
  if (std::stoi(h) > 23) return {};

  std::string result = SpeakTime(h, nullptr, nullptr);
  return {true, end - start, Utf8ToUtf32(result)};
}

// Shared "read a number-like run" for percent/fraction/number: digits plus
// internal '.'/','; no arithmetic operators (v1 simplification, see
// notes/phase0-vi-normalizer-scope.md category #1).
size_t ReadNumberRun(const std::u32string &s, size_t i) {
  size_t n = 0;
  size_t start = i;
  if (i + n < s.size() && (s[i + n] == U'-' || s[i + n] == U'+')) ++n;
  if (start + n >= s.size() || !IsAsciiDigit(s[start + n])) return 0;
  while (start + n < s.size() &&
        (IsAsciiDigit(s[start + n]) || s[start + n] == U'.' ||
         s[start + n] == U',')) {
    ++n;
  }
  return n;
}

Match MatchPercent(const std::u32string &s, size_t i) {
  size_t start = i;
  if (start > 0) {
    char32_t before = s[start - 1];
    if (IsWordChar(before) || before == U'.' || before == U',') return {};
  }
  size_t nlen = ReadNumberRun(s, i);
  if (nlen == 0) return {};
  size_t p = i + nlen;
  while (p < s.size() && s[p] == U' ') ++p;
  if (p >= s.size() || s[p] != U'%') return {};
  size_t end = p + 1;
  if (end < s.size() && IsWordChar(s[end])) return {};

  std::string raw = U32ToAscii(s, i, nlen);
  while (!raw.empty() && (raw.back() == '.' || raw.back() == ',')) {
    raw.pop_back();
  }
  std::string result = ZeroTtsExpandNumber(raw) + " phần trăm";
  return {true, end - start, Utf8ToUtf32(result)};
}

Match MatchFraction(const std::u32string &s, size_t i) {
  size_t start = i;
  if (start > 0) {
    char32_t before = s[start - 1];
    if (IsWordChar(before) || before == U'/' || before == U'.' ||
        before == U',')
      return {};
  }
  size_t alen = ReadDigits(s, i, 9);
  if (alen == 0) return {};
  size_t p = i + alen;
  while (p < s.size() && s[p] == U' ') ++p;
  if (p >= s.size() || s[p] != U'/') return {};
  ++p;
  while (p < s.size() && s[p] == U' ') ++p;
  size_t blen = ReadDigits(s, p, 9);
  if (blen == 0) return {};
  size_t end = p + blen;

  if (end < s.size()) {
    char32_t after = s[end];
    if (IsWordChar(after) || after == U'/' || after == U',') return {};
    if (after == U'.' && end + 1 < s.size() && IsAsciiDigit(s[end + 1]))
      return {};
  }

  std::string a = U32ToAscii(s, i, alen);
  std::string b = U32ToAscii(s, p, blen);
  std::string result =
      ZeroTtsExpandNumber(a) + " trên " + ZeroTtsExpandNumber(b);
  return {true, end - start, Utf8ToUtf32(result)};
}

Match MatchNumber(const std::u32string &s, size_t i) {
  size_t start = i;
  if (start > 0) {
    char32_t before = s[start - 1];
    if (IsWordChar(before) || before == U'.' || before == U',') return {};
  }
  size_t nlen = ReadNumberRun(s, i);
  if (nlen == 0) return {};
  size_t end = i + nlen;

  if (end < s.size()) {
    char32_t after = s[end];
    if (after == U'.' || after == U',') {
      if (end + 1 < s.size() && IsAsciiDigit(s[end + 1])) return {};
    }
  }

  std::string raw = U32ToAscii(s, i, nlen);
  std::string suffix;
  while (!raw.empty() && (raw.back() == '.' || raw.back() == ',')) {
    suffix.insert(suffix.begin(), raw.back());
    raw.pop_back();
  }
  if (raw.empty()) return {};

  // long separator-less digit run -> identifier, leave unchanged
  bool all_digits = raw.find_first_not_of("0123456789") == std::string::npos;
  if (all_digits && raw.size() > 8) return {};

  // digit-letter-digit compound ("1m65") -> leave unchanged
  if (end < s.size() && IsLowerLetter(s[end]) && end + 1 < s.size() &&
      IsAsciiDigit(s[end + 1])) {
    return {};
  }

  std::string result = ZeroTtsExpandNumber(raw) + suffix;
  return {true, end - start, Utf8ToUtf32(result)};
}

Match MatchAbbreviation(const std::u32string &s, size_t i) {
  size_t start = i;
  if (start > 0) {
    char32_t before = s[start - 1];
    if (IsUpperLetter(before) || IsLowerLetter(before) || before == U'.')
      return {};
  }
  if (!IsUpperLetter(s[start])) return {};

  size_t p = start + 1;
  while (p < s.size() && (IsUpperLetter(s[p]) || IsAsciiDigit(s[p]))) ++p;
  // allow dotted continuations: ('.' UPPER (UPPER|digit)*)*
  size_t end = p;
  while (end < s.size() && s[end] == U'.' && end + 1 < s.size() &&
        IsUpperLetter(s[end + 1])) {
    size_t q = end + 1;
    while (q < s.size() && (IsUpperLetter(s[q]) || IsAsciiDigit(s[q]))) ++q;
    end = q;
  }

  if (end - start < 2) return {};
  if (end < s.size() && (IsLowerLetter(s[end]) || IsAsciiDigit(s[end])))
    return {};

  std::string token;
  for (size_t k = start; k < end; ++k) {
    token += Utf32ToUtf8(s[k]);
  }

  const auto &table = AbbrevTable();
  auto it = table.find(token);
  if (it == table.end()) {
    std::string stripped;
    for (char c : token)
      if (c != '.' && c != '-') stripped.push_back(c);
    it = table.find(stripped);
    if (it == table.end()) return {};
  }

  return {true, end - start, Utf8ToUtf32(it->second)};
}

// Very small, deliberately conservative URL/email protector: consumes a
// run of non-whitespace codepoints starting at i if it looks like a URL
// (http://, https://, ftp://, www.) or contains an '@' with a '.' after it
// (an email address). v1 simplification of _PROTECTED_RE -- see
// notes/phase0-vi-normalizer-scope.md category #16.
size_t MatchProtectedSpan(const std::u32string &s, size_t i) {
  auto starts_with = [&](const char *lit) {
    std::u32string u = Utf8ToUtf32(lit);
    if (i + u.size() > s.size()) return false;
    return s.compare(i, u.size(), u) == 0;
  };
  bool looks_like_url =
      starts_with("http://") || starts_with("https://") ||
      starts_with("ftp://") || starts_with("www.");

  size_t run_end = i;
  while (run_end < s.size() && s[run_end] != U' ' && s[run_end] != U'\t' &&
        s[run_end] != U'\n') {
    ++run_end;
  }
  if (run_end == i) return 0;

  if (looks_like_url) return run_end - i;

  bool has_at = false, has_dot_after_at = false;
  for (size_t k = i; k < run_end; ++k) {
    if (s[k] == U'@') {
      has_at = true;
    } else if (has_at && s[k] == U'.') {
      has_dot_after_at = true;
    }
  }
  if (has_at && has_dot_after_at) return run_end - i;

  return 0;
}

bool IsAlnumForSpacing(char32_t cp) {
  return IsAsciiDigit(cp) || IsUpperLetter(cp) || IsLowerLetter(cp);
}

}  // namespace

std::string NormalizeViText(const std::string &text) {
  if (text.empty()) return text;

  std::string nfc = ZeroTtsToNfc(text);
  std::u32string s = Utf8ToUtf32(nfc);
  std::u32string out;
  out.reserve(s.size());

  size_t i = 0;
  while (i < s.size()) {
    size_t protected_len = MatchProtectedSpan(s, i);
    if (protected_len > 0) {
      out.append(s, i, protected_len);
      i += protected_len;
      continue;
    }

    Match m;
    if (!m.ok) m = MatchDate(s, i);
    if (!m.ok) m = MatchMonthYear(s, i);
    if (!m.ok) m = MatchTimeHms(s, i);
    if (!m.ok) m = MatchTimeHm(s, i);
    if (!m.ok) m = MatchTimeBareH(s, i);
    if (!m.ok) m = MatchPercent(s, i);
    if (!m.ok) m = MatchFraction(s, i);
    if (!m.ok) m = MatchNumber(s, i);
    if (!m.ok) m = MatchAbbreviation(s, i);
    if (!m.ok && s[i] == U'@') {
      m = {true, 1, Utf8ToUtf32("a còng")};
    }

    if (m.ok) {
      char32_t before = (i > 0) ? s[i - 1] : U' ';
      char32_t after = (i + m.consumed < s.size()) ? s[i + m.consumed] : U' ';
      std::u32string expanded = m.replacement;
      if (IsAlnumForSpacing(before) &&
          (expanded.empty() || expanded.front() != U' ')) {
        out.push_back(U' ');
      }
      out += expanded;
      if (IsAlnumForSpacing(after) &&
          (expanded.empty() || expanded.back() != U' ')) {
        out.push_back(U' ');
      }
      i += m.consumed;
    } else {
      out.push_back(s[i]);
      ++i;
    }
  }

  std::string result = Utf32ToUtf8(out);

  // collapse runs of 2+ spaces/tabs left behind by expansions (matches
  // vi_normalizer.py's final `re.sub(r"[ \t]{2,}", " ", ...)`).
  std::string collapsed;
  collapsed.reserve(result.size());
  for (size_t k = 0; k < result.size(); ++k) {
    char c = result[k];
    if (c == ' ' || c == '\t') {
      collapsed.push_back(' ');
      while (k + 1 < result.size() &&
            (result[k + 1] == ' ' || result[k + 1] == '\t')) {
        ++k;
      }
    } else {
      collapsed.push_back(c);
    }
  }
  return collapsed;
}

}  // namespace sherpa_onnx
