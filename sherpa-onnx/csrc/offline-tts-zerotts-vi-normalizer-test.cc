// sherpa-onnx/csrc/offline-tts-zerotts-vi-normalizer-test.cc
//
// Copyright (c)  2026
//
// Proof for plan.md Phase 5: run the same sample set (numbers, dates,
// times, percent, fractions, abbreviations, code-switched English, and a
// deliberately-untouched long digit run) through both `normalize_vi_text()`
// (Python reference) and this C++ port, and diff string-for-string. The
// expected strings below are pinned literals captured from a one-off
// `python3 -c "from zerotts.text_norm.vi_normalizer import
// normalize_vi_text; ..."` run against the real Python reference -- see
// notes/phase0-vi-normalizer-scope.md for which categories this v1 port
// covers (and which are deliberately deferred).

#include "sherpa-onnx/csrc/offline-tts-zerotts-vi-normalizer.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(OfflineTtsZeroTtsViNormalizer, TestAgainstPythonReference) {
  struct Case {
    const char *input;
    const char *expected;
  };

  std::vector<Case> cases = {
      {"Ngày 23/8/2024 lúc 15h30.",
      "Ngày hai mươi ba tháng tám năm hai nghìn không trăm hai mươi tư lúc "
      "mười lăm giờ ba mươi phút."},
      {"Giá 1.250.000 đồng, tăng 12,5 điểm.",
      "Giá một triệu hai trăm năm mươi nghìn đồng, tăng mười hai phẩy năm "
      "điểm."},
      {"UBND TP.HCM và ATM của NHNN.",
      "Ủy ban Nhân dân Thành phố Hồ Chí Minh và máy rút tiền tự động của "
      "Ngân hàng Nhà nước."},
      {"Cuộc họp diễn ra lúc 9:05:30 sáng.",
      "Cuộc họp diễn ra lúc chín giờ năm phút ba mươi giây sáng."},
      {"Tôi có 3/4 số máy, còn ANTT thì ổn.",
      "Tôi có ba trên bốn số máy, còn an ninh trật tự thì ổn."},
      {"Hẹn gặp lúc 15h, đến muộn 5 phút.",
      "Hẹn gặp lúc mười lăm giờ, đến muộn năm phút."},
      {"Ngân sách tăng 25%, còn lãi suất 12,5 %.",
      "Ngân sách tăng hai mươi lăm phần trăm, còn lãi suất mười hai phẩy "
      "năm phần trăm."},
      {"Sinh năm 1990, hiện là ĐHQG sinh viên.",
      "Sinh năm một nghìn chín trăm chín mươi, hiện là Đại học Quốc gia "
      "sinh viên."},
      {"Contact: hello world today.", "Contact: hello world today."},
      {"Số điện thoại có 09123456789 nhé.",
      "Số điện thoại có 09123456789 nhé."},
  };

  for (const auto &c : cases) {
    EXPECT_EQ(NormalizeViText(c.input), c.expected) << "input: " << c.input;
  }
}

}  // namespace sherpa_onnx
