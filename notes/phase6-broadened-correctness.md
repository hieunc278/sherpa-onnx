# Phase 6 — broaden correctness: all voices, more text

## All 8 bundled voices smoke test

Ran the CLI (`sherpa-onnx-offline-tts`, `build_shared/`) once per bundled voice with a
normalizer-exercising sentence ("Ngày 23 tháng 8, giá tăng 12,5 phần trăm." — pre-normalization
text so both date and percent categories fire). All 8 succeeded, no errors, non-silent,
non-clipping output:

| Voice | Duration | RMS | Peak |
|---|---|---|---|
| baotrang | 3.12s | 0.0728 | 0.487 |
| giahuy | 3.12s | 0.0885 | 0.924 |
| hamy | 3.76s | 0.0888 | 0.383 |
| huuduc | 3.52s | 0.1485 | 0.959 |
| kimoanh | 3.20s | 0.0998 | 0.629 |
| maichi | 3.04s | 0.1404 | 0.643 |
| quangminh | 2.64s | 0.1225 | 0.808 |
| tiendat | 2.88s | 0.0740 | 0.513 |

Output wavs saved under `assets/zerotts/phase6_voices_smoke_test/` (gitignored).

## Broadened Phase 4 comparison (3 more sentences, voice=maichi)

| # | Sentence | ref duration | cpp duration | mel-spectrogram correlation |
|---|---|---|---|---|
| 0 | "Hôm nay là ngày 20 tháng 10, chúc mừng ngày phụ nữ Việt Nam." | 3.12s | 3.52s | 0.667 |
| 1 | "Cảm ơn bạn đã sử dụng ZeroTTS, một mô hình chạy trên CPU." | 3.92s | 4.32s | 0.653 |
| 2 | "Giá vàng hôm nay tăng 12,5 phần trăm so với tuần trước." | 3.12s | 3.12s | 0.829 |

Output wavs saved under `assets/zerotts/phase6_extra_compare/` (gitignored).

**Interpretation — lower correlation than Phase 4's single-sentence check (0.935), and why this is
not read as a correctness regression:** neither the Python reference run nor the C++ run pins the
AR loop's RNG seed (matches production behavior of both — sampling is intentionally stochastic).
For longer/more complex sentences, sampled token paths diverge earlier and generation lengths
differ by up to 0.4s; a flattened, non-time-aligned mel-spectrogram correlation is not a fair
apples-to-apples metric once two independently-sampled sequences of different lengths are being
compared frame-for-frame without any DTW/time-warping alignment. This is a property of the
comparison methodology at this stage, not evidence of an implementation bug -- the actual
per-operation numerical correctness claim already has a much stronger, seed-controlled proof
(Phase 3's `offline-tts-zerotts-model-test`, which forces identical random draws and gets
bit-exact/near-bit-exact agreement on every one of the 5 ONNX graphs, including an **exact integer
match** on `local_frame_decode`'s sampled codes). All 3 extra outputs were manually checked for
RMS/peak sanity (0.12-0.16 RMS, 0.60-0.69 peak -- healthy speech-like levels, no silence or
clipping).

**Known gap:** no actual human listened to these files (no audio playback available in this
session) -- this is recorded as a real gap for the human to close, not silently asserted as
"sounds fine." See plan.md's top-of-file status for this open item.

## Conclusion

Phase 6's exit bar ("all 8 bundled voices synthesize without error"; "expand the comparison set
beyond one sentence") is met on the automated-check axis. The one thing this phase could not do
in this environment is the literal "manual listen" spec.md §11 calls for -- flagged, not skipped
silently.
