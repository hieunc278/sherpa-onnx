# Plan: ZeroTTS support in sherpa-onnx

- **Status:** draft — pending author review
- **Derived from:** [spec.md](spec.md), [intent/zerotts-sherpa-onnx-support.md](intent/zerotts-sherpa-onnx-support.md)
- **Date:** 2026-09-14

## How to read this plan

Each phase produces something concretely checkable ("Proof") before moving to the next — no phase
depends on a later phase's code existing yet. Phases are ordered to front-load the riskiest/most
uncertain pieces (the AR loop, the tokenizer) rather than saving them for last, and to keep the
Android/edge work from blocking the desktop-first correctness work. Given solo/no-CI, "proof" here
means a manual, repeatable check I run myself before considering a phase done — not a hope.

Target build platform throughout: desktop Linux first (fast iteration), cross-checked on
Android/AAOS via NDK (arm64-v8a) starting at Phase 8, using sherpa-onnx's existing
`build-android-arm64-v8a.sh`.

## Phase 0 — Spikes (resolve spec.md §12's deferred unknowns before writing real code)

No production code yet. Just answers, written down in a **structured, checkable** form — not
narrative notes. Given no cross-review, "I looked into it" isn't a strong enough bar for something
Phases 1–4 build directly on; each item below has an explicit required output and a done/not-done
check, not a subjective one.

### 0.1 — `voice.bin` byte layout (resolves spec §6/§12.3)

Fetch one real `voices/maichi/voice.bin` + `meta.json`. Produce
`notes/phase0-voice-bin-format.md` containing, at minimum:
- File size in bytes, and `meta.json`'s stated `n_voice_queries`/`d_model` (or wherever those are
  recorded)
- The arithmetic check: does `file_size_bytes == n_voice_queries * d_model * 4` (float32, no
  header)? Show the actual numbers, not just "yes/no"
- A hex dump of the first 32 bytes, annotated with what they decode to as float32, to catch a
  header/magic-number case the size check alone might miss
- An explicit one-line conclusion: exact byte layout (e.g. "raw float32, row-major, no header,
  confirmed by size match") that Phase 2's parser can be written against verbatim

**Done when:** the arithmetic check and hex dump are both present and agree with the stated
conclusion — not done if the conclusion is asserted without showing the check.

### 0.2 — `decode_step` tensor I/O (resolves spec §12.1)

Fetch `onnx/codec/codec_browser_onnx_meta.json` directly (not just re-reading `codec.py`'s
handling of it). Produce `notes/phase0-codec-decode-step-io.md` containing:
- A table of every input tensor: name, shape, dtype
- A table of every output tensor: name, shape, dtype
- For each tensor, an explicit tag: `data` (audio codes/output) vs. `state` (carried between
  `decode_chunk` calls — KV cache, ring buffer positions, layer offsets)
- Cross-check: grep `codec.py` for each tensor name to confirm it's actually referenced there by
  that exact name (catches stale/renamed fields in the meta file)

**Done when:** every tensor in the meta file has a name+shape+dtype+tag row — not done if any
tensor is left unlisted or tagged "unclear."

### 0.3 — Real generation hyperparameter defaults (resolves spec §12.2)

Read `synthesizer.py`'s actual function signatures. **Directly edit spec.md §4**, replacing every
value currently marked `// placeholder` with the real default, each annotated with its source,
e.g. `int32_t max_frames = 1500;  // synthesizer.py: synthesize(max_frames=1500)`.

**Done when:** `grep -c placeholder spec.md` returns `0`. This is a literal, checkable completion
condition, not a judgment call.

### 0.4 — Vietnamese normalizer scope (resolves spec §12.5)

Read `vi_normalizer.py` + `abbreviations.txt` in full. Produce
`notes/phase0-vi-normalizer-scope.md` as a checklist, one row per normalization category actually
present in the source (numbers, dates, currency, acronyms/abbreviations, code-switch handling,
punctuation, and whatever else is actually there — don't presume this list is exhaustive until
the source is read), each marked `Port for v1` or `Defer`, with a one-line reason for every
deferred item.

**Done when:** every distinct normalization rule/category found in the source file appears as its
own row with a decision — not done if categories are grouped/summarized away.

### Phase 0 exit criteria

All four sub-items' "Done when" conditions hold, **and** spec.md's §4 config table and §12 deferred
list have been updated to reflect what was actually found (removing/resolving the items now
answered) before Phase 1 starts.

## Phase 1 — Tokenizer

Files: `sherpa-onnx/csrc/offline-tts-zerotts-tokenizer.{h,cc}`

Implement the self-contained `tokenizer.json` BPE loader per spec §5 (NFC normalize → whitespace
collapse → whitespace/punctuation/digit pre-tokenizer splits → BPE merge → `[<bos>, ...ids, <eot>]`
wrapping).

**Proof:** unit test (`offline-tts-zerotts-tokenizer-test.cc`, following the existing
`sentence-piece-tokenizer-test.cc` pattern) encoding a fixed set of Vietnamese/English/mixed
strings, asserting identical token ID sequences against the real Python `tokenizers`-backed
reference (run once, offline, to generate the expected-output fixture).

## Phase 2 — Config + asset loading (no inference yet)

Files: `offline-tts-zerotts-model-config.{h,cc}`, `offline-tts-zerotts-voice.{h,cc}`, a small
shared `.npy` reader (per spec §6's decision) for `null_voice_emb.npy`/`silence_frame.npy`.

**Proof:** load all 8 real voice packs + both `.npy` files, assert expected shapes
(`(1, n_voice_queries, d_model)`) and that `Validate()` correctly rejects missing/malformed paths.
No audio produced yet — this phase is purely "do the files parse correctly."

## Phase 3 — Model wrapper: individual ONNX sessions

Files: `offline-tts-zerotts-model.{h,cc}`

Wrap the 5 `Ort::Session`s (`text_encoder`, `prefix_step`, `local_frame_decode`,
`codec_decode_full`, `codec_decode_step`) with one method each
(`RunTextEncoder`/`RunPrefixStep`/`RunLocalFrameDecode`/`RunCodecDecodeFull`/`RunCodecDecodeStep`),
using the real tensor names/shapes from Phase 0.

**Proof:** call each method once in isolation with real (not synthetic) intermediate tensors
captured from a single instrumented run of the Python reference implementation (add a temporary
debug dump to `synthesizer.py` locally, not committed), and diff shapes + a numerical tolerance
check against the Python-side output for that same call. This catches ONNX I/O binding mistakes
before they're buried inside a full loop.

## Phase 4 — Offline generation loop, end to end

Files: `offline-tts-zerotts-impl.h` (implements `OfflineTtsZeroTtsImpl::Generate`, per spec §7)

Wire Phase 1–3 together into the full loop: text → tokenize → text_encoder → prefix_step (cold
start) → loop(local_frame_decode → prefix_step) → stop condition → codec_decode_full → waveform.

**Proof:** one fixed sentence + `voice="maichi"`, compare the produced waveform against the
reference Python `zerotts` package's output for the identical input (spec §11's reference-parity
check) — duration match, spectrogram correlation, and a manual listen. This is the first point
audio actually comes out; it's the main correctness gate for the whole feature.

## Phase 5 — Vietnamese text normalizer

Files: `offline-tts-zerotts-vi-normalizer.{h,cc}`

Port whatever Phase 0.4 scoped as realistic from `vi_normalizer.py`/`abbreviations.txt`, wired in
before tokenization in the `Generate` path.

**Proof:** run the same sample set (numbers, dates, acronyms, code-switched English) through both
`normalize_vi_text()` and the C++ port, diff outputs string-for-string.

## Phase 6 — Broaden correctness: all voices, more text

No new files — exercising Phase 1–5 more widely.

**Proof:** smoke-test all 8 bundled voices synthesize without error; expand the Phase 4 comparison
set beyond one sentence to catch normalizer/tokenizer edge cases the single-sentence check missed.

## Phase 7 — Wire into sherpa-onnx's dispatch + build system

Files: edit `offline-tts-impl.cc` (new dispatch branch keyed on
`config.model.zerotts.text_encoder` non-empty, per spec §3) and
`sherpa-onnx/csrc/CMakeLists.txt` (add new `.cc` files to the existing TTS source list — no new
`SHERPA_ONNX_ENABLE_*` flag needed, confirmed in spec §3).

**Proof:** build and run the existing `sherpa-onnx-offline-tts` CLI binary end-to-end pointed at
ZeroTTS config fields, on desktop Linux — this is the point where it's a real sherpa-onnx backend,
not just an internal test harness.

## Phase 8 — Android/AAOS build

No new source files — validates Phase 1–7 under the NDK toolchain.

**Proof:** run `build-android-arm64-v8a.sh` and confirm the new sources compile and link cleanly
into `libsherpa-onnx-jni.so`; if feasible, run the offline synthesis path on a real device/emulator
and sanity-check output (even a basic "it produces non-silent audio of the right duration" check is
enough here — full parity checks already happened on desktop in Phase 4/6).

## Phase 9 — Streaming

Files: additive changes to `offline-tts.h`/`offline-tts-impl.h` (new `SupportsStreaming()` /
`GenerateStreaming()` virtuals, default no-op — per spec §8), plus `GenerateStreaming` implemented
in `offline-tts-zerotts-impl.h` using `codec_decode_step` with the doubling chunk schedule.

**Proof:**
- Every other existing TTS model still builds/runs unchanged (the additive-interface check that
  makes this safe per spec §8).
- Reassembled streamed-chunk output matches Phase 4's offline output for the same input.
- Time-to-first-chunk measured and compared to the ~70ms reference figure (desktop first, then
  Android in Phase 8's build once this phase is done — may need revisiting Phase 8 once streaming
  exists).

## Phase 10 — INT8 quantization (dynamic) + evaluation baseline

No new C++ source files — this phase is Python tooling + measurement.

1. Run `onnxruntime.quantization.quantize_dynamic` on all 5 graphs, producing `*.int8.onnx`
   variants (spec §9).
2. Point config at the int8 paths (no code change needed — confirmed in spec §9) and confirm it
   loads/runs through the exact same C++ path as Phase 4/9.
3. Run the full evaluation methodology from spec §10 (quality similarity, WER via an existing
   sherpa-onnx ASR backend, RTF/latency on the actual Android target from Phase 8, model size,
   manual spot-check across all 8 voices, and the AR-loop length-sensitivity check) to get the
   **first real fp32-vs-int8 numbers** — these numbers are what set the acceptance thresholds
   deferred in spec §12.7, so define the threshold immediately after seeing the first baseline,
   not before.

**Proof:** a filled-in results table (the metric list from spec §10) checked into the repo (e.g.
`evals/zerotts-int8-baseline.md`), with an explicit pass/fail call against the thresholds set in
step 3.

## Phase 11 — Static/QDQ quantization (only if Phase 10 isn't good enough)

Conditional phase — only if Phase 10's dynamic-quantization numbers miss the acceptance bar,
per spec §9's explicit "attempt only if dynamic quantization's quality loss isn't acceptable."

1. Assemble a small Vietnamese calibration text set (spanning normal + edge cases: numbers, dates,
   code-switched English) and run it through the fp32 pipeline to collect activation ranges.
2. Run `quantize_static`, re-run the Phase 10 evaluation methodology, compare against both the
   fp32 baseline and the Phase 10 dynamic-quantization numbers.

**Proof:** same results table format as Phase 10, three-way comparison (fp32 / dynamic-int8 /
static-int8).

## Sequencing notes / risks carried from spec.md §13

- Phases 1–7 are all desktop-only and don't depend on Android or quantization — if time is tight,
  this is a coherent, shippable-to-yourself milestone on its own (offline Vietnamese TTS on
  desktop) before touching streaming/edge/quantization at all.
- Phase 3's "capture real intermediate tensors from an instrumented Python run" step is the
  highest-leverage debugging investment in the whole plan — skipping it and going straight to
  Phase 4 risks debugging a 1500-step loop with only end-to-end pass/fail signal, which is much
  harder to root-cause.
- Phase 9 (streaming) is scoped after offline correctness is solid on purpose — it reuses the same
  AR loop and only changes how output is chunked/delivered, so there's little value validating it
  before Phase 4-6 are trustworthy.
- Phase 10/11's acceptance thresholds are deliberately *not* pre-set in spec.md or here — spec §10
  was explicit that they need a real fp32 baseline first. Don't skip straight to "good enough" by
  eyeballing it once numbers exist; write the threshold down before judging pass/fail against it.
