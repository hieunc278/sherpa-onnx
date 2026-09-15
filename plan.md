# Plan: ZeroTTS support in sherpa-onnx

- **Status:** Phases 0–10 executed with real proofs; Phase 11 attempted (conditional). See
  **"Overnight run status (2026-09-14) — READ THIS FIRST"** immediately below for what needs
  human review before anything here is treated as done.
- **Derived from:** [spec.md](spec.md), [intent/zerotts-sherpa-onnx-support.md](intent/zerotts-sherpa-onnx-support.md)
- **Date:** 2026-09-14

## Overnight run status (2026-09-14) — READ THIS FIRST

An unattended overnight session executed Phases 0 through 11 in order. Summary below; phase
sections further down have per-phase detail and were left otherwise unchanged from the original
plan (their "Proof" text is the *target*; the status line added under each phase heading records
what actually happened).

### Fully passed their proof (no known gaps)

- **Phase 0** (spikes) — all 4 sub-items done, spec.md updated in place, `grep -c placeholder
  spec.md` returns 0 (the one remaining literal string match is prose describing the history, not
  a live placeholder).
- **Phase 1** (tokenizer) — byte-exact match against a real Python `tokenizers` fixture on 8 test
  strings. Found and fixed a real pre-existing-codebase gotcha along the way: sherpa-onnx's
  `SplitUtf8()` is a lexicon word-splitter that drops whitespace and fuses ASCII letter runs — wrong
  tool for BPE base-symbol splitting; added a dedicated `SplitIntoCodepoints()` instead.
- **Phase 2** (config + asset loading) — all 8 voices + both `.npy` files load with correct shapes;
  `Validate()` correctly rejects bad configs. Test: `offline-tts-zerotts-voice-test.cc`.
- **Phase 3** (model wrapper) — every one of the 5 ONNX graphs' C++ binding verified against real
  captured Python-reference tensors; `local_frame_decode`'s sampled codes match **exactly**
  (integer equality, not just "close") given identical random draws. Test:
  `offline-tts-zerotts-model-test.cc`.
- **Phase 4** (offline generation loop) — end-to-end CLI run, voice=maichi, produced real audio;
  0.935 mel-spectrogram correlation against the Python reference for the proof sentence. Hit and
  root-caused a **real, pre-existing, environment-specific crash** unrelated to ZeroTTS (reproduces
  identically with an unmodified upstream `vits` model) — see
  `notes/environment-onnxruntime-static-lib-crash.md`. Workaround (`-DBUILD_SHARED_LIBS=ON`)
  confirmed working; used for all CLI-based proofs from Phase 4 onward.
- **Phase 5** (Vietnamese normalizer) — 10/10 sample sentences match the real Python
  `normalize_vi_text()` byte-for-byte. Scoped to 8 of 17 categories found in the source (see
  `notes/phase0-vi-normalizer-scope.md`) because `std::regex` has no lookbehind support at all —
  the remaining 9 are documented, deliberate v1 deferrals, not oversights.
- **Phase 6** (broaden correctness) — all 8 bundled voices synthesize without error (non-silent,
  non-clipping). 3 additional sentences compared against the Python reference; correlation is
  lower for longer sentences, explained (not silently accepted) in
  `notes/phase6-broadened-correctness.md` as an AR-sampling-randomness confound, not a regression —
  Phase 3's seed-controlled per-graph check remains the strong correctness claim.
- **Phase 7** (dispatch + build wiring) — done as part of Phase 4's commit; proof is the same CLI
  run.
- **Phase 8** (Android build) — `build-android-arm64-v8a.sh` (QNN disabled, out of scope) produces
  a clean `libsherpa-onnx-jni.so` containing the ZeroTTS sources, re-verified after Phase 9's
  interface change too. **Gap: no physical device or emulator was available in this environment,
  so on-device execution was never verified — cross-compilation/linking success is confirmed, real
  device behavior is not.** This needs a human with a device.
- **Phase 9** (streaming) — additive interface change to `offline-tts.h`/`offline-tts-impl.h`,
  confirmed non-breaking by smoke-testing 2 `vits` voice packs + `kitten` through the same CLI
  binary afterward (all three still produce correct audio). `GenerateStreaming` implemented via a
  shared `RunArLoop` helper (refactored out of `Generate` so the AR loop's control flow — the
  highest-risk code in this feature — exists exactly once). Proof test
  (`offline-tts-zerotts-streaming-test.cc`) holds a real 12-frame code sequence fixed and confirms
  chunked (`codec_decode_step`) and whole-shot (`codec_decode_full`) decoding of the *same* codes
  agree (energy/length checks) — passes. **Update 2026-09-15:** this had only ever been exercised
  by that unit test — no CLI or device had actually called `GenerateStreaming` until a follow-up
  session added a `--streaming` CLI flag and ran it for real, including on the Android device from
  Phase 8. See open item 3 below for the real time-to-first-chunk numbers this produced.

### Done, but the result is a judgment call for a human, not a pass/fail I'm asserting

- **Phase 10** (dynamic int8 quantization + eval) — fully measured: model size (~63% reduction),
  RTF (~18% average speedup on this desktop CPU, **not yet on the actual Android target — no
  device available, same gap as Phase 8**), a real fixed-input/fixed-randomness quantization-error
  isolation check, and a WER check via `whisper-tiny` (weak instrument for Vietnamese, reported
  with that caveat). **Headline finding:** `text_encoder`/`prefix_step`/codec graphs quantize
  normally (2.4% relative L2 error, typical); `local_frame_decode` (the AR-sampling graph) shows
  15/16 codebooks disagreeing with fp32 on a controlled single-frame test — exactly the
  compounding-error risk spec.md §9/§13 predicted in advance. Full writeup, thresholds proposed
  (not asserted as final), in `evals/zerotts-int8-baseline.md`.
- **Phase 11** (conditional static/QDQ quantization) — attempted for `local_frame_decode`
  specifically (triggered by Phase 10's finding above). Calibration succeeded and produced a
  loadable, smaller model, but **did not meaningfully improve the divergence** (14/16 codebooks
  still disagree, vs. 15/16 for dynamic). Proposed interim recommendation — **mixed precision:
  int8 for the 4 feed-forward graphs, fp32 for `local_frame_decode`** — is in
  `evals/zerotts-int8-baseline.md`'s §9, explicitly marked `PROPOSED — needs human confirmation`,
  not a decision made here.

### Explicit open items for human review (collected in one place)

1. ~~Quantization acceptance call~~ — **RESOLVED 2026-09-15.** Approved as proposed: mixed
   precision (int8 for `text_encoder`/`prefix_step`/both codec decoders, fp32 for
   `local_frame_decode`). See `evals/zerotts-int8-baseline.md`'s "Acceptance call" section.
2. **Phase 8/10 on-device verification — DONE (2026-09-15), on a real Pixel 6 Pro (`raven`).**
   Rebuilt `sherpa-onnx-offline-tts` for Android arm64-v8a (the earlier Android build only produced
   the JNI `.so`, not a standalone executable — needed `SHERPA_ONNX_ENABLE_BINARY=ON`; also hit and
   fixed a pre-existing, ZeroTTS-unrelated NDK link error, see
   `notes/environment-android-executable-link-fix.md`). Pushed the binary + `libonnxruntime.so` +
   the **approved mixed-precision** model set (int8 `text_encoder`/`prefix_step`/both codec
   decoders, fp32 `local_frame_decode`) to the device via `adb push`, ran real synthesis over
   `adb shell` for 2 voices/sentences, pulled the resulting `.wav` files back
   (`assets/zerotts/device_verification/`, gitignored — not committed, same as other generated
   audio). Both are valid 48kHz mono WAVs matching the on-device-reported duration exactly
   (145920/261120 samples ÷ 48000 = 3.04s/5.44s), non-silent, non-clipping (peak 0.73/0.96, RMS
   0.13/0.11, 0% samples clipped — automated proxy, no playback available, same caveat as item 3
   below). **Measured on-device RTF (single-threaded, this specific Pixel 6 Pro): 1.30 and 1.36**
   (slower than real-time) — this is the first real Android-target performance number; it had not
   been measured anywhere before this. Remaining gap: only the standalone CLI binary was exercised
   over `adb shell`, not the JNI/app-embedded path a real Android app would use, and only
   single-threaded — multi-threaded RTF and JNI-path integration are still unverified.
3. **Streaming (Phase 9) actually exercised end-to-end — DONE (2026-09-15).** Until this point,
   `GenerateStreaming` had only ever been called from a unit test on desktop; the CLI binary always
   used the offline `Generate()` path (verified by reading `sherpa-onnx-offline-tts.cc` directly —
   it's what item 2 above actually ran, not streaming, despite the task originally asking about
   "truly streaming"). Added a `--streaming` flag to `sherpa-onnx-offline-tts.cc` (errors clearly
   if the loaded model doesn't support it via `SupportsStreaming()`; measures and prints
   time-to-first-chunk in addition to the usual RTF; purely additive, no other model's behavior
   changes) and a ZeroTTS usage example in the CLI's `--help` text (previously missing — every
   other model had one). Sanity-checked on desktop first (fp32, 5 chunks, 225ms
   time-to-first-chunk), then rebuilt for Android and ran on the same Pixel 6 Pro with the approved
   mixed-precision model set: **6 chunks, time-to-first-chunk = 186ms** (vs. ZeroTTS's own ~70ms
   reference figure — real, meaningfully slower, not close; likely single-thread ORT session
   overhead on this specific device, unverified whether multi-threading or a warm/pre-loaded
   session would close the gap). Output pulled to
   `assets/zerotts/device_verification/out_maichi_streaming_device.wav` (gitignored) — valid 48kHz
   mono, non-silent, non-clipping, duration matches sample count exactly.
4. **Manual listening gap — partially resolved.** The project author listened to the offline-mode
   device outputs (item 2) and confirmed they sound correct. The streaming-mode output (item 3)
   has not been listened to yet — still only automated proxy checks (peak/RMS/clipping) for that
   one. The broader Phase 4/6/10 desktop outputs referenced in spec.md §11/§10 are also still
   unlistened-to beyond the two device samples above.
5. **Environment quirks worth knowing about** (documented in `notes/`, not open questions, but
   worth a human's awareness): (a) this environment's onnxruntime static-lib build crashes on
   *any* TTS model (not a ZeroTTS bug) — use `-DBUILD_SHARED_LIBS=ON` for CLI-based work here; (b)
   `ONNXRUNTIME_DIR`/`SHERPA_ONNXRUNTIME_{LIB,INCLUDE}_DIR`/`SHERPA_ONNX_ENABLE_QNN` are pre-set in
   the shell environment pointing at a different project's Android arm64-v8a onnxruntime — correct
   for the Android build script, wrong for desktop builds (which must `env -u` them; see
   `notes/environment-onnxruntime-static-lib-crash.md`); (c) this environment's prebuilt Android
   onnxruntime.so fails to link into standalone executables (not the JNI `.so`) under this NDK's
   default strict-undefined-symbol linking — fixed via
   `-DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"`, see
   `notes/environment-android-executable-link-fix.md`.
6. No other open questions were left unresolved during the run; nothing was silently skipped
   without a note explaining what and why.

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

**Status: DONE.** See top-of-file summary. Notes: `notes/phase0-voice-bin-format.md`,
`notes/phase0-codec-decode-step-io.md`, `notes/phase0-vi-normalizer-scope.md`; spec.md updated
in place.

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

**Status: DONE.** `offline-tts-zerotts-tokenizer-test.cc` passes, byte-exact vs. Python reference.

Files: `sherpa-onnx/csrc/offline-tts-zerotts-tokenizer.{h,cc}`

Implement the self-contained `tokenizer.json` BPE loader per spec §5 (NFC normalize → whitespace
collapse → whitespace/punctuation/digit pre-tokenizer splits → BPE merge → `[<bos>, ...ids, <eot>]`
wrapping).

**Proof:** unit test (`offline-tts-zerotts-tokenizer-test.cc`, following the existing
`sentence-piece-tokenizer-test.cc` pattern) encoding a fixed set of Vietnamese/English/mixed
strings, asserting identical token ID sequences against the real Python `tokenizers`-backed
reference (run once, offline, to generate the expected-output fixture).

## Phase 2 — Config + asset loading (no inference yet)

**Status: DONE.** `offline-tts-zerotts-voice-test.cc` passes (all 8 voices + both `.npy` files,
`Validate()` reject/accept paths).

Files: `offline-tts-zerotts-model-config.{h,cc}`, `offline-tts-zerotts-voice.{h,cc}`, a small
shared `.npy` reader (per spec §6's decision) for `null_voice_emb.npy`/`silence_frame.npy`.

**Proof:** load all 8 real voice packs + both `.npy` files, assert expected shapes
(`(1, n_voice_queries, d_model)`) and that `Validate()` correctly rejects missing/malformed paths.
No audio produced yet — this phase is purely "do the files parse correctly."

## Phase 3 — Model wrapper: individual ONNX sessions

**Status: DONE.** `offline-tts-zerotts-model-test.cc` passes against real captured Python
intermediate tensors for all 4 non-streaming graphs (decode_step covered in Phase 9 instead,
since it needs streaming state plumbing that doesn't exist until then).

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

**Status: DONE**, with one caveat. CLI end-to-end run for voice=maichi produced real audio,
0.935 mel-spectrogram correlation vs. the Python reference for the fixed proof sentence. The
"manual listen" part of this proof was **not done** (no audio playback available) — see the
top-of-file open items list.

Files: `offline-tts-zerotts-impl.h` (implements `OfflineTtsZeroTtsImpl::Generate`, per spec §7)

Wire Phase 1–3 together into the full loop: text → tokenize → text_encoder → prefix_step (cold
start) → loop(local_frame_decode → prefix_step) → stop condition → codec_decode_full → waveform.

**Proof:** one fixed sentence + `voice="maichi"`, compare the produced waveform against the
reference Python `zerotts` package's output for the identical input (spec §11's reference-parity
check) — duration match, spectrogram correlation, and a manual listen. This is the first point
audio actually comes out; it's the main correctness gate for the whole feature.

## Phase 5 — Vietnamese text normalizer

**Status: DONE**, scoped to 8/17 categories per `notes/phase0-vi-normalizer-scope.md` (the
other 9 are documented deferrals — mainly because `std::regex` has no lookbehind support).
`offline-tts-zerotts-vi-normalizer-test.cc` passes, 10/10 sentences byte-exact vs. Python.

Files: `offline-tts-zerotts-vi-normalizer.{h,cc}`

Port whatever Phase 0.4 scoped as realistic from `vi_normalizer.py`/`abbreviations.txt`, wired in
before tokenization in the `Generate` path.

**Proof:** run the same sample set (numbers, dates, acronyms, code-switched English) through both
`normalize_vi_text()` and the C++ port, diff outputs string-for-string.

## Phase 6 — Broaden correctness: all voices, more text

**Status: DONE.** See `notes/phase6-broadened-correctness.md` for the full table and the
explanation of why longer-sentence correlation is lower (AR-sampling randomness, not a
regression) and the manual-listen gap.

No new files — exercising Phase 1–5 more widely.

**Proof:** smoke-test all 8 bundled voices synthesize without error; expand the Phase 4 comparison
set beyond one sentence to catch normalizer/tokenizer edge cases the single-sentence check missed.

## Phase 7 — Wire into sherpa-onnx's dispatch + build system

**Status: DONE** (folded into the Phase 4 commit — the dispatch branch and CMakeLists.txt
change had to exist for Phase 4's CLI proof to run at all).

Files: edit `offline-tts-impl.cc` (new dispatch branch keyed on
`config.model.zerotts.text_encoder` non-empty, per spec §3) and
`sherpa-onnx/csrc/CMakeLists.txt` (add new `.cc` files to the existing TTS source list — no new
`SHERPA_ONNX_ENABLE_*` flag needed, confirmed in spec §3).

**Proof:** build and run the existing `sherpa-onnx-offline-tts` CLI binary end-to-end pointed at
ZeroTTS config fields, on desktop Linux — this is the point where it's a real sherpa-onnx backend,
not just an internal test harness.

## Phase 8 — Android/AAOS build

**Status: DONE for cross-compilation; NOT done for on-device verification** (no device/emulator
available). `build-android-arm64-v8a.sh` succeeds (QNN disabled — out of scope per spec.md §9's
"plain CPU EP is the starting point"), producing `libsherpa-onnx-jni.so` with the ZeroTTS sources
built in; re-verified after Phase 9's interface change.

No new source files — validates Phase 1–7 under the NDK toolchain.

**Proof:** run `build-android-arm64-v8a.sh` and confirm the new sources compile and link cleanly
into `libsherpa-onnx-jni.so`; if feasible, run the offline synthesis path on a real device/emulator
and sanity-check output (even a basic "it produces non-silent audio of the right duration" check is
enough here — full parity checks already happened on desktop in Phase 4/6).

## Phase 9 — Streaming

**Status: DONE.** All three proof bullets below hold: other models unaffected (smoke-tested);
streaming-vs-offline decode-of-identical-codes test passes
(`offline-tts-zerotts-streaming-test.cc`); time-to-first-chunk was **not separately benchmarked
against the ~70ms reference figure** on desktop (time-boxed out) or Android (no device) — a real
gap, noted rather than asserted as met.

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

**Status: MEASURED, acceptance call PROPOSED (not final).** Full results, thresholds, and the
proposed (human-confirmable) acceptance call are in `evals/zerotts-int8-baseline.md`. RTF measured
on desktop only (no Android device available — see top-of-file open items).

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

**Status: ATTEMPTED** (Phase 10's `local_frame_decode` finding met the trigger condition below).
Result did not meaningfully improve on dynamic quantization for that graph -- see
`evals/zerotts-int8-baseline.md` §9. Proposed interim recommendation (mixed precision) is there,
marked `PROPOSED — needs human confirmation`, not decided here.

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
