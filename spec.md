# Spec: ZeroTTS support in sherpa-onnx

- **Status:** draft — pending author review
- **Derived from:** [intent/zerotts-sherpa-onnx-support.md](intent/zerotts-sherpa-onnx-support.md)
- **Date:** 2026-09-14

## 1. Recap of scope

Add ZeroTTS (Vietnamese zero-shot TTS, ONNX-only, 202M params) as a new TTS backend in sherpa-onnx's
C++ core, in this private fork (`hieunc278/sherpa-onnx`). In scope: the 8 bundled voice packs,
**both offline (batch) and true low-latency streaming synthesis**, C++ core only (no
language bindings yet). Out of scope: arbitrary reference-audio voice cloning (blocked — encoder
unpublished), non-Vietnamese/English languages.

## 2. High-level pipeline (confirmed from ZeroTTS's `synthesizer.py`/`codec.py`)

```
text ─▶ [Vietnamese normalizer] ─▶ [ZeroTtsTokenizer] ─▶ token_ids
                                                            │
                                                            ▼
                                                  text_encoder.onnx (once)
                                          ──▶ text_states, soa_embed, cross_kv
                                                            │
                                     voice_emb (from voice.bin) ─┐
                                                            ▼    ▼
                                          prefix_step.onnx (cold start, external_embed = concat(voice_emb, soa_embed))
                                                            │
                              ┌─────────────────────────────┘
                              │  loop until <eoa>+tail or max_frames=1500:
                              ▼
                  local_frame_decode.onnx (samples 1 frame, all codebooks, incl. temperature/top-k/top-p/rep-penalty)
                              │
                  prefix_step.onnx (advances cache by one step)
                              │
                              ▼
                    frame_codes accumulated (12.5 Hz)
                              │
              ┌───────────────┴────────────────┐
              ▼ (offline)                       ▼ (streaming)
  moss_audio_tokenizer_decode_full.onnx   moss_audio_tokenizer_decode_step.onnx
  (whole sequence → waveform, once)       (chunked, stateful KV-cache/ring-buffer decode,
                                            doubling chunk schedule: first_chunk_frames,
                                            then ×2 up to max_chunk_frames)
```

`moss_audio_tokenizer_decode_shared.data` is ONNX external-data (shared weights) referenced by
both codec graphs, not a separate model to run.

## 3. New files (following sherpa-onnx's existing per-model pattern)

All added under `sherpa-onnx/csrc/`, mirroring how Kitten/Supertonic/Zipvoice/Pocket are structured:

| File | Purpose |
|---|---|
| `offline-tts-zerotts-model-config.h/.cc` | Config struct + validation (see §4) |
| `offline-tts-zerotts-tokenizer.h/.cc` | New self-contained `tokenizer.json` BPE loader (see §5) |
| `offline-tts-zerotts-model.h/.cc` | Owns the 5 `Ort::Session`s, exposes `RunTextEncoder`, `RunPrefixStep`, `RunLocalFrameDecode`, `RunCodecDecodeFull`, `RunCodecDecodeStep` |
| `offline-tts-zerotts-voice.h/.cc` | Loads `voice.bin` + `meta.json` into a `(1, n_voice_queries, d_model)` float buffer |
| `offline-tts-zerotts-vi-normalizer.h/.cc` | Port of `vi_normalizer.py` (regex-based; ships its own `abbreviations.txt`-derived data table) |
| `offline-tts-zerotts-impl.h` | `OfflineTtsZeroTtsImpl : public OfflineTtsImpl` — orchestrates the pipeline in §2 for both offline and streaming entry points |

Wired into the existing dispatch in `offline-tts-impl.cc`, alongside the other
`if (!config.model.xxx.yyy.empty()) return OfflineTtsXxxImpl::Create(...)` branches — new branch
keyed on `config.model.zerotts.text_encoder` (or similar) being non-empty.

Build wiring: **no new CMake option needed.** There's a single blanket `SHERPA_ONNX_ENABLE_TTS`
flag gating all TTS backends (`CMakeLists.txt:71`); we just add the new `.cc` files to the existing
TTS source list in `sherpa-onnx/csrc/CMakeLists.txt` (next to the Kitten/Supertonic/Zipvoice
entries around line 299-315).

## 4. Config schema — `OfflineTtsZeroTtsModelConfig`

Following the shape of `OfflineTtsKittenModelConfig`/`OfflineTtsSupertonicModelConfig`:

```cpp
struct OfflineTtsZeroTtsModelConfig {
  std::string text_encoder;       // text_encoder.onnx
  std::string prefix_step;        // prefix_step.onnx
  std::string local_frame_decode; // local_frame_decode.onnx
  std::string codec_decode_full;  // onnx/codec/moss_audio_tokenizer_decode_full.onnx
  std::string codec_decode_step;  // onnx/codec/moss_audio_tokenizer_decode_step.onnx
  // codec_decode_shared (.data) is resolved implicitly by ORT next to codec_decode_full/step
  std::string tokenizer;          // tokenizer.json
  std::string null_voice_emb;     // null_voice_emb.npy, read via a minimal .npy parser (see §12.4)
  std::string silence_frame;      // silence_frame.npy, read via the same minimal .npy parser
  std::string voice;              // path to a voices/<name>/ dir (voice.bin + meta.json)
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  // Generation defaults -- confirmed against synthesizer.py's ZeroTTS.synthesize()/
  // synthesize_stream() default keyword arguments (Phase 0.3 spike, 2026-09-14).
  int32_t min_frames = 4;              // synthesizer.py: synthesize(min_frames=4)
  int32_t max_frames = 1500;           // synthesizer.py: DEFAULT_MAX_FRAMES = 1500
  int32_t eoa_extra_frames = 1;        // synthesizer.py: synthesize(eoa_extra_frames=1)
  float text_temperature = 1.0f;       // synthesizer.py: synthesize(text_temperature=1.0)
  int32_t text_topk = 50;              // synthesizer.py: synthesize(text_topk=50)
  float audio_temperature = 0.8f;      // synthesizer.py: synthesize(audio_temperature=0.8)
  int32_t audio_topk = 25;             // synthesizer.py: synthesize(audio_topk=25)
  float audio_topp = 0.95f;            // synthesizer.py: synthesize(audio_topp=0.95)
  float audio_repetition_penalty = 1.2f;  // synthesizer.py: synthesize(audio_repetition_penalty=1.2)
  float cfg_scale = 1.0f;              // synthesizer.py: synthesize(cfg_scale=1.0) -- >1.0 doubles
                                        // per-frame cost (adds an unconditional CFG branch); not
                                        // used by default.

  // Streaming-specific -- confirmed against synthesizer.py's synthesize_stream() defaults.
  int32_t first_chunk_frames = 1;      // synthesizer.py: synthesize_stream(first_chunk_frames=1)
  int32_t max_chunk_frames = 16;       // synthesizer.py: synthesize_stream(max_chunk_frames=16)
};
```

Sample rate is fixed at 48kHz (from the model card) and doesn't need to be user-configurable.

## 5. Tokenizer — why a new loader is required

Checked sherpa-onnx's existing tokenizer code (`QwenAsrTokenizer`, `FunASRNanoTokenizer`,
`SentencePieceTokenizer`): none of them load a **self-contained** `tokenizer.json` the way
ZeroTTS's does — they all expect `vocab.json` + `merges.txt` as separate files, using
`tokenizer.json` only for `added_tokens`. ZeroTTS ships only `tokenizer.json`, with `model.vocab`
and `model.merges` embedded directly inside it.

`OfflineTtsZeroTtsTokenizer` needs to replicate, from the parsed JSON:
- **Normalizer:** NFC normalization, then collapse runs of whitespace to a single space
- **Pre-tokenizer:** split on whitespace (isolated), then punctuation (isolated), then split
  digits into individual characters
- **Model:** BPE merge per pre-token, using the embedded `vocab`/`merges` maps (no byte-fallback,
  no continuing-subword-prefix)
- **Wrapping:** `[<bos>=1, ...body_ids, <eot>=2]` — confirmed from `tokenizer.py`; language tags
  (`<en>`/`<vi>`) and slot tokens (`<soa>`/`<slot>`/`<eoa>`) exist in the vocab but are **not**
  auto-inserted around text — they're used elsewhere in the pipeline (`soa_embed` from the text
  encoder, `<eoa>` as the loop's stop signal), not by the tokenizer itself.

This is the single largest net-new piece of code in this feature (small, but not a copy of an
existing sherpa-onnx pattern) — flag it for extra care/tests in `plan.md`.

## 6. Voice loading — decision: use `voice.bin`, not `voice.npz`

The reference Python implementation reads `voice.npz` (a zip-based NumPy archive). Parsing `.npz`
in C++ means depending on a zip library plus NumPy's header format. ZeroTTS's own repo already
ships `voice.bin` per voice pack, documented in its source as "raw f32 of voice_emb, for the JS
demo" — unused by Python, but exactly the format we want: a flat float32 buffer we can read
directly. **Decision: use `voice.bin` + `meta.json` (for `n_voice_queries`/display metadata), skip
`.npz` entirely.**

**Confirmed (Phase 0.1 spike, see `notes/phase0-voice-bin-format.md`):** `voice.bin` is raw
float32, little-endian, row-major, **no header at all** — verified by exact size match
(`n_voice_queries * d_model * 4 == file_size_bytes`, 30720 == 30720 for `voices/maichi/`) and by
bit-exact numerical agreement against the reference `.npz`'s `voice_emb` array over its full
contents, not just a size coincidence.

`null_voice_emb.npy` and `silence_frame.npy` are plain NumPy `.npy` files (not `.npz`) — still need
either a minimal `.npy` header parser (simpler than full npz: fixed magic + shape/dtype header +
raw data, no zip) or a one-time offline conversion to raw `.bin` similar to the voice packs. Lean
towards a tiny shared `.npy` reader since it's a well-defined, small format, rather than hand
re-exporting these two files — decide in `plan.md`.

## 7. Offline (batch) generation flow

`OfflineTtsZeroTtsImpl::Generate(text, config)`:
1. Normalize text (Vietnamese normalizer) → tokenize → `token_ids`
2. Run `text_encoder` once
3. Run `prefix_step` (cold start) with `external_embed = concat(voice_emb, soa_embed)`
4. Loop: `local_frame_decode` → `prefix_step`, appending sampled frame codes, until stop condition
5. Reshape collected codes to `(1, K, T_gen)`, run `codec_decode_full` once
6. Return `GeneratedAudio{samples, sample_rate=48000}`

This fits the existing `OfflineTtsImpl::Generate` virtual interface unmodified.

## 8. Streaming generation flow + new public API surface

This is the one place where the change isn't purely additive-new-files — it requires a small,
**additive** extension to the shared `OfflineTts`/`OfflineTtsImpl` interface, because the existing
`GeneratedAudioCallback` fires only per-sentence-batch (confirmed by reading `offline-tts.h`), not
at the intra-utterance granularity ZeroTTS needs (~70ms first chunk).

Proposed shape:
```cpp
// offline-tts.h — new, default-unimplemented virtual; existing models unaffected.
using StreamingAudioCallback = std::function<int32_t(
    const float * /*samples*/, int32_t /*n*/, int32_t /*sample_rate*/)>;

virtual bool SupportsStreaming() const { return false; }
virtual void GenerateStreaming(const std::string &text,
                                const GenerationConfig &config,
                                StreamingAudioCallback callback) const {
  SHERPA_ONNX_LOGE("This model does not support streaming synthesis");
}
```
`OfflineTtsZeroTtsImpl` overrides both; every other existing model keeps its default (no-op),
so **no other model's behavior changes** — satisfies intent.md's "without modifying unrelated
parts of sherpa-onnx" as an additive, non-breaking change to the shared interface rather than
literally zero shared-file diffs (which isn't achievable for a genuinely new capability).

`GenerateStreaming` runs the same loop as §7, but instead of accumulating all frames:
- Buffers frames per the doubling schedule (`first_chunk_frames`, ×2 up to `max_chunk_frames`)
- Calls `codec_decode_step` per buffered chunk, carrying forward the codec's ring-buffer/KV-cache
  state between calls (mirrors `codec.py`'s `streaming_decoder()`/`decode_chunk()`)
- Invokes `callback` with each decoded chunk; stops early if callback returns 0

## 9. INT8 quantization for edge deployment

Goal: produce an int8-quantized variant of the ONNX graphs so the model can run on CPU-constrained
edge devices, without adding a separate code path — the C++ backend should just load whichever set
of graph files (fp32 or int8) the config points at.

- **No config/schema change needed.** `OfflineTtsZeroTtsModelConfig` (§4) already stores plain
  file paths for each graph. Producing `text_encoder.int8.onnx`, `prefix_step.int8.onnx`, etc. and
  pointing config at those is enough — this matches the naming convention several existing
  sherpa-onnx model releases already use (`model.onnx` vs. `model.int8.onnx`).
- **Quantization is an offline, one-time step** (Python, using `onnxruntime.quantization`), not
  something the C++ runtime does — it happens before the graphs are shipped, same as how the
  fp32 graphs are exported once upstream.
- **Two approaches to evaluate, not assume:**
  - *Dynamic (weight-only) quantization* (`quantize_dynamic`) — simplest, no calibration data
    needed, weights quantized to int8, activations computed in fp32. Good first baseline.
  - *Static/QDQ quantization* (`quantize_static`) — needs a representative calibration set
    (Vietnamese text spanning normal/edge cases, run through the fp32 pipeline to collect
    activation ranges) but usually preserves quality better. Attempt only if dynamic quantization's
    quality loss (per §10 below) isn't acceptable.
- **Not all five graphs carry equal risk.** `text_encoder` and the codec decoders
  (`codec_decode_full`/`codec_decode_step`) are feed-forward and quantize the way most ONNX models
  do. `local_frame_decode` is different: it performs the actual autoregressive *sampling*
  (temperature/top-k/top-p/repetition penalty) inside the graph, and its output feeds back into
  `prefix_step` for up to 1500 steps — small per-step quantization error can compound over the
  loop in a way a single feed-forward graph's error can't. Quantize and evaluate this graph
  **separately** from the others, not as part of one blanket "quantize everything" pass.
- **Edge target confirmed: Android/AAOS via NDK (arm64-v8a primary).** sherpa-onnx already has
  Android NDK build scripts, so no new build tooling is needed — just verifying the new sources
  build under that existing path (see `plan.md`). Plain CPU int8 (ORT's default CPU EP) is the
  starting point; NNAPI or other Android execution providers are a possible later optimization,
  not required for a first working int8 build.

## 10. Evaluation methodology: fp32 vs. int8

This directly extends §9's verification approach — same fixed test set (sentences × the 8 bundled
voices), run through both the fp32 and int8 pipelines, with explicit before/after numbers rather
than a subjective "sounds fine":

| Dimension | Metric | How |
|---|---|---|
| Perceptual/objective quality | Waveform/mel similarity (e.g. mel-cepstral distortion or spectrogram correlation) between fp32 and int8 output for identical inputs | Automated, run per sentence/voice pair |
| Intelligibility | WER from feeding synthesized audio through an existing ASR model (sherpa-onnx has ASR backends already available for this), compared against the input text | Automated; compare fp32-WER vs. int8-WER, not just int8 in isolation |
| Latency | RTF (real-time factor) for offline synthesis; time-to-first-chunk for streaming | Measured on the actual target edge device once §9's open question on device profile is resolved — a laptop/dev-machine number isn't representative |
| Model size | Total on-disk size of the graph set | Trivial, but report it — it's half the point of quantizing |
| Manual spot-check | Listen to a handful of int8 outputs across all 8 voices | Cheap sanity check; catches artifacts objective metrics might miss (e.g. buzzing, mispronunciations from AR error accumulation) |

**Acceptance bar** (to decide concretely before running the comparison, not after, so the exercise
has a pass/fail rather than "eyeballing it"): define acceptable deltas for WER and quality-metric
degradation up front once a first fp32 baseline number exists — deferred to `plan.md` since it
needs a real baseline measurement first.

**AR-loop-specific check:** because `local_frame_decode`'s sampling runs in a loop, also compare
quantized vs. fp32 output **at increasing generation lengths** (e.g. short vs. long sentences) to
see whether quality degradation grows with more autoregressive steps — a flat short-sentence-only
comparison could hide compounding error that only shows up on longer utterances.

## 11. Verification approach (solo, no CI/CD yet)

- **Reference parity:** for a fixed Vietnamese sentence + a fixed bundled voice (e.g. `maichi`),
  run both the reference Python `zerotts` package and this C++ implementation, compare waveforms
  (not bit-exact — check duration, spectrogram similarity/correlation, and a manual listen).
  Do this for offline mode first, then streaming (comparing reassembled streamed chunks against
  the offline output from the same inputs).
- **Tokenizer unit test:** encode a handful of Vietnamese/English/mixed strings with both the C++
  loader and Python's `tokenizers`-backed reference, assert identical token ID sequences.
- **All 8 bundled voices:** smoke-test each loads and synthesizes without error.
- **Text normalization:** compare `normalize_vi_text()` output against the ported C++ normalizer
  on a sample set covering numbers, dates, acronyms, code-switched English.
- This all runs as manual `make test`-style targets for now, consistent with the intent's
  "no CI/CD yet" constraint — but structured so wiring them into CI later is mechanical.

## 12. Design decisions deferred to `plan.md` (resolved during Phase 0, 2026-09-14 — kept here as a record)

1. ~~`decode_step` exact tensor names/shapes~~ — **resolved:** fully data-driven from
   `codec_browser_onnx_meta.json`'s `streaming_decode` section (108 tensors: 4 data + 8
   transformer-offset + 96 attention-cache, all confirmed against the live ONNX graph and
   cross-referenced against `codec.py`'s dynamic name resolution). See
   `notes/phase0-codec-decode-step-io.md`. The C++ model wrapper parses this JSON at load time
   rather than hardcoding tensor names/shapes — same approach the Python reference itself uses.
2. ~~Exact default sampling hyperparameters~~ — **resolved:** §4's config struct now carries the
   real defaults read off `synthesizer.py`'s `synthesize()`/`synthesize_stream()` signatures, each
   annotated with its source line. One correction along the way: the original placeholder schema
   conflated a single `temperature`/`top_k`/`top_p`/`repetition_penalty` — the real model has
   **two independent pairs** (`text_temperature`/`text_topk` for the control channel,
   `audio_temperature`/`audio_topk`/`audio_topp`/`audio_repetition_penalty` for the codebooks); §4
   has been corrected to match.
3. ~~`voice.bin` byte layout~~ — **resolved:** raw float32, little-endian, row-major, no header —
   confirmed by exact size match and bit-exact agreement against the reference `.npz`. See
   `notes/phase0-voice-bin-format.md`.
4. ~~`.npy` reader scope~~ — **resolved: minimal parser, not offline conversion.** Both
   `null_voice_emb.npy` (`<f4`, shape `(1, 10, 768)`) and `silence_frame.npy` (`<i8`, shape
   `(1, 16)`) use the standard NumPy v1.0 `.npy` header (`\x93NUMPY` magic + version + a
   Python-dict-literal header line padded to a 64-byte boundary, then raw data) — confirmed by
   inspecting the real files' header bytes. A ~30-line reader supporting exactly these two dtypes
   (no full NumPy format generality needed) is simpler and more maintainable than a one-time
   conversion step that would need re-running if the upstream files ever change.
5. ~~Vietnamese normalizer port fidelity~~ — **resolved:** scoped to 8 of 17 categories found in
   `vi_normalizer.py` for v1 (numbers, percent, time, date, month-year, abbreviations, fractions,
   `@`-sign, plus URL/email protection as a prerequisite guard), deferring 9 lower-value/niche
   categories, each with a documented reason. Key finding: `vi_normalizer.py`'s scanner regex
   depends heavily on **negative lookbehind assertions, which C++ `std::regex` does not support at
   all** — v1 ports the guard logic as manual character checks around lookbehind-free regexes
   rather than a direct regex-for-regex port. See `notes/phase0-vi-normalizer-scope.md` for the
   full per-category table.
6. ~~Target edge device profile~~ — **resolved: Android/AAOS via NDK, arm64-v8a primary** (see §9).
7. **Quantization acceptance thresholds** (WER delta, quality-metric delta) — still deferred; needs
   a real fp32 baseline measurement first before numbers can be set, per §10. Not resolvable during
   Phase 0 since it requires the full pipeline (Phases 1–9) to exist first.

## 13. Risks

- Streaming latency parity with the ~70ms reference figure depends on ONNX Runtime session/thread
  overhead in C++ being comparable to the Python path — not guaranteed, needs empirical check
  early rather than assumed.
- The AR loop + codec state management is the most complex part of this codebase to get exactly
  right without the model's original authors' tests to check against — self-verification (§11) is
  the only safety net given no cross-review, so it needs to be genuinely rigorous, not perfunctory.
- **Quantization error compounding in the AR loop.** Because `local_frame_decode` runs inside a
  loop of up to 1500 steps, int8 quantization error there isn't a one-shot approximation error —
  it can accumulate and drift in ways that only show up on longer utterances (see §10's
  AR-loop-specific check). Treat this graph's quantization as higher-risk than the feed-forward
  ones, not equal risk.
