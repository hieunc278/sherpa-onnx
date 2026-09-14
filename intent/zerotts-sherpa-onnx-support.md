# Intent: Add ZeroTTS support to sherpa-onnx

- **Status:** draft — pending author review
- **Author:** hieunc4@fpt.com
- **Date:** 2026-09-14

## Problem statement

I need fast, on-CPU Vietnamese text-to-speech for a project I'm building. [ZeroTTS](https://huggingface.co/zeroweight-ai/ZeroTTS)
is a 202M-parameter Vietnamese (+ code-switched English) TTS model that already runs entirely on
ONNX Runtime (no PyTorch, no GPU) and ships with 8 prebuilt voice packs, but it only has a
standalone Python inference package (`zerotts` on PyPI/HF). [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
is the C++/multi-binding ONNX Runtime speech toolkit I want to deliver TTS through, and it has no
ZeroTTS support today. There's no existing C++ implementation of ZeroTTS's inference pipeline to
build on.

## Proposed outcome

Implement ZeroTTS as a new TTS model backend inside sherpa-onnx's C++ core, following the existing
pattern used by other TTS models in the codebase (e.g. Kitten, Supertonic, Zipvoice):

- `offline-tts-zerotts-model-config.{h,cc}` — config struct (paths to the ONNX graphs, tokenizer,
  voice pack, sample rate, etc.)
- `offline-tts-zerotts-model.{h,cc}` — ONNX Runtime session wrapper(s) around the model's graphs
  (`text_encoder.onnx`, `prefix_step.onnx`, `local_frame_decode.onnx`, and the
  `moss_audio_tokenizer_decode_*.onnx` codec decoder)
- `OfflineTtsZeroTtsImpl`, wired into `offline-tts-impl.cc`'s dispatch logic alongside the other
  model types
- Vietnamese text normalization/frontend equivalent to the Python package's `normalize_vi_text()`
- **Both batch (offline) and streaming synthesis**, mirroring the Python package's `synthesize()`
  and `synthesize_stream()`. Whether this rides on sherpa-onnx's existing progressive-audio
  callback mechanism in `OfflineTts`, or needs new streaming-specific infra, is a design question
  for `spec.md` — see open questions.
- A minimal C++-level example/test that synthesizes Vietnamese text with a bundled voice and
  produces audio comparable to the reference Python implementation, in both modes

This is being built as a **private fork** of sherpa-onnx, with remote origin at
`https://github.com/hieunc278/sherpa-onnx`. It is not intended as an upstream contribution to
`k2-fsa/sherpa-onnx` — I set my own bar for style/tests/docs rather than upstream's.

## Non-goals (this iteration)

- **Arbitrary reference-audio voice cloning.** ZeroTTS's voice encoder (reference audio → speaker
  latent) is unpublished; only the 8 prebuilt `voices/*.npz` packs are usable client-side. Cloning
  from a user-supplied audio clip is out of scope until/unless that encoder becomes available.
- **Non-C++ bindings.** Python/Java/Swift/etc. bindings are not part of this pass — the goal is a
  correct, working C++ core implementation first (both offline and streaming). Bindings can follow
  once the core is validated.
- **Languages other than Vietnamese (+ code-switched English)** — matches the base model's scope,
  not an expansion of it.

## Affected users / systems

- **Me**, as the sole developer and consumer of this integration, via sherpa-onnx's C++ API.
- The project(s) I intend to embed this TTS engine into (not yet named/spec'd).
- No other users/systems are affected — this is a private fork, not submitted upstream, so it
  doesn't touch other sherpa-onnx consumers.

## Constraints

- **License:** ZeroTTS code/weights are MIT; its bundled MOSS-Audio-Tokenizer-Nano codec is
  Apache-2.0. sherpa-onnx is Apache-2.0. All compatible — no licensing blocker.
- **No PyTorch at inference time.** The reference Python package already runs on
  `numpy`/`onnxruntime`/`tokenizers` only — the C++ port should have the same property (ONNX
  Runtime only, matching how sherpa-onnx runs everything else).
- **Multi-graph pipeline, not a single model file.** Unlike simple VITS-style models, ZeroTTS is
  four ONNX graphs chained together (text encoder → autoregressive prefix/frame-decode loop →
  codec decoder). The C++ model wrapper needs to manage multiple `Ort::Session`s and the
  autoregressive loop state, similar to how sherpa-onnx already handles other multi-graph TTS
  models (Zipvoice, Supertonic). The exact loop control flow (stopping condition, max length, how
  speaker latents are injected into `prefix_step.onnx`/`local_frame_decode.onnx`) isn't publicly
  documented and will be **reverse-engineered from the Python `zerotts` package source** during the
  design/spec phase.
- **Tokenizer compatibility.** ZeroTTS ships a `tokenizer.json` (HF `tokenizers`-format). This needs
  to be made compatible with however sherpa-onnx already loads/represents tokenizers in its C++
  frontend code, rather than introducing a one-off parser — the exact approach (reuse an existing
  sherpa-onnx tokenizer loader vs. adapt one) is for `spec.md`.
- **Voice packs provided at runtime.** `voices/<name>/{voice.npz,voice.bin,meta.json}` are supplied
  as user-provided file paths at runtime (matching sherpa-onnx's convention for other model
  assets), not embedded at build time or downloaded automatically.
- **Repo setup:** private fork of `k2-fsa/sherpa-onnx`, remote origin
  `https://github.com/hieunc278/sherpa-onnx`.
- **Solo project, no CI/CD yet, no cross-review.** Verification of correctness (matching the
  reference Python implementation's output) is on me; see `plan.md`/testing approach in later
  stages for how self-verification will work without a second reviewer.

## Success criteria

- Given the same input text and the same bundled voice, the C++ implementation produces audio
  that is perceptually/qualitatively equivalent to the reference Python `zerotts` package's output
  (exact bit-for-bit match not required, given floating-point/ONNX Runtime version differences),
  **in both offline and streaming modes**.
- Streaming mode delivers audio incrementally with latency in the same ballpark as the reference
  (~70ms to first chunk), not just "streaming API that internally waits for the full utterance."
- All 8 bundled voices load and synthesize without errors.
- Vietnamese text normalization (numbers, dates, acronyms, code-switched English) behaves
  equivalently to `normalize_vi_text()`.
- Builds and runs as a standard sherpa-onnx C++ TTS backend (i.e., selectable via the same
  config-field-presence dispatch pattern as other models), without modifying unrelated parts of
  sherpa-onnx.

## Open questions

1. **Streaming mechanism.** Does sherpa-onnx's existing `OfflineTts` progressive-audio callback
   give real incremental delivery, or does true low-latency streaming (matching ZeroTTS's
   `synthesize_stream()`) require new streaming-specific plumbing (closer to how ASR's "online"
   models work)? Needs investigation during `spec.md` before the plan can commit to an approach.
2. **Autoregressive loop details.** The exact control flow of `prefix_step.onnx` +
   `local_frame_decode.onnx` isn't documented publicly — confirmed direction is to
   reverse-engineer it from the Python `zerotts` package source during the design/spec phase.
3. **Tokenizer integration point.** Which existing sherpa-onnx tokenizer/frontend code (if any) is
   the right place to hook ZeroTTS's `tokenizer.json` into, vs. writing a new loader — to be
   resolved in `spec.md` after looking at how other HF-`tokenizers`-based models (if any) are
   handled in sherpa-onnx today.
