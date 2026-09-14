# CLAUDE.md

This repo is `hieunc278/sherpa-onnx`, a **private fork** of
[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (a large, existing C++/multi-binding
ONNX Runtime speech toolkit — ASR, TTS, VAD, speaker ID, etc.). Everything under version control
here except the docs below is upstream sherpa-onnx as of the fork point — treat it as a known
quantity, not something to re-explore from scratch each session.

**Our own work in this fork is scoped to one feature: adding ZeroTTS (Vietnamese zero-shot TTS)
support.** The full paper trail for that work, in order, is:

1. [intent/zerotts-sherpa-onnx-support.md](intent/zerotts-sherpa-onnx-support.md) — why, scope, constraints
2. [spec.md](spec.md) — architecture/design, derived from the intent
3. [plan.md](plan.md) — phased implementation plan with a concrete "proof" per phase

**Always check `plan.md` for current phase status before starting work** — it's the source of
truth for what's done vs. next. If implementation reveals the plan or spec was wrong about
something, update that doc in the same change, don't silently diverge from it.

## Process notes (solo project, no CI/CD, no cross-review yet)

This project follows an intent → spec → plan → build loop (an AI-native SDLC playbook, adapted for
solo work). There is no second reviewer — self-verification via each phase's "Proof" step in
`plan.md` is the only correctness gate, so treat those proofs as non-negotiable, not aspirational.
`REVIEW.md` doesn't exist yet; when it's added, run `/code-review` against it before considering
any diff done.

## Build

Standard out-of-source CMake build (desktop Linux, for Phases 0–7/9–11 of plan.md):

```bash
mkdir -p build && cd build
cmake -DSHERPA_ONNX_ENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

- `SHERPA_ONNX_ENABLE_TESTS` defaults `OFF` — must be explicitly `ON` to get gtest-based test
  binaries and `ctest` wiring.
- `SHERPA_ONNX_ENABLE_TTS` defaults `ON` and gates **all** TTS backends behind one flag — there is
  no per-model build flag. Adding ZeroTTS support means adding its `.cc` files to the existing TTS
  source list in `sherpa-onnx/csrc/CMakeLists.txt` (near the Kitten/Supertonic/Zipvoice entries,
  ~line 299–315), not introducing a new option.

Android/AAOS build (plan.md Phase 8 onward), via the repo's existing NDK scripts — requires
`ANDROID_NDK` set:

```bash
./build-android-arm64-v8a.sh
```

## Test

```bash
cd build && ctest --output-on-failure
```

New unit tests are added as `<name>-test.cc` files under `sherpa-onnx/csrc/`, appended to the
`sherpa_onnx_test_srcs` list in `sherpa-onnx/csrc/CMakeLists.txt` (the
`sherpa_onnx_add_test(source)` CMake function wires each into `add_test`/gtest automatically —
follow the existing `sentence-piece-tokenizer-test.cc` pattern for the ZeroTTS tokenizer test in
plan.md Phase 1).

## Conventions for adding a new TTS backend (what we're doing here)

sherpa-onnx's existing pattern for every TTS model (Kitten, Supertonic, Zipvoice, Pocket, etc.),
which our ZeroTTS work follows exactly — see spec.md §3 for the full file list:

- `offline-tts-<name>-model-config.{h,cc}` — config struct + `Validate()`
- `offline-tts-<name>-model.{h,cc}` — `Ort::Session` wrapper(s)
- `offline-tts-<name>-impl.h` — `OfflineTts<Name>Impl : public OfflineTtsImpl`, does the actual
  generation orchestration
- Dispatched via a new `if (!config.model.<name>.<required_field>.empty())` branch in
  `offline-tts-impl.cc`, alongside the existing per-model branches — sequential if-else, order
  matters only in that the first matching branch wins.

## Repo-specific working directories

- `intent/` — one file per feature/initiative (this repo will eventually have more than one)
- `notes/` — spike/investigation findings that don't belong in spec.md's design narrative but are
  needed as a citable record (see plan.md Phase 0)
- `evals/` — quantization and other before/after evaluation results (see plan.md Phase 10/11)
- `.claude/skills/`, `.claude/hooks/` — currently empty scaffolding for future org-policy-style
  skills/hooks; not populated yet, add as real policies emerge rather than speculatively

## External references used while designing this feature

- ZeroTTS model/weights: https://huggingface.co/zeroweight-ai/ZeroTTS
- ZeroTTS source (Python reference implementation, `src/zerotts/`):
  https://github.com/zeroweight-ai/ZeroTTS
- sherpa-onnx upstream: https://github.com/k2-fsa/sherpa-onnx
