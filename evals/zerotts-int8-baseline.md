# ZeroTTS INT8 (dynamic quantization) baseline evaluation

- **Status: PROPOSED — needs human confirmation.** This document reports measurements and a
  proposed acceptance call. It is explicitly **not** a final pass/fail verdict — per the task's
  instructions, quantization acceptance is a human judgment call, not something to decide
  unilaterally. See "Proposed acceptance call" at the end.
- Date: 2026-09-14
- Quantization method: `onnxruntime.quantization.quantize_dynamic` (weight-only int8), per
  spec.md §9/plan.md Phase 10. No calibration data used (dynamic quantization needs none).
- Script: `scripts/zerotts/quantize_int8.py`, `scripts/zerotts/run_eval_generation.sh`,
  `scripts/zerotts/score_eval.py`.

## 1. Model size

| Graph | fp32 | int8 | Reduction |
|---|---|---|---|
| text_encoder.onnx | 309 MB | 78 MB | 74.8% |
| prefix_step.onnx | 333 MB | 84 MB | 74.8% |
| local_frame_decode.onnx | 179 MB | 130 MB | 27.4% |
| moss_audio_tokenizer_decode_full.onnx (+ shared .data) | 0.65 MB + 43 MB shared | 14 MB (self-contained) | — |
| moss_audio_tokenizer_decode_step.onnx (+ shared .data) | 0.34 MB + 43 MB shared (same file) | 13 MB (self-contained) | — |
| **Total on-disk (5 graphs, shared data counted once for fp32)** | **≈865 MB** | **≈319 MB** | **~63%** |

Notes:
- `local_frame_decode` shrinks far less than the other two large graphs (27% vs. 75%) because
  `quantize_dynamic`'s default op selection (Conv+MatMul+Gemm) only quantizes some of its weight
  tensors; see §4 below for why this graph needed different handling anyway.
- **A real portability issue found along the way:** `quantize_dynamic`'s default settings produced
  `moss_audio_tokenizer_decode_full.int8.onnx` / `..._decode_step.int8.onnx` that **fail to load**
  in this onnxruntime build (1.28.2): `NOT_IMPLEMENTED: Could not find an implementation for
  ConvInteger(10)`. The codec decoders contain Conv layers, and this CPU EP build has no
  `ConvInteger` kernel. Workaround (applied in `quantize_int8.py`): restrict
  `op_types_to_quantize=["MatMul", "Gemm"]` for these two graphs only, leaving their Conv weights
  fp32. All 5 resulting `.int8.onnx` files load and run correctly (verified both via a standalone
  `onnxruntime.InferenceSession` check and end-to-end through the C++ CLI, see §2).

## 2. Loads/runs through the same C++ path

Confirmed: pointing `OfflineTtsZeroTtsModelConfig` at the `*.int8.onnx` paths (no code change,
per spec.md §9) loads and runs successfully through `sherpa-onnx-offline-tts`, producing
non-silent, non-clipping audio for a fixed sentence + voice=maichi. Same CLI binary, same
dispatch path as the fp32 case (Phase 4/7/9's proof).

## 3. Latency (RTF)

Fixed test set: 3 sentences (short/medium/long) × 2 voices (maichi, giahuy) = 6 combinations,
measured on this desktop dev machine (not yet the Android target — see §6's open item).
RTF = the CLI's own reported `Elapsed seconds / Audio duration` (post-model-load inference time).

| Case | fp32 RTF | int8 RTF | Speedup |
|---|---|---|---|
| short / maichi | 2.40 | 1.90 | 20.5% |
| short / giahuy | 2.43 | 2.04 | 15.8% |
| medium / maichi | 2.73 | 1.92 | 29.8% |
| medium / giahuy | 2.41 | 2.06 | 14.6% |
| long / maichi | 2.52 | 2.11 | 16.3% |
| long / giahuy | 2.55 | 2.33 | 8.8% |
| **Average** | **2.51** | **2.06** | **17.8%** |

Dynamic int8 quantization gives a consistent, real (~18% average) latency improvement on this
desktop CPU. **Not yet measured on the actual Android/AAOS arm64-v8a target** — spec.md §10
explicitly says "a laptop/dev-machine number isn't representative"; this is a recorded open item,
not a substitute for on-device numbers (see plan.md's status header — no device was available in
this environment for Phase 8 either, so this gap is linked to that same constraint).

## 4. Quality: two very different pictures depending on the graph

This is the most important finding of this evaluation, and it's exactly the risk spec.md §9/§13
called out in advance: **`local_frame_decode` (the AR-loop sampling graph) is dramatically more
sensitive to int8 quantization than the feed-forward graphs.** Measured directly, isolated from
AR-loop-length/sampling-randomness confounds, using the real intermediate tensors captured during
Phase 3 (fixed `global_hidden`, fixed `seen_mask`, fixed random draws -- identical inputs to both
the fp32 and int8 graphs):

| Graph | Check | Result |
|---|---|---|
| `text_encoder` | Relative L2 error of `soa_embed` (768-dim), fp32 vs int8, identical `text_ids` input | **2.4%** relative L2, max abs diff 0.021 -- small, typical feed-forward quantization drift |
| `local_frame_decode` | Number of the 16 codebooks whose **sampled code** differs, fp32 vs int8, with **identical** `global_hidden`/`seen_mask`/random draws | **15 / 16 codebooks differ** |

The `local_frame_decode` result deserves an explicit warning against over-reading it as "94%
broken": because it directly samples from a softmax, even a *small* quantization-induced shift in
logits can flip which discrete outcome a fixed uniform random draw lands on when the distribution
has many closely-ranked candidates (exactly what `audio_temperature=0.8`/`audio_topk=25` produce
by design, for natural-sounding variety). So this isn't necessarily "int8 sounds 94% worse" --
it's "int8 measurably shifts the sampling distribution enough that, at this operating point,
almost every draw picks a different bucket than fp32 would have." That distinction matters, but it
also means the two numbers above are not on the same footing, and the practical consequence is
the same either way: **quantizing `local_frame_decode` changes what gets generated, per spec.md
§9's explicit warning that this graph's small per-step error can compound over up to 1500
autoregressive steps** in a way the other four graphs' errors can't.

## 5. End-to-end audio comparison (full pipeline, both graphs int8'd, no fixed seed)

Same 6-combination test set as §3. **Neither the fp32 nor the int8 run pins the AR loop's RNG
seed** (matches production behavior of both -- see notes/phase6-broadened-correctness.md for the
same caveat already recorded there), so — as already observed in Phase 6 — mel-spectrogram
correlation for longer utterances is dominated by independent-sampling divergence as much as by
quantization, and should not be read as a clean quantization-quality signal on its own (§4's
fixed-input checks are the trustworthy signal for that). Reported here for completeness and for
the length-sensitivity pattern:

| Case | Spectrogram correlation (fp32 vs int8) |
|---|---|
| short / maichi | 0.906 |
| short / giahuy | 0.894 |
| medium / maichi | 0.565 |
| medium / giahuy | 0.369 |
| long / maichi | 0.493 |
| long / giahuy | 0.470 |
| **Average** | **0.616** |

The pattern (high correlation for short utterances, dropping for longer ones) is consistent with
§4's finding: more AR steps means more chances for a sampling-flip to occur and then compound.

## 6. Intelligibility (WER via whisper-tiny, `--whisper-language=vi`)

**Caveat, stated up front:** `whisper-tiny` is a small multilingual model with weak Vietnamese
support (confirmed by high WER on **both** fp32 and int8 -- it's not a strong-enough instrument to
finely separate the two). Reported for completeness per spec.md §10's explicit ask, not as a
strong signal on its own.

| Case | fp32 WER | int8 WER | Reference | fp32 ASR | int8 ASR |
|---|---|---|---|---|---|
| short/maichi | 0.00 | 0.00 | "Xin chào." | "Xin chào" | "Xin chào" |
| short/giahuy | 0.00 | 0.00 | "Xin chào." | "Xin chào" | "Xin chào" |
| medium/maichi | 0.50 | 0.50 | "Xin chào Việt Nam." | "Xin chào viên" | "Xin chào vinh n" |
| medium/giahuy | 0.00 | 0.50 | "Xin chào Việt Nam." | "Xin chào Việt Nam" | "xin chào vĩ" |
| long/maichi | 0.62 | 0.50 | "Cảm ơn bạn đã sử dụng ZeroTTS, một mô hình chạy trên CPU, không cần GPU." | "Cảm ơn bạn đã sử dụng \\" (truncated) | "Cảm ơn bạn đã sử dụng zero tts, một môi hình trạy trên cp ư không" |
| long/giahuy | 0.56 | 0.62 | (same as above) | "Cám ơn bạn đã sĩ dng Zero TTS, một mua hình trải trên CPU không cần giảm ưu." | "Cảm ơn bạn đã sử dụng \\" (truncated) |
| **Average** | **0.28** | **0.35** | | | |

With only 6 samples and a weak ASR instrument, this +0.07 average WER delta is **not statistically
meaningful** on its own -- it neither confirms nor rules out a real intelligibility regression. It
does not contradict §4's finding (a real per-step sampling shift); it just isn't precise enough to
independently corroborate the magnitude.

## 7. Manual spot-check across all 8 voices

**Not done as "listening."** No audio playback is available in this session (same constraint noted
in notes/phase6-broadened-correctness.md). What *was* checked: all 6 generated int8 combinations
in §3/§5 produce non-silent, non-clipping audio (RMS/peak sanity, same check as Phase 6). Actual
human listening across all 8 voices for both fp32 and int8 is an **open item for the human**, not
silently skipped.

## 8. AR-loop length-sensitivity check

Directly addressed by §4 (fixed-input, single-frame divergence -- a length-independent per-step
measurement) and corroborated by §5's pattern (correlation drops as utterance length/AR-step-count
increases: 0.90 for ~7-word input down to ~0.47-0.49 for a 12-word input). Both point the same
direction: **quantization error is real and appears to compound with generation length**, matching
spec.md §13's predicted risk rather than being a purely theoretical concern.

## Proposed acceptance call — PROPOSED, needs human confirmation

Per the task's explicit instruction, this is a proposal with reasoning, not a decision:

- **text_encoder, prefix_step, the two codec decoders (MatMul/Gemm-only int8):** the one direct,
  isolated-input measurement available (§4's `text_encoder` check, 2.4% relative L2) looks like
  ordinary, unremarkable quantization drift. Combined with the ~63% total size reduction and ~18%
  RTF improvement, **I'd propose these four are acceptable to ship int8 by default**, but this
  rests on one measurement (`text_encoder`'s `soa_embed`) standing in for all four feed-forward
  graphs -- `prefix_step`'s own output wasn't separately isolated-input-tested here for time
  reasons, which a human reviewer should weigh.
- **`local_frame_decode`:** given the 15/16-codebook divergence on a single, fixed-input,
  fixed-randomness frame, and spec.md §13's explicit warning about compounding AR-loop error, **I'd
  propose treating dynamic quantization of this specific graph as NOT acceptable as-is**, and that
  Phase 11 (static/QDQ quantization, which calibrates against real activation ranges rather than
  quantizing blind) should be attempted for this graph specifically, per spec.md §9's explicit
  "attempt only if dynamic quantization's quality loss isn't acceptable" condition -- which, for
  this one graph, this data suggests has been met.
- **Numeric thresholds proposed for future automated pass/fail** (since spec.md §10 asked for
  these to be set once a baseline exists): relative L2 error on a feed-forward graph's output
  &lt;5% (text_encoder's 2.4% comfortably clears this); for `local_frame_decode`, a
  fixed-input/fixed-randomness codebook-agreement rate &gt;80% (i.e. &lt;20% of codebooks flipping)
  as a first bar -- current dynamic quantization is nowhere close (6.25% agreement), which is the
  concrete number behind the recommendation above.

**This is not a final decision.** A human should confirm or override both the per-graph
acceptance calls and the specific numeric thresholds before this int8 build is treated as
shippable, per the task's explicit instruction not to unilaterally decide quantization quality is
"acceptable."

## 9. Phase 11 (conditional): static/QDQ quantization of `local_frame_decode`, attempted

Per spec.md §9's "attempt only if dynamic quantization's quality loss isn't acceptable" and §4
above's finding, static/QDQ quantization was attempted for `local_frame_decode` specifically
(`scripts/zerotts/quantize_static_local_frame_decode.py`).

**Calibration data:** 168 real `global_hidden` activations captured from the fp32 Python
reference, across 8 diverse Vietnamese calibration sentences (plain greetings, dates, a percent
figure, code-switched "ZeroTTS"/"CPU"/"GPU", a phone number) × ~20 AR steps each. The other float
inputs (temperatures/top-k/top-p/repetition-penalty/cfg_scale) are fixed production constants, not
data-dependent activations, so a calibration *range* for them isn't meaningful the way it is for
`global_hidden`.

**Result: quantize_static succeeded and produced a loadable model** (`local_frame_decode.qdq.onnx`,
84.4MB, smaller than the dynamic version's 130MB) using the same `op_types_to_quantize=["MatMul",
"Gemm"]` restriction as the dynamic case (Conv still has no usable CPU EP kernel for quantized
execution in this onnxruntime build). **But re-running §4's exact fixed-input/fixed-randomness
divergence check against it found no meaningful improvement: 14 of 16 codebooks still differ from
fp32** (vs. 15/16 for dynamic quantization of the same graph).

**Interpretation:** this graph's ONNX export does not expose its pre-sampling logits/probabilities
as an output (only the already-sampled `is_eoa`/`codes`), so there's no way to measure "how much
did quantization move the probability mass" directly -- only "did the discrete sampling outcome
change," which this quick calibration pass shows is still highly sensitive to *any* int8
quantization of this graph's weights, not specifically to a lack of calibration. A larger/more
diverse calibration set was not tried (time-boxed for this pass) and might do better, but the
size of the gap (94% -> 88% still differing) makes that a weak bet without evidence.

**Proposed revised recommendation, still PROPOSED / needs human confirmation:** given static
quantization didn't resolve the issue in this attempt, the most defensible interim path is
**mixed precision**: ship `text_encoder.int8.onnx`, `prefix_step.int8.onnx`, and the two codec
`*.int8.onnx` graphs as int8 (the ~63% size reduction and ~18% RTF improvement mostly come from
`text_encoder`/`prefix_step`, the two largest graphs, so this captures most of the benefit), while
keeping **`local_frame_decode.onnx` in fp32**. `OfflineTtsZeroTtsModelConfig` already supports this
natively -- each of the 5 graph paths is independently configurable, so no code change is needed,
only which paths a deployment points at. This is a proposal for a human to confirm, override, or
ask for a larger calibration-set retry on, not a final call.

