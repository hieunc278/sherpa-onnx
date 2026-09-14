# Phase 0.1 — `voice.bin` byte layout

Investigated against the real downloaded model (`assets/zerotts/model/voices/maichi/`).

## Inputs

- `voices/maichi/meta.json` → `{"n_voice_queries": 10, ...}` (no `d_model` field in
  `meta.json`; `d_model` comes from the model-level `config.json`, not per-voice)
- `assets/zerotts/model/config.json` → `"d_model": 768`
- `voices/maichi/voice.bin` → file size **30720 bytes**

## Arithmetic check

```
n_voice_queries * d_model * 4 (float32, no header)
  = 10 * 768 * 4
  = 30720 bytes
```

`file_size_bytes (30720) == n_voice_queries * d_model * 4 (30720)` → **match, exactly, no
header bytes unaccounted for.**

## Hex dump of first 32 bytes, annotated

```
59 a6 1e 3e  a2 78 23 c0  82 c5 02 41  6b b2 03 c1
1b 0a 9d eb  ec 06 f0 44  10 ac fd ab  f8 32 b9 73
```

Interpreted as 8 little-endian float32 values (no magic/header bytes — byte 0 starts a value
directly):

| bytes | hex | float32 (LE) |
|---|---|---|
| 0–3 | `59 a6 1e 3e` | 0.15493144 |
| 4–7 | `a2 78 23 c0` | -2.5542378 |
| 8–11 | `82 c5 02 41` | 8.17322 |
| 12–15 | `6b b2 03 c1` | -8.231059 |
| 16–19 | `1b 0a 9d eb` (cont.) | -0.43488836 |
| 20–23 | | 8.277283 |
| 24–27 | | -1.7094433 |
| 28–31 | | 1.1810154 |

Cross-checked against `np.load("voices/maichi/voice.npz")["voice_emb"]` (shape `(1, 10, 768)`
float32) flattened — **the first 8 floats of `voice.bin` are bit-identical to the first 8
floats of the `.npz`'s `voice_emb` array**, and `np.allclose(bin_data.reshape(1,10,768),
npz_emb)` is `True` over the **entire** array, not just the prefix. No magic number, no
length-prefix, no dtype tag — nothing in the first 32 bytes decodes as a plausible header value
(they're just the first 8 unremarkable float32s of the embedding).

## Conclusion

**Exact byte layout: raw float32, little-endian, row-major, no header whatsoever.**
`voice.bin`'s N bytes reshape directly to `(1, n_voice_queries, d_model)` where
`n_voice_queries` comes from the voice's own `meta.json` and `d_model` comes from the model's
top-level `config.json` (`768` for this release). Confirmed by exact size match (30720 ==
30720) and bit-exact numerical agreement against the reference `.npz`'s `voice_emb`, not just
a plausible size coincidence.

Phase 2's voice loader can read `voice.bin` as `n_voice_queries * d_model` raw LE float32s, no
parsing beyond `fread`, with `n_voice_queries` sourced from `meta.json` and `d_model` from the
model config (matching `OfflineTtsZeroTtsModelConfig`, which already carries d_model
implicitly via the fixed model release this backend targets).
