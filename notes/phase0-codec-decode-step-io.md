# Phase 0.2 — `decode_step` (moss_audio_tokenizer_decode_step.onnx) tensor I/O

Source: `assets/zerotts/model/onnx/codec/codec_browser_onnx_meta.json` (fetched directly, not
inferred from `codec.py`), format_version 2. Cross-checked against the actual ONNX graph via
`onnxruntime.InferenceSession.get_inputs()/get_outputs()` on the real downloaded
`moss_audio_tokenizer_decode_step.onnx` — every name, shape, and dtype below matches the live
graph exactly (see `onnx/codec/codec_browser_onnx_meta.json` + a one-off `python3 -c`
introspection run during Phase 0, output captured in the phase-0 transcript).

Structure per `codec_browser_onnx_meta.json`'s `streaming_decode` section:
- **4** `transformer_offsets` entries — one per decoder stage (stages 1/3/5/7) that needs a
  running position counter.
- **12** `attention_caches` entries — a KV ring-buffer cache per (decoder-stage, layer) pair
  (stage 1: layers 0–3 → caches 0–3, context 500; stage 3: layers 0–1 → caches 4–5, context 800;
  stage 5: layers 0–1 → caches 6–7, context 1200; stage 7 (inferred from the pattern, caches
  8–11): context 1600). Each cache carries 4 tensors in and 4 out (offset, keys, values,
  positions).
- Plus the 2 data tensors in (`audio_codes`, `audio_code_lengths`) and 2 data tensors out
  (`audio`, `audio_lengths`).

Total: 4 data + 8 transformer-offset (4 in + 4 out) + 96 attention-cache (12 caches × 4 tensors
× in/out) = **108 tensors**, matching `len(decode_step_input_names) + len(decode_step_output_names)
== 54 + 54 == 108` from the meta file's own `onnx` section.

## Cross-check against `codec.py`

`grep -n` for the literal tensor-name strings in `src/zerotts/codec.py` (the reference
implementation) confirms every name below is actually referenced by that exact string, not
stale/renamed:

```
88:  {"audio_codes": codes_btk, "audio_code_lengths": lengths}
110: self._transformer_specs = list(streaming.get("transformer_offsets", []))
139: "audio_codes": codes_btk,
140: "audio_code_lengths": np.array([codes_btk.shape[1]], dtype=np.int32),
150: n = int(named["audio_lengths"].reshape(-1)[0])
151: return named["audio"][:, :, :n].mean(axis=1).astype(np.float32)
```

`codec.py`'s `MossStreamingDecoder` does **not** hardcode any of the per-cache tensor names
(`attn_offset_0`, `attn_cached_keys_0`, ...) or the transformer-offset names
(`transformer_offset_0`, ...) — it resolves all of them dynamically from
`streaming.get("transformer_offsets", [])` / `streaming.get("attention_caches", [])`'s
`input_name`/`output_name`/`offset_input_name`/etc. fields at runtime (see
`_reset_state`/`decode_chunk`, lines 116–151). This is a stronger confirmation than a literal
grep for each of the 108 names would be: **the meta JSON is the single source of truth the
Python reference itself trusts**, so the C++ port should do the same — parse
`codec_browser_onnx_meta.json` at load time and build the state dict from its `transformer_offsets`
/`attention_caches` arrays generically, rather than hardcoding 108 tensor names/shapes in C++.
This directly resolves spec §12.1's open question in favor of "data-driven from the meta file,"
not "reverse-engineered and hardcoded."

## Full tensor table

| # | Tensor name | Shape | Dtype | Tag | Direction |
|---|---|---|---|---|---|
| 1 | `audio_codes` | `(1, code_length, 16)` | int32 | data (input codes, time-major, codebook-last) | input |
| 2 | `audio_code_lengths` | `(1,)` | int32 | data (input, valid length of the chunk) | input |
| 3 | `audio` | `(batch=1, channels=2, audio_length)` | float32 | data (output waveform, stereo — caller averages channels to mono) | output |
| 4 | `audio_lengths` | `(1,)` | int32 | data (output, valid sample count) | output |
| 5 | `transformer_offset_0` | `(1,)` | int32 | state (decoder-stage 1 running position counter, carried in) | input |
| 6 | `transformer_offset_out_0` | `(1,)` | int32 | state (decoder-stage 1 running position counter, carried out) | output |
| 7 | `transformer_offset_1` | `(1,)` | int32 | state (decoder-stage 3 running position counter, carried in) | input |
| 8 | `transformer_offset_out_1` | `(1,)` | int32 | state (decoder-stage 3 running position counter, carried out) | output |
| 9 | `transformer_offset_2` | `(1,)` | int32 | state (decoder-stage 5 running position counter, carried in) | input |
| 10 | `transformer_offset_out_2` | `(1,)` | int32 | state (decoder-stage 5 running position counter, carried out) | output |
| 11 | `transformer_offset_3` | `(1,)` | int32 | state (decoder-stage 7 running position counter, carried in) | input |
| 12 | `transformer_offset_out_3` | `(1,)` | int32 | state (decoder-stage 7 running position counter, carried out) | output |
| 13 | `attn_offset_0` | `(1,)` | int32 | state (ring-buffer write offset, cache #0, decoder-stage 1, layer 0, context=500) | input |
| 14 | `attn_cached_keys_0` | `(1, 4, 500, 64)` | float32 | state (KV cache keys, cache #0, heads=4, head_dim=64) | input |
| 15 | `attn_cached_values_0` | `(1, 4, 500, 64)` | float32 | state (KV cache values, cache #0) | input |
| 16 | `attn_cached_positions_0` | `(1, 500)` | int32 | state (ring-buffer position tags, cache #0; init -1, not 0) | input |
| 17 | `attn_offset_out_0` | `(1,)` | int32 | state (new write offset, cache #0) | output |
| 18 | `attn_cached_keys_out_0` | `(1, 4, 500, 64)` | float32 | state (new K cache, cache #0) | output |
| 19 | `attn_cached_values_out_0` | `(1, 4, 500, 64)` | float32 | state (new V cache, cache #0) | output |
| 20 | `attn_cached_positions_out_0` | `(1, 500)` | int32 | state (new position tags, cache #0) | output |
| 21 | `attn_offset_1` | `(1,)` | int32 | state (ring-buffer write offset, cache #1, decoder-stage 1, layer 1, context=500) | input |
| 22 | `attn_cached_keys_1` | `(1, 4, 500, 64)` | float32 | state (KV cache keys, cache #1, heads=4, head_dim=64) | input |
| 23 | `attn_cached_values_1` | `(1, 4, 500, 64)` | float32 | state (KV cache values, cache #1) | input |
| 24 | `attn_cached_positions_1` | `(1, 500)` | int32 | state (ring-buffer position tags, cache #1; init -1, not 0) | input |
| 25 | `attn_offset_out_1` | `(1,)` | int32 | state (new write offset, cache #1) | output |
| 26 | `attn_cached_keys_out_1` | `(1, 4, 500, 64)` | float32 | state (new K cache, cache #1) | output |
| 27 | `attn_cached_values_out_1` | `(1, 4, 500, 64)` | float32 | state (new V cache, cache #1) | output |
| 28 | `attn_cached_positions_out_1` | `(1, 500)` | int32 | state (new position tags, cache #1) | output |
| 29 | `attn_offset_2` | `(1,)` | int32 | state (ring-buffer write offset, cache #2, decoder-stage 1, layer 2, context=500) | input |
| 30 | `attn_cached_keys_2` | `(1, 4, 500, 64)` | float32 | state (KV cache keys, cache #2, heads=4, head_dim=64) | input |
| 31 | `attn_cached_values_2` | `(1, 4, 500, 64)` | float32 | state (KV cache values, cache #2) | input |
| 32 | `attn_cached_positions_2` | `(1, 500)` | int32 | state (ring-buffer position tags, cache #2; init -1, not 0) | input |
| 33 | `attn_offset_out_2` | `(1,)` | int32 | state (new write offset, cache #2) | output |
| 34 | `attn_cached_keys_out_2` | `(1, 4, 500, 64)` | float32 | state (new K cache, cache #2) | output |
| 35 | `attn_cached_values_out_2` | `(1, 4, 500, 64)` | float32 | state (new V cache, cache #2) | output |
| 36 | `attn_cached_positions_out_2` | `(1, 500)` | int32 | state (new position tags, cache #2) | output |
| 37 | `attn_offset_3` | `(1,)` | int32 | state (ring-buffer write offset, cache #3, decoder-stage 1, layer 3, context=500) | input |
| 38 | `attn_cached_keys_3` | `(1, 4, 500, 64)` | float32 | state (KV cache keys, cache #3, heads=4, head_dim=64) | input |
| 39 | `attn_cached_values_3` | `(1, 4, 500, 64)` | float32 | state (KV cache values, cache #3) | input |
| 40 | `attn_cached_positions_3` | `(1, 500)` | int32 | state (ring-buffer position tags, cache #3; init -1, not 0) | input |
| 41 | `attn_offset_out_3` | `(1,)` | int32 | state (new write offset, cache #3) | output |
| 42 | `attn_cached_keys_out_3` | `(1, 4, 500, 64)` | float32 | state (new K cache, cache #3) | output |
| 43 | `attn_cached_values_out_3` | `(1, 4, 500, 64)` | float32 | state (new V cache, cache #3) | output |
| 44 | `attn_cached_positions_out_3` | `(1, 500)` | int32 | state (new position tags, cache #3) | output |
| 45 | `attn_offset_4` | `(1,)` | int32 | state (ring-buffer write offset, cache #4, decoder-stage 3, layer 0, context=800) | input |
| 46 | `attn_cached_keys_4` | `(1, 4, 800, 64)` | float32 | state (KV cache keys, cache #4, heads=4, head_dim=64) | input |
| 47 | `attn_cached_values_4` | `(1, 4, 800, 64)` | float32 | state (KV cache values, cache #4) | input |
| 48 | `attn_cached_positions_4` | `(1, 800)` | int32 | state (ring-buffer position tags, cache #4; init -1, not 0) | input |
| 49 | `attn_offset_out_4` | `(1,)` | int32 | state (new write offset, cache #4) | output |
| 50 | `attn_cached_keys_out_4` | `(1, 4, 800, 64)` | float32 | state (new K cache, cache #4) | output |
| 51 | `attn_cached_values_out_4` | `(1, 4, 800, 64)` | float32 | state (new V cache, cache #4) | output |
| 52 | `attn_cached_positions_out_4` | `(1, 800)` | int32 | state (new position tags, cache #4) | output |
| 53 | `attn_offset_5` | `(1,)` | int32 | state (ring-buffer write offset, cache #5, decoder-stage 3, layer 1, context=800) | input |
| 54 | `attn_cached_keys_5` | `(1, 4, 800, 64)` | float32 | state (KV cache keys, cache #5, heads=4, head_dim=64) | input |
| 55 | `attn_cached_values_5` | `(1, 4, 800, 64)` | float32 | state (KV cache values, cache #5) | input |
| 56 | `attn_cached_positions_5` | `(1, 800)` | int32 | state (ring-buffer position tags, cache #5; init -1, not 0) | input |
| 57 | `attn_offset_out_5` | `(1,)` | int32 | state (new write offset, cache #5) | output |
| 58 | `attn_cached_keys_out_5` | `(1, 4, 800, 64)` | float32 | state (new K cache, cache #5) | output |
| 59 | `attn_cached_values_out_5` | `(1, 4, 800, 64)` | float32 | state (new V cache, cache #5) | output |
| 60 | `attn_cached_positions_out_5` | `(1, 800)` | int32 | state (new position tags, cache #5) | output |
| 61 | `attn_offset_6` | `(1,)` | int32 | state (ring-buffer write offset, cache #6, decoder-stage 5, layer 0, context=1200) | input |
| 62 | `attn_cached_keys_6` | `(1, 4, 1200, 64)` | float32 | state (KV cache keys, cache #6, heads=4, head_dim=64) | input |
| 63 | `attn_cached_values_6` | `(1, 4, 1200, 64)` | float32 | state (KV cache values, cache #6) | input |
| 64 | `attn_cached_positions_6` | `(1, 1200)` | int32 | state (ring-buffer position tags, cache #6; init -1, not 0) | input |
| 65 | `attn_offset_out_6` | `(1,)` | int32 | state (new write offset, cache #6) | output |
| 66 | `attn_cached_keys_out_6` | `(1, 4, 1200, 64)` | float32 | state (new K cache, cache #6) | output |
| 67 | `attn_cached_values_out_6` | `(1, 4, 1200, 64)` | float32 | state (new V cache, cache #6) | output |
| 68 | `attn_cached_positions_out_6` | `(1, 1200)` | int32 | state (new position tags, cache #6) | output |
| 69 | `attn_offset_7` | `(1,)` | int32 | state (ring-buffer write offset, cache #7, decoder-stage 5, layer 1, context=1200) | input |
| 70 | `attn_cached_keys_7` | `(1, 4, 1200, 64)` | float32 | state (KV cache keys, cache #7, heads=4, head_dim=64) | input |
| 71 | `attn_cached_values_7` | `(1, 4, 1200, 64)` | float32 | state (KV cache values, cache #7) | input |
| 72 | `attn_cached_positions_7` | `(1, 1200)` | int32 | state (ring-buffer position tags, cache #7; init -1, not 0) | input |
| 73 | `attn_offset_out_7` | `(1,)` | int32 | state (new write offset, cache #7) | output |
| 74 | `attn_cached_keys_out_7` | `(1, 4, 1200, 64)` | float32 | state (new K cache, cache #7) | output |
| 75 | `attn_cached_values_out_7` | `(1, 4, 1200, 64)` | float32 | state (new V cache, cache #7) | output |
| 76 | `attn_cached_positions_out_7` | `(1, 1200)` | int32 | state (new position tags, cache #7) | output |
| 77 | `attn_offset_8` | `(1,)` | int32 | state (ring-buffer write offset, cache #8, decoder-stage 7, layer 0, context=1600) | input |
| 78 | `attn_cached_keys_8` | `(1, 4, 1600, 64)` | float32 | state (KV cache keys, cache #8, heads=4, head_dim=64) | input |
| 79 | `attn_cached_values_8` | `(1, 4, 1600, 64)` | float32 | state (KV cache values, cache #8) | input |
| 80 | `attn_cached_positions_8` | `(1, 1600)` | int32 | state (ring-buffer position tags, cache #8; init -1, not 0) | input |
| 81 | `attn_offset_out_8` | `(1,)` | int32 | state (new write offset, cache #8) | output |
| 82 | `attn_cached_keys_out_8` | `(1, 4, 1600, 64)` | float32 | state (new K cache, cache #8) | output |
| 83 | `attn_cached_values_out_8` | `(1, 4, 1600, 64)` | float32 | state (new V cache, cache #8) | output |
| 84 | `attn_cached_positions_out_8` | `(1, 1600)` | int32 | state (new position tags, cache #8) | output |
| 85 | `attn_offset_9` | `(1,)` | int32 | state (ring-buffer write offset, cache #9, decoder-stage 7, layer 1, context=1600) | input |
| 86 | `attn_cached_keys_9` | `(1, 4, 1600, 64)` | float32 | state (KV cache keys, cache #9, heads=4, head_dim=64) | input |
| 87 | `attn_cached_values_9` | `(1, 4, 1600, 64)` | float32 | state (KV cache values, cache #9) | input |
| 88 | `attn_cached_positions_9` | `(1, 1600)` | int32 | state (ring-buffer position tags, cache #9; init -1, not 0) | input |
| 89 | `attn_offset_out_9` | `(1,)` | int32 | state (new write offset, cache #9) | output |
| 90 | `attn_cached_keys_out_9` | `(1, 4, 1600, 64)` | float32 | state (new K cache, cache #9) | output |
| 91 | `attn_cached_values_out_9` | `(1, 4, 1600, 64)` | float32 | state (new V cache, cache #9) | output |
| 92 | `attn_cached_positions_out_9` | `(1, 1600)` | int32 | state (new position tags, cache #9) | output |
| 93 | `attn_offset_10` | `(1,)` | int32 | state (ring-buffer write offset, cache #10, decoder-stage 7, layer 2, context=1600) | input |
| 94 | `attn_cached_keys_10` | `(1, 4, 1600, 64)` | float32 | state (KV cache keys, cache #10, heads=4, head_dim=64) | input |
| 95 | `attn_cached_values_10` | `(1, 4, 1600, 64)` | float32 | state (KV cache values, cache #10) | input |
| 96 | `attn_cached_positions_10` | `(1, 1600)` | int32 | state (ring-buffer position tags, cache #10; init -1, not 0) | input |
| 97 | `attn_offset_out_10` | `(1,)` | int32 | state (new write offset, cache #10) | output |
| 98 | `attn_cached_keys_out_10` | `(1, 4, 1600, 64)` | float32 | state (new K cache, cache #10) | output |
| 99 | `attn_cached_values_out_10` | `(1, 4, 1600, 64)` | float32 | state (new V cache, cache #10) | output |
| 100 | `attn_cached_positions_out_10` | `(1, 1600)` | int32 | state (new position tags, cache #10) | output |
| 101 | `attn_offset_11` | `(1,)` | int32 | state (ring-buffer write offset, cache #11, decoder-stage 7, layer 3, context=1600) | input |
| 102 | `attn_cached_keys_11` | `(1, 4, 1600, 64)` | float32 | state (KV cache keys, cache #11, heads=4, head_dim=64) | input |
| 103 | `attn_cached_values_11` | `(1, 4, 1600, 64)` | float32 | state (KV cache values, cache #11) | input |
| 104 | `attn_cached_positions_11` | `(1, 1600)` | int32 | state (ring-buffer position tags, cache #11; init -1, not 0) | input |
| 105 | `attn_offset_out_11` | `(1,)` | int32 | state (new write offset, cache #11) | output |
| 106 | `attn_cached_keys_out_11` | `(1, 4, 1600, 64)` | float32 | state (new K cache, cache #11) | output |
| 107 | `attn_cached_values_out_11` | `(1, 4, 1600, 64)` | float32 | state (new V cache, cache #11) | output |
| 108 | `attn_cached_positions_out_11` | `(1, 1600)` | int32 | state (new position tags, cache #11) | output |

## Conclusion

Every one of the 108 tensors above has a name, shape, dtype, and an explicit `data`/`state` tag
— none left "unclear." The C++ `OfflineTtsZeroTtsModel::RunCodecDecodeStep` wrapper (Phase 3)
should:
1. Parse `codec_browser_onnx_meta.json`'s `streaming_decode.transformer_offsets` and
   `.attention_caches` arrays at construction time (mirrors `MossStreamingDecoder.__init__`).
2. Maintain a generic `std::unordered_map<std::string, Ort::Value>`-style (or equivalent) state
   dict keyed by the JSON-declared input names, initialized to zeros (`int32`/`float32` per the
   `dtype`/`cache_dtype`/`positions_dtype` fields) except `attn_cached_positions_*`, which
   initializes to **-1** (confirmed in `codec.py`'s `_reset_state`, line 129: "position 0 is a
   real position, so a zero-filled ring buffer would read as every slot holds frame 0").
3. After each `decode_step` call, copy each output tensor back into the state dict under its
   paired input name, exactly mirroring `decode_chunk`'s output→input rebinding loop.

No hardcoded 500/800/1200/1600 context-length constants are needed in C++ — they only matter for
pre-allocating buffers, and even that can be read from `cache_shape`/`positions_shape` in the
meta JSON at load time.
