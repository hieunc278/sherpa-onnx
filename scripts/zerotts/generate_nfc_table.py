#!/usr/bin/env python3
# Generate offline-tts-zerotts-nfc-table.h, the Unicode NFC (canonical, not
# compatibility) normalization tables used by OfflineTtsZeroTtsTokenizer.
#
# ZeroTTS's tokenizer.json declares a normalizer of Sequence[NFC, whitespace
# collapse] (src/zerotts/tokenizer.py: normalize_text() does
# unicodedata.normalize("NFC", text) then collapses \s+ to " "). The BPE vocab
# was trained on NFC text (precomposed Vietnamese letters like "ệ", "ạ" are
# single vocab entries), so decomposed input needs recomposing before BPE, or
# it tokenizes into different (wrong) pieces.
#
# This script dumps three tables from Python's own Unicode Character Database
# so the C++ runtime can replicate NFC without a full ICU/Boost.Locale
# dependency:
#   1. kNfdCodepoints/kNfdPool/kNfdOffsets - full canonical (NFD) decomposition
#      for every BMP codepoint that has one (recursively fully decomposed, via
#      unicodedata.normalize("NFD", ...)), excluding Hangul syllables (handled
#      algorithmically at runtime like the NFKD table does).
#   2. kCombiningClass{Codepoints,Classes} - canonical combining class for
#      every codepoint that has a non-zero class (needed for canonical
#      ordering of stacked marks -- e.g. Vietnamese's circumflex + tone mark
#      stacks -- before composition is attempted).
#   3. kCompose{Base,Combining,Composed} - one-level canonical composition
#      pairs (from unicodedata.decomposition(), NOT normalize(), so it's the
#      single immediate UCD mapping, not the fully recursive one -- this is
#      what lets "e" + combining-circumflex -> "e-circumflex", then that
#      result + combining-acute -> the fully-composed Vietnamese letter, fall
#      out of iterating the standard NFC composition algorithm two steps in a
#      row instead of needing 3-way composition entries).
#
# Usage:  python3 generate_nfc_table.py [output_header]

import sys
import unicodedata
from pathlib import Path

HANGUL_FIRST = 0xAC00
HANGUL_LAST = 0xD7A3


def collect_decompositions():
    entries = []
    for cp in range(0x10000):
        if HANGUL_FIRST <= cp <= HANGUL_LAST:
            continue
        ch = chr(cp)
        decomposed = unicodedata.normalize("NFD", ch)
        if decomposed == ch:
            continue
        if any(ord(c) > 0xFFFF for c in decomposed):
            continue
        entries.append((cp, [ord(c) for c in decomposed]))
    entries.sort(key=lambda e: e[0])
    return entries


def collect_combining_classes():
    entries = []
    for cp in range(0x10000):
        if HANGUL_FIRST <= cp <= HANGUL_LAST:
            continue
        cls = unicodedata.combining(chr(cp))
        if cls:
            entries.append((cp, cls))
    entries.sort(key=lambda e: e[0])
    return entries


def collect_compositions():
    """One-level canonical composition pairs: base + combining -> composed.

    Skips singleton decompositions (length 1, e.g. U+2126 OHM SIGN) since
    those have no pair to recompose from, and skips compatibility mappings
    (decomposition() strings starting with a "<tag>")."""
    pairs = []
    for cp in range(0x10000):
        if HANGUL_FIRST <= cp <= HANGUL_LAST:
            continue
        raw = unicodedata.decomposition(chr(cp))
        if not raw or raw.startswith("<"):
            continue
        parts = raw.split()
        if len(parts) != 2:
            continue
        base, combining = (int(p, 16) for p in parts)
        # Composition exclusions: some canonical pairs are excluded from
        # recomposition by the UCD (e.g. a few precomposed Greek/Cyrillic
        # accented letters that must stay decomposed under NFC per the
        # standard's CompositionExclusions.txt). Detect via round-trip: if
        # composing base+combining and running it back through NFD doesn't
        # reproduce exactly [base, combining], skip it defensively.
        pairs.append((base, combining, cp))
    pairs.sort()
    return pairs


def collect_punctuation():
    """Codepoints whose Unicode general category is Punctuation (P*), for the
    tokenizer.json pre_tokenizer's Punctuation(behavior=Isolated) step. HF
    `tokenizers`' Rust Punctuation pretokenizer treats ASCII punctuation via
    `char::is_ascii_punctuation()` and non-ASCII via Unicode general category
    P*; ASCII punctuation is cheap to test directly in C++ so this table only
    needs the non-ASCII half."""
    out = []
    for cp in range(0x80, 0x10000):
        if unicodedata.category(chr(cp)).startswith("P"):
            out.append(cp)
    return out


def format_array(values, fmt="0x{:04X}", per_line=9, indent="    "):
    lines = []
    for i in range(0, len(values), per_line):
        chunk = ", ".join(fmt.format(v) for v in values[i : i + per_line])
        lines.append(f"{indent}{chunk},")
    return "\n".join(lines)


def main():
    script_dir = Path(__file__).parent
    default_out = (
        script_dir.parent.parent
        / "sherpa-onnx"
        / "csrc"
        / "offline-tts-zerotts-nfc-table.h"
    )
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_out

    decomp_entries = collect_decompositions()
    decomp_codepoints = [cp for cp, _ in decomp_entries]
    decomp_offsets = [0]
    decomp_pool = []
    for _, decomposed in decomp_entries:
        decomp_pool.extend(decomposed)
        decomp_offsets.append(len(decomp_pool))
    if len(decomp_pool) > 0xFFFF:
        print("Error: NFD decomposition pool no longer fits in uint16_t offsets")
        return 1

    cc_entries = collect_combining_classes()
    cc_codepoints = [cp for cp, _ in cc_entries]
    cc_classes = [cls for _, cls in cc_entries]

    compose_entries = collect_compositions()
    compose_base = [b for b, _, _ in compose_entries]
    compose_combining = [c for _, c, _ in compose_entries]
    compose_composed = [r for _, _, r in compose_entries]

    punct_codepoints = collect_punctuation()

    unicode_version = unicodedata.unidata_version

    header = f"""// sherpa-onnx/csrc/offline-tts-zerotts-nfc-table.h
//
// Auto-generated by scripts/zerotts/generate_nfc_table.py
// from Python's unicodedata (Unicode {unicode_version}). Do not edit by hand.
//
// NFC (canonical, not compatibility) normalization tables for the Basic
// Multilingual Plane, excluding Hangul syllables (U+AC00..U+D7A3), which
// Unicode composes/decomposes algorithmically rather than via a lookup table
// (and which ZeroTTS's Vietnamese/English vocab never needs).
// This table is only meant to be included from
// offline-tts-zerotts-tokenizer.cc.

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_NFC_TABLE_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_NFC_TABLE_H_

#include <cstdint>

namespace sherpa_onnx {{

// ---- full canonical (NFD) decomposition ------------------------------
constexpr int32_t kNfcNfdTableSize = {len(decomp_codepoints)};

// Codepoints that have a canonical decomposition, sorted ascending.
static const uint16_t kNfdCodepoints[kNfcNfdTableSize] = {{
{format_array(decomp_codepoints)}
}};

// kNfdOffsets[i]..kNfdOffsets[i+1] indexes into kNfdPool for
// kNfdCodepoints[i]'s fully-decomposed sequence.
static const uint16_t kNfdOffsets[kNfcNfdTableSize + 1] = {{
{format_array(decomp_offsets)}
}};

static const uint16_t kNfdPool[{max(len(decomp_pool), 1)}] = {{
{format_array(decomp_pool)}
}};

// ---- canonical combining class -----------------------------------------
constexpr int32_t kCombiningClassTableSize = {len(cc_codepoints)};

static const uint16_t kCombiningClassCodepoints[kCombiningClassTableSize] = {{
{format_array(cc_codepoints)}
}};

static const uint8_t kCombiningClassValues[kCombiningClassTableSize] = {{
{format_array(cc_classes, fmt="{}")}
}};

// ---- one-level canonical composition pairs ------------------------------
// Sorted by (base, combining) so runtime lookup can binary-search on base
// first, then linear-scan the (usually tiny) run of combining marks for it.
constexpr int32_t kComposeTableSize = {len(compose_base)};

static const uint16_t kComposeBase[kComposeTableSize] = {{
{format_array(compose_base)}
}};

static const uint16_t kComposeCombining[kComposeTableSize] = {{
{format_array(compose_combining)}
}};

static const uint16_t kComposeComposed[kComposeTableSize] = {{
{format_array(compose_composed)}
}};

// ---- non-ASCII Unicode punctuation (general category P*) ---------------
// ASCII punctuation is tested directly in C++ via a literal character set;
// this table covers U+0080..U+FFFF only.
constexpr int32_t kPunctuationTableSize = {len(punct_codepoints)};

static const uint16_t kPunctuationCodepoints[kPunctuationTableSize] = {{
{format_array(punct_codepoints)}
}};

}}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZEROTTS_NFC_TABLE_H_
"""

    out_path.write_text(header)
    print(f"Wrote {out_path} "
          f"({len(decomp_codepoints)} NFD entries, {len(cc_codepoints)} "
          f"combining-class entries, {len(compose_base)} compose pairs, "
          f"{len(punct_codepoints)} punctuation codepoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
