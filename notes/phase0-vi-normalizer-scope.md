# Phase 0.4 — Vietnamese normalizer port scope

Source read in full: `src/zerotts/text_norm/vi_normalizer.py` (621 lines) and
`src/zerotts/text_norm/data/abbreviations.txt` (2258 lines, `ABBR:reading[,reading...]` per line,
both upper- and lower-case spellings as separate keys).

## Key portability constraint driving every decision below

`vi_normalizer.py`'s pattern scanner (`_SCANNER`) is **one big alternation regex that leans
heavily on negative lookbehind assertions** (`(?<!...)`) to stop one alternative from matching
inside a longer one (e.g. "don't match a bare number if it's actually part of a date"). **C++
`std::regex`'s ECMAScript grammar does not support lookbehind at all** (this is a real,
load-bearing limitation, not a style preference — confirmed against the C++ standard's regex
grammar). Two ways around it exist: (a) hand-roll a scanner that checks the preceding character in
code instead of via regex lookbehind, or (b) bring in a third-party regex engine (RE2/PCRE) with
lookbehind/lookaround support. Given sherpa-onnx has no existing regex-engine dependency beyond
`std::regex` (checked: no PCRE/RE2 in `cmake/`), **decision: hand-roll the scanner, no new
dependency**, porting only the categories where doing so is worth the manual-boundary-check
rewrite for a first pass.

## Every category found in the source, with a port/defer decision

| # | Category | Regex/logic involved | Lookbehind-heavy? | Decision | Reason |
|---|---|---|---|---|---|
| 1 | Number expansion (`expand_number`/`expand_digit`/`_split_chunks`/`_speak_chunk`/`_apply_sandhi`) — integers, decimals (`,` and `.`), signed, thousands-separated, small arithmetic expressions | Pure string/arithmetic logic, **no regex at all** | No | **Port for v1** | Highest value (numbers are extremely common in TTS input), zero lookbehind dependency, self-contained — this is the core of the whole module and trivially portable 1:1. |
| 2 | Percent (`25%`, `12,5 %` → "... phần trăm") | `(?<![\w.,])...%(?!\w)` | Yes (guard only) | **Port for v1** | High value, and the lookbehind is just "don't match mid-token" — replaceable with a manual left/right character check in C++ after a lookbehind-free regex match. |
| 3 | Time (`HH:MM:SS`, `HHhMMmSS`, `HH:MM`, `HHhMM`, `HHgMM`, bare `HHh`) | `(?<![\d:])...` guards | Yes (guard only) | **Port for v1** | Common in spoken content (schedules, news); same manual-guard replacement as #2. |
| 4 | Date `DD/MM/YYYY`, `DD-MM-YYYY`, `DD.MM.YYYY` | `(?<![\d/.\-])...(?![\d/-])(?!\.\d)` guards | Yes (guard only) | **Port for v1** | Common; same manual-guard replacement. |
| 5 | Month-year `MM/YYYY`, `MM-YYYY` | Same guard style, plus a **content lookbehind** (checks up to 8 chars back for the word "tháng" to avoid doubling it) | Yes (content, not just boundary) | **Port for v1, simplified** | Port the date-without-"tháng" case (always emit "tháng X năm Y"); accept the rare cosmetic doubling ("tháng 8/2024" after an already-written "tháng" → "tháng tháng tám năm...") as a known, documented v1 simplification rather than porting the context-lookback. |
| 6 | Abbreviations/acronyms (`ABBR:reading` dictionary, uppercase-only, dotted forms like `TP.HCM`) | `(?<![\w.])[UPPER][UPPER\d]+(?:\.[UPPER][UPPER\d]*)*(?![lower]|\d)` guard; dictionary lookup has no lookbehind | Yes (guard only) | **Port for v1** | High value (Vietnamese news/formal text is full of acronyms); ships `abbreviations.txt` verbatim as a runtime-loaded data file (same format, same file), avoiding re-encoding 2258 lines into C++ source. Dotted lookup (`TP.HCM` is a literal dict key) already covers the common prefix-abbreviation case — the separate `_PREFIX_ABBR`/`pfx` regex branch (splits "T.Ư" into parts and joins) is **deferred** (see #11). |
| 7 | Day-month after a cue word (`ngày 3/4` → date, bare `3/4` → fraction) | Cue-word lookaround + 40-char-back context lookbehind for "và"/"hoặc" disambiguation | Yes (content, multi-branch) | **Defer** | Real ambiguity-resolution logic the CRF used to handle upstream; the plain fraction reading (`#8`) is a safe, simpler fallback that's correct in the more common case (bare fraction) and only wrong in the specific "date after a bare cue word" case — an acceptable v1 gap, documented rather than silently wrong. |
| 8 | Fraction `a/b` (when not a date) | `(?<![\w/.,])...(?![\w/,])(?!\.\d)` guards | Yes (guard only) | **Port for v1** | Cheap once the date patterns (#4/#5) are checked first in priority order; same manual-guard replacement. |
| 9 | Version strings (`v1.2`, `1.2.3`) | Guard + a 1000-separator disambiguation regex | Yes | **Defer** | Rare in spoken/prose Vietnamese TTS input (changelogs/code contexts are not this model's target use); low value for the complexity of another disambiguation branch. |
| 10 | Degree (`38°C`) | Guard-only | Yes (guard only) | **Defer** | Low expected frequency in the target use case (Vietnamese conversational/narrative TTS, not weather data feeds); revisit if real usage shows otherwise. |
| 11 | Prefix-abbreviation + proper noun (`TP. HCM` as one unit, splits `T.Ư`) | Guard + lookahead for a following capitalized word | Yes | **Defer** | The common case (`TP.HCM`, no space) is already covered by #6's dictionary lookup on the joined token; this branch only adds the rarer split-then-join path (`T.Ư` → "trung ương" from two separately-resolved halves). |
| 12 | Acronym pair over a slash (`USD/VND`) — left verbatim, matched only so #6 doesn't mis-claim half of it | Guard-only, and the action is a no-op (return unchanged) | Yes (guard only) | **Defer** | Because the action is "leave unchanged," *not* matching this pattern at all produces the same output unless the abbreviation scanner (#6) would otherwise wrongly claim one half — an edge case, acceptable v1 gap. |
| 13 | Alphanumeric code (`AB-1234`, `VN-215`) — letters spelled, digits spaced | Guard-only | Yes (guard only) | **Defer** | Identifier-style input (vehicle plates, product codes) is not the primary target content (Vietnamese narrative/conversational text); revisit if usage data shows otherwise. |
| 14 | `@` → "a còng" | Trivial single-char match, no guard needed beyond URL/email protection (#16) | No | **Port for v1** | Trivial to port once #16 (URL/email protection) exists to guard it. |
| 15 | CamelCase splitting (`ChatGPT` → `Chat GPT`) | Custom char-by-char scan (not the main `_SCANNER` regex), no lookbehind | No | **Defer** | Not regex/lookbehind-blocked at all (could be ported cheaply), but scoped out of v1 purely for time — it's a text-quality nicety (brand-name pronunciation) rather than a correctness-critical category like numbers/dates/times. Tracked as a fast-follow, not blocked on any technical issue. |
| 16 | URL/email protection (`_PROTECTED_RE`) — skip normalizing inside URLs/emails | Alternation regex, no lookbehind (it's a "find and skip" pass, not a guard) | No | **Port for v1** | No lookbehind involved, cheap, and a **prerequisite guard** for #2/#3/#4/#6/#14 above — without it, e.g. a phone number inside a URL or the digits in an email could get mis-normalized. |
| 17 | Roman numerals as ordinals (`quý III` → "quý ba") | Cue-word lookbehind (12-char window) | Yes (content) | **Defer** | Niche (formal/legal register); low expected frequency. |

## Summary: v1 port scope

**Port for v1** (8 of 17 categories, chosen for highest value-per-lookbehind-workaround-cost):
numbers/digits (#1), percent (#2), time (#3), date DD/MM/YYYY (#4), month-year (#5, simplified),
abbreviations via the shipped `abbreviations.txt` data file (#6), fraction (#8), `@` sign (#14),
plus URL/email protection (#16) as a prerequisite guard for the others.

**Defer** (9 categories, each with its own one-line reason above): day-month cue disambiguation
(#7), version strings (#9), degree symbol (#10), prefix-abbreviation splitting (#11), acronym pair
over slash (#12), alphanumeric codes (#13), CamelCase splitting (#15), roman numeral ordinals
(#17).

Implementation approach for the 9 ported categories: a hand-rolled left-to-right scanner in C++
(not `std::regex` with lookbehind, which doesn't exist) — for each category, use a lookbehind-free
`std::regex` (or manual character scanning) to find candidate spans, then manually inspect the
character immediately before/after the match in code to enforce the same "don't match mid-token"
guards the Python lookbehinds enforce. This is Phase 5's implementation plan; this note only
records *what* is in scope, not the final code.

**Done when:** every distinct normalization category actually present in `vi_normalizer.py` has
its own row above with an explicit decision and reason — 17 rows, none grouped away, satisfying
Phase 0.4's exit condition.
