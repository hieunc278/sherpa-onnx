#!/usr/bin/env python3
"""Phase 10 evaluation scoring: mel-spectrogram correlation (fp32 vs int8,
same input), RTF from timings.txt, and WER via whisper-tiny ASR (through
the sherpa-onnx-offline CLI) for both variants.

Usage: python3 score_eval.py
"""
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

REPO = Path("/mnt/disk2/projects/sherpa-onnx-zero-tts")
EVAL_DIR = REPO / "assets/zerotts/phase10_eval"
BUILD = REPO / "build_shared"
WHISPER_DIR = Path("/tmp/sherpa-onnx-whisper-tiny")

SENTENCES = {
    "short": "Xin chào.",
    "medium": "Xin chào Việt Nam.",
    "long": "Cảm ơn bạn đã sử dụng ZeroTTS, một mô hình chạy trên CPU, "
            "không cần GPU.",
}
VOICES = ["maichi", "giahuy"]


def normalize_for_wer(s: str) -> list:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    return s.split()


def wer(ref: list, hyp: list) -> float:
    # standard edit-distance WER
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / max(1, n)


def transcribe(wav_path: Path) -> str:
    ld_lib = None
    for p in BUILD.rglob("libonnxruntime.so*"):
        ld_lib = str(p.parent)
        break
    cmd = [
        str(BUILD / "bin/sherpa-onnx-offline"),
        f"--whisper-encoder={WHISPER_DIR}/tiny-encoder.onnx",
        f"--whisper-decoder={WHISPER_DIR}/tiny-decoder.onnx",
        f"--tokens={WHISPER_DIR}/tiny-tokens.txt",
        "--whisper-language=vi",
        "--whisper-task=transcribe",
        "--model-type=whisper",
        str(wav_path),
    ]
    env = {"LD_LIBRARY_PATH": ld_lib} if ld_lib else {}
    import os
    full_env = dict(os.environ)
    full_env.update(env)
    result = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
    text = ""
    for line in result.stdout.splitlines():
        if line.strip().startswith("{"):
            text = line.strip()
    # sherpa-onnx-offline prints a line like: text: ...
    m = re.search(r'"text"\s*:\s*"([^"]*)"', result.stdout)
    if m:
        text = m.group(1)
    else:
        m2 = re.search(r"^text:\s*(.*)$", result.stdout, re.MULTILINE)
        text = m2.group(1).strip() if m2 else ""
    return text, result.stdout


def spectrogram_correlation(a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int) -> float:
    import librosa
    Sa = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=sr_a, n_mels=80))
    Sb = librosa.power_to_db(librosa.feature.melspectrogram(y=b, sr=sr_b, n_mels=80))
    n = min(Sa.shape[1], Sb.shape[1])
    return float(np.corrcoef(Sa[:, :n].flatten(), Sb[:, :n].flatten())[0, 1])


def load_wav(path: Path):
    sr, d = wavfile.read(path)
    return d.astype(np.float32) / 32768.0, sr


def main():
    timings = {}
    for line in (EVAL_DIR / "timings.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        variant, voice, path, secs = parts
        timings[Path(path).name] = float(secs)

    rows = []
    for label, text in SENTENCES.items():
        for voice in VOICES:
            fp32_path = EVAL_DIR / f"fp32_{label}_{voice}.wav"
            int8_path = EVAL_DIR / f"int8_{label}_{voice}.wav"
            if not fp32_path.exists() or not int8_path.exists():
                print(f"SKIP missing {fp32_path} or {int8_path}")
                continue

            fp32_audio, fp32_sr = load_wav(fp32_path)
            int8_audio, int8_sr = load_wav(int8_path)

            corr = spectrogram_correlation(fp32_audio, fp32_sr, int8_audio, int8_sr)

            fp32_dur = len(fp32_audio) / fp32_sr
            int8_dur = len(int8_audio) / int8_sr
            fp32_rtf = timings.get(fp32_path.name, float("nan")) / fp32_dur
            int8_rtf = timings.get(int8_path.name, float("nan")) / int8_dur

            fp32_text, fp32_raw = transcribe(fp32_path)
            int8_text, int8_raw = transcribe(int8_path)

            ref_tokens = normalize_for_wer(text)
            fp32_wer = wer(ref_tokens, normalize_for_wer(fp32_text))
            int8_wer = wer(ref_tokens, normalize_for_wer(int8_text))

            rows.append({
                "label": label, "voice": voice, "text": text,
                "fp32_dur": fp32_dur, "int8_dur": int8_dur,
                "fp32_rtf": fp32_rtf, "int8_rtf": int8_rtf,
                "spectrogram_corr": corr,
                "fp32_asr": fp32_text, "int8_asr": int8_text,
                "fp32_wer": fp32_wer, "int8_wer": int8_wer,
            })
            print(f"{label}/{voice}: corr={corr:.3f} "
                 f"fp32_rtf={fp32_rtf:.2f} int8_rtf={int8_rtf:.2f} "
                 f"fp32_wer={fp32_wer:.2f} int8_wer={int8_wer:.2f}")
            print(f"  ref:  {text}")
            print(f"  fp32: {fp32_text}")
            print(f"  int8: {int8_text}")

    import json
    (EVAL_DIR / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nWrote {EVAL_DIR / 'results.json'}")


if __name__ == "__main__":
    sys.exit(main())
