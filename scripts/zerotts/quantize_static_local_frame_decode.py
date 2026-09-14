#!/usr/bin/env python3
"""Phase 11: static/QDQ quantization of local_frame_decode.onnx specifically.

Conditional phase per spec.md §9 / plan.md Phase 11: only attempted because
evals/zerotts-int8-baseline.md's Phase 10 measurement found dynamic
quantization of this one graph causes severe fixed-input/fixed-randomness
sampling divergence (15/16 codebooks). Uses a small calibration set of real
`global_hidden` activations captured across 8 diverse Vietnamese calibration
sentences (numbers, dates, code-switched content, a phone number) x ~20 AR
steps each = 168 samples (assets/zerotts/phase3_capture/calib_global_hidden.npy,
gitignored) -- the other float inputs (temperatures/top-k/top-p/repetition
penalty/cfg_scale) are held at their fixed production values (they are
scalar constants in production, not data-dependent activations), and the
non-float inputs (forbid_eoa, seen_mask, *_topk as int64) are not affected by
weight/activation quantization ranges.

Usage: python3 quantize_static_local_frame_decode.py <model_dir>
"""
import sys
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_static,
)


class LocalFrameDecodeCalibReader(CalibrationDataReader):
    def __init__(self, hidden_states: np.ndarray, codebook_size: int, num_codebooks: int):
        self._hidden = hidden_states
        self._i = 0
        self._codebook_size = codebook_size
        self._num_codebooks = num_codebooks

    def get_next(self):
        if self._i >= len(self._hidden):
            return None
        h = self._hidden[self._i : self._i + 1]
        self._i += 1
        rng = np.random.default_rng(self._i)
        feed = {
            "global_hidden": h.astype(np.float32),
            "forbid_eoa": np.array([False]),
            "text_temperature": np.array([1.0], dtype=np.float32),
            "text_topk": np.array([50], dtype=np.int64),
            "audio_temperature": np.array([0.8], dtype=np.float32),
            "audio_topk": np.array([25], dtype=np.int64),
            "audio_topp": np.array([0.95], dtype=np.float32),
            "audio_repetition_penalty": np.array([1.2], dtype=np.float32),
            "seen_mask": np.zeros((1, self._num_codebooks, self._codebook_size), dtype=bool),
            "ctrl_random_u": rng.random(1).astype(np.float32),
            "audio_random_u": rng.random((1, self._num_codebooks)).astype(np.float32),
            "cfg_scale": np.array([1.0], dtype=np.float32),
        }
        return feed


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    model_dir = Path(sys.argv[1])
    calib_path = (
        model_dir.parent / "phase3_capture" / "calib_global_hidden.npy"
        if (model_dir.parent / "phase3_capture").exists()
        else Path("assets/zerotts/phase3_capture/calib_global_hidden.npy")
    )
    if not calib_path.exists():
        print(f"Calibration data not found: {calib_path}")
        return 1

    hidden = np.load(calib_path)
    print(f"Loaded {hidden.shape[0]} calibration samples from {calib_path}")

    src = model_dir / "onnx" / "local_frame_decode.onnx"
    dst = model_dir / "onnx" / "local_frame_decode.qdq.onnx"

    reader = LocalFrameDecodeCalibReader(hidden, codebook_size=1024, num_codebooks=16)
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=None,  # default QDQ
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"Wrote {dst} ({dst.stat().st_size/1e6:.1f}MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
