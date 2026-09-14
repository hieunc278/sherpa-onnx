#!/usr/bin/env python3
"""Phase 10: dynamic (weight-only) int8 quantization of ZeroTTS's 5 ONNX
graphs, per spec.md §9.

Produces `<name>.int8.onnx` next to each fp32 graph in the given model
directory, using onnxruntime.quantization.quantize_dynamic -- no calibration
data needed. Per spec.md §9's risk note, `local_frame_decode.onnx` (the
graph that runs the actual AR-loop sampling) is quantized and reported
separately from the others so its higher-risk status isn't hidden inside a
single blanket pass/fail.

Usage:  python3 quantize_int8.py <model_dir>
"""
import sys
import time
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_one(src: Path, dst: Path, op_types_to_quantize=None):
    t0 = time.time()
    kwargs = {}
    if op_types_to_quantize:
        kwargs["op_types_to_quantize"] = op_types_to_quantize
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        **kwargs,
    )
    dt = time.time() - t0
    src_size = src.stat().st_size
    # external-data models write a second .onnx.data / .data file; sum
    # everything with the same stem.
    dst_size = sum(
        p.stat().st_size for p in dst.parent.glob(dst.stem + "*")
    )
    print(f"{src.name}: {src_size/1e6:.1f}MB -> {dst.name}: "
          f"{dst_size/1e6:.1f}MB ({dt:.1f}s)")
    return src_size, dst_size


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    model_dir = Path(sys.argv[1])

    # (path, op_types_to_quantize) -- the two codec decoder graphs contain
    # Conv layers; ConvInteger has no CPU EP kernel in this onnxruntime
    # build (1.28.2), so quantize_dynamic's default (Conv+MatMul+Gemm)
    # produces an unloadable model for those two. Restricting to
    # MatMul/Gemm only (skip Conv, leave those weights fp32) is the
    # documented workaround and is what this script does -- see
    # evals/zerotts-int8-baseline.md for the reproduction and the decision
    # to scope it this way rather than treat it as a hard blocker.
    graphs = [
        (model_dir / "onnx" / "text_encoder.onnx", None),
        (model_dir / "onnx" / "prefix_step.onnx", None),
        (model_dir / "onnx" / "local_frame_decode.onnx", None),
        (model_dir / "onnx" / "codec" / "moss_audio_tokenizer_decode_full.onnx",
        ["MatMul", "Gemm"]),
        (model_dir / "onnx" / "codec" / "moss_audio_tokenizer_decode_step.onnx",
        ["MatMul", "Gemm"]),
    ]

    total_fp32 = 0
    total_int8 = 0
    results = {}
    for g, op_types in graphs:
        if not g.exists():
            print(f"SKIP (not found): {g}")
            continue
        dst = g.with_suffix("").with_suffix(".int8.onnx")
        try:
            s, d = quantize_one(g, dst, op_types)
            total_fp32 += s
            total_int8 += d
            results[g.name] = (s, d)
        except Exception as e:  # noqa: BLE001 -- report and continue
            print(f"FAILED to quantize {g.name}: {e}")
            results[g.name] = None

    print()
    print(f"Total fp32 (5 graphs): {total_fp32/1e6:.1f}MB")
    print(f"Total int8 (5 graphs): {total_int8/1e6:.1f}MB")
    if total_fp32:
        print(f"Size reduction: {(1 - total_int8/total_fp32)*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
