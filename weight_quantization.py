#!/usr/bin/env python3
"""
Quantize Conv weights from FP32 to INT8.

Expected Conv weight layout:
    [out_channels, in_channels, kernel_h, kernel_w]

Supports:
    1. Per-tensor symmetric INT8 quantization
    2. Per-output-channel symmetric INT8 quantization
"""

import argparse
import numpy as np


QMIN = -127
QMAX = 127


def quantize_per_tensor(weight: np.ndarray):
    """
    Symmetric per-tensor INT8 quantization.

    q = round(weight / scale)
    scale = max(abs(weight)) / 127
    """
    max_abs = np.max(np.abs(weight))

    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / QMAX

    q_weight = np.round(weight / scale)
    q_weight = np.clip(q_weight, QMIN, QMAX).astype(np.int8)

    return q_weight, np.float32(scale)


def quantize_per_channel(weight: np.ndarray):
    """
    Symmetric per-output-channel INT8 quantization.

    Conv weight layout:
        [OC, IC, KH, KW]

    Each output channel gets its own scale.
    """
    if weight.ndim != 4:
        raise ValueError(
            f"Expected 4D Conv weight [OC, IC, KH, KW], got shape {weight.shape}"
        )

    # Reduce over IC, KH, KW.
    max_abs = np.max(np.abs(weight), axis=(1, 2, 3))

    scales = max_abs / QMAX
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)

    # Broadcast scales: [OC] -> [OC, 1, 1, 1]
    q_weight = np.round(weight / scales[:, None, None, None])
    q_weight = np.clip(q_weight, QMIN, QMAX).astype(np.int8)

    return q_weight, scales


def dequantize_per_tensor(q_weight: np.ndarray, scale: float):
    return q_weight.astype(np.float32) * scale


def dequantize_per_channel(q_weight: np.ndarray, scales: np.ndarray):
    return q_weight.astype(np.float32) * scales[:, None, None, None]


def main():
    parser = argparse.ArgumentParser(
        description="Quantize FP32 Conv weights to symmetric INT8."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input .npy file containing FP32 Conv weights",
    )

    parser.add_argument(
        "--output",
        default="conv_weight_int8.npz",
        help="Output .npz file",
    )

    parser.add_argument(
        "--mode",
        choices=["per_tensor", "per_channel"],
        default="per_channel",
        help="Quantization mode",
    )

    args = parser.parse_args()

    weight = np.load(args.input).astype(np.float32)

    print("Input shape :", weight.shape)
    print("Input dtype :", weight.dtype)
    print("Weight range:", float(weight.min()), "to", float(weight.max()))

    if args.mode == "per_tensor":
        q_weight, scale = quantize_per_tensor(weight)
        dequant = dequantize_per_tensor(q_weight, scale)

        np.savez(
            args.output,
            weight=q_weight,
            scale=scale,
            zero_point=np.int32(0),
        )

        print("Mode        : per_tensor")
        print("Scale       :", float(scale))

    else:
        q_weight, scales = quantize_per_channel(weight)
        dequant = dequantize_per_channel(q_weight, scales)

        np.savez(
            args.output,
            weight=q_weight,
            scale=scales,
            zero_point=np.zeros(weight.shape[0], dtype=np.int32),
            axis=np.int32(0),
        )

        print("Mode        : per_channel")
        print("Scale shape :", scales.shape)

    abs_error = np.abs(weight - dequant)

    print("Output dtype:", q_weight.dtype)
    print("INT8 range  :", int(q_weight.min()), "to", int(q_weight.max()))
    print("Max error   :", float(abs_error.max()))
    print("Mean error  :", float(abs_error.mean()))
    print("Saved to    :", args.output)


if __name__ == "__main__":
    main()
