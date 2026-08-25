import numpy as np
from dataclasses import dataclass
from typing import Iterable, Literal


CalibrationMethod = Literal["minmax", "percentile", "mse"]


@dataclass
class QuantizationParams:
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    bits: int
    symmetric: bool
    method: str


def get_quant_range(bits: int):
    """
    Signed integer quantization range.

    INT8  : [-128, 127]
    INT16 : [-32768, 32767]
    """
    if bits not in (8, 16):
        raise ValueError("Only INT8 and INT16 are supported.")

    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    return qmin, qmax


def collect_calibration_values(
    calibration_dataset: Iterable[np.ndarray]
) -> np.ndarray:
    """
    Flatten and concatenate calibration tensors.

    NOTE:
    For very large calibration datasets you should use
    histogram/streaming calibration instead of concatenating everything.
    """

    values = []

    for sample in calibration_dataset:
        sample = np.asarray(sample, dtype=np.float32)

        # Remove NaN / Inf
        sample = sample[np.isfinite(sample)]

        if sample.size > 0:
            values.append(sample.reshape(-1))

    if not values:
        raise ValueError("Calibration dataset is empty.")

    return np.concatenate(values)


# ============================================================
# Calibration methods
# ============================================================

def minmax_range(values, symmetric):
    """
    Standard min-max calibration.
    """

    if symmetric:
        abs_max = np.max(np.abs(values))
        return -abs_max, abs_max

    return float(np.min(values)), float(np.max(values))


def percentile_range(
    values,
    symmetric,
    percentile=99.99
):
    """
    Percentile calibration.

    Symmetric:
        threshold = percentile(abs(x))

    Asymmetric:
        clip both low and high tails.
    """

    if symmetric:
        threshold = np.percentile(
            np.abs(values),
            percentile
        )

        return -float(threshold), float(threshold)

    tail = (100.0 - percentile) / 2.0

    rmin = np.percentile(values, tail)
    rmax = np.percentile(values, 100.0 - tail)

    return float(rmin), float(rmax)


def mse_range(
    values,
    bits,
    symmetric,
    num_candidates=100
):
    """
    Simple MinMSE clipping search.

    Tries different clipping thresholds and selects the range
    producing the smallest quantization/dequantization MSE.
    """

    qmin, qmax = get_quant_range(bits)

    original_min = float(np.min(values))
    original_max = float(np.max(values))

    if symmetric:
        abs_max = max(abs(original_min), abs(original_max))

        candidates = np.linspace(
            abs_max * 0.5,
            abs_max,
            num_candidates
        )

        best_mse = float("inf")
        best_range = (-abs_max, abs_max)

        for threshold in candidates:

            rmin = -threshold
            rmax = threshold

            # For symmetric signed quantization, use qmax magnitude.
            scale = threshold / qmax

            if scale <= 0:
                continue

            q = np.round(values / scale)

            q = np.clip(
                q,
                -qmax,
                qmax
            )

            dq = q * scale

            error = np.mean(
                (values - dq) ** 2
            )

            if error < best_mse:
                best_mse = error
                best_range = (
                    float(rmin),
                    float(rmax)
                )

        return best_range

    else:

        # Scale both sides inward and search.
        ratios = np.linspace(
            0.5,
            1.0,
            num_candidates
        )

        best_mse = float("inf")
        best_range = (
            original_min,
            original_max
        )

        for ratio in ratios:

            rmin = original_min * ratio
            rmax = original_max * ratio

            if rmin >= rmax:
                continue

            scale = (rmax - rmin) / (qmax - qmin)

            if scale <= 0:
                continue

            zero_point = np.round(
                qmin - rmin / scale
            )

            zero_point = np.clip(
                zero_point,
                qmin,
                qmax
            )

            q = np.round(
                values / scale + zero_point
            )

            q = np.clip(
                q,
                qmin,
                qmax
            )

            dq = (
                q - zero_point
            ) * scale

            error = np.mean(
                (values - dq) ** 2
            )

            if error < best_mse:
                best_mse = error
                best_range = (
                    float(rmin),
                    float(rmax)
                )

        return best_range


# ============================================================
# Calculate Scale / Zero Point
# ============================================================

def calculate_quant_params(
    calibration_dataset,
    bits=8,
    symmetric=True,
    method="minmax",
    percentile=99.99,
):
    """
    Generate quantization parameters from calibration data.
    """

    qmin, qmax = get_quant_range(bits)

    values = collect_calibration_values(
        calibration_dataset
    )

    # -------------------------
    # Calibration
    # -------------------------

    if method == "minmax":

        rmin, rmax = minmax_range(
            values,
            symmetric
        )

    elif method == "percentile":

        rmin, rmax = percentile_range(
            values,
            symmetric,
            percentile
        )

    elif method == "mse":

        rmin, rmax = mse_range(
            values,
            bits,
            symmetric
        )

    else:
        raise ValueError(
            f"Unsupported calibration method: {method}"
        )

    # Avoid zero range
    if rmin == rmax:
        rmin -= 1e-8
        rmax += 1e-8

    # ========================================================
    # Symmetric quantization
    # ========================================================

    if symmetric:

        max_abs = max(
            abs(rmin),
            abs(rmax)
        )

        # Use positive magnitude 127 / 32767.
        scale = max_abs / qmax

        zero_point = 0

    # ========================================================
    # Asymmetric quantization
    # ========================================================

    else:

        scale = (
            rmax - rmin
        ) / (
            qmax - qmin
        )

        zero_point = round(
            qmin - rmin / scale
        )

        zero_point = int(
            np.clip(
                zero_point,
                qmin,
                qmax
            )
        )

    return QuantizationParams(
        scale=float(scale),
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        bits=bits,
        symmetric=symmetric,
        method=method
    )


# ============================================================
# Quantization
# ============================================================

def quantize_input(
    input_data,
    params: QuantizationParams
):
    """
    FP32 -> INT8 / INT16
    """

    x = np.asarray(
        input_data,
        dtype=np.float32
    )

    q = np.round(
        x / params.scale
        + params.zero_point
    )

    q = np.clip(
        q,
        params.qmin,
        params.qmax
    )

    if params.bits == 8:
        return q.astype(np.int8)

    elif params.bits == 16:
        return q.astype(np.int16)

    raise ValueError(
        "Unsupported bit width."
    )


# ============================================================
# Dequantization
# ============================================================

def dequantize_input(
    quantized_data,
    params: QuantizationParams
):
    """
    INT8 / INT16 -> FP32
    """

    q = np.asarray(
        quantized_data,
        dtype=np.float32
    )

    return (
        q - params.zero_point
    ) * params.scale