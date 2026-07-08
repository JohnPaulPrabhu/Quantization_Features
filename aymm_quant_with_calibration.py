from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


# ============================================================
# EDIT THESE
# ============================================================

INPUT_FOLDER = "calibration_raw_inputs"

RAW_DTYPE = np.float32
# Examples:
# RAW_DTYPE = np.float32
# RAW_DTYPE = np.uint8
# RAW_DTYPE = np.int16

DIVIDE_BY_255 = False
# If raw input values are already [0, 1], keep False.
# If raw input values are [0, 255], set True.

CALIBRATION_METHOD = "minmax"
# Supported:
# "minmax"
# "percentile"
# "histogram_percentile"
# "mean_std"
# "mse"

LOWER_PERCENTILE = 0.1
UPPER_PERCENTILE = 99.9

HISTOGRAM_BINS = 2048

NUM_STD = 3.0

MSE_CANDIDATES = 100

INCLUDE_ZERO = True


# ============================================================
# Data classes
# ============================================================

@dataclass
class CalibrationResult:
    method: str
    rmin: float
    rmax: float
    details: Dict[str, Any]


@dataclass
class QuantParams:
    dtype: str
    qmin: int
    qmax: int
    scale: float
    zero_point: int
    rmin: float
    rmax: float
    calibration_method: str


# ============================================================
# Load raw folder
# ============================================================

def get_raw_files(input_folder: str) -> List[Path]:
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    files = sorted([p for p in input_folder.rglob("*") if p.is_file()])

    if not files:
        raise ValueError(f"No files found in folder: {input_folder}")

    return files


def load_all_raw_values(
    input_folder: str,
    raw_dtype,
    divide_by_255: bool = False,
) -> np.ndarray:
    files = get_raw_files(input_folder)

    all_values = []

    for file_path in files:
        x = np.fromfile(file_path, dtype=raw_dtype).astype(np.float32)

        if divide_by_255:
            x = x / 255.0

        x = x[np.isfinite(x)]

        if x.size > 0:
            all_values.append(x)

    if not all_values:
        raise ValueError("No valid finite values found.")

    values = np.concatenate(all_values, axis=0)

    return values


# ============================================================
# Range helpers
# ============================================================

def ensure_valid_range(
    rmin: float,
    rmax: float,
    include_zero: bool = True,
    eps: float = 1e-12,
) -> Tuple[float, float]:

    rmin = float(rmin)
    rmax = float(rmax)

    if include_zero:
        rmin = min(rmin, 0.0)
        rmax = max(rmax, 0.0)

    if abs(rmax - rmin) < eps:
        rmin -= eps
        rmax += eps

    if rmax < rmin:
        raise ValueError(f"Invalid range: rmin={rmin}, rmax={rmax}")

    return rmin, rmax


# ============================================================
# Calibration methods
# ============================================================

def calibrate_minmax(
    values: np.ndarray,
    include_zero: bool = True,
) -> CalibrationResult:

    observed_min = float(np.min(values))
    observed_max = float(np.max(values))

    rmin, rmax = ensure_valid_range(
        observed_min,
        observed_max,
        include_zero=include_zero,
    )

    return CalibrationResult(
        method="minmax",
        rmin=rmin,
        rmax=rmax,
        details={
            "observed_min": observed_min,
            "observed_max": observed_max,
        },
    )


def calibrate_percentile(
    values: np.ndarray,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
    include_zero: bool = True,
) -> CalibrationResult:

    rmin = float(np.percentile(values, lower_percentile))
    rmax = float(np.percentile(values, upper_percentile))

    rmin, rmax = ensure_valid_range(
        rmin,
        rmax,
        include_zero=include_zero,
    )

    return CalibrationResult(
        method="percentile",
        rmin=rmin,
        rmax=rmax,
        details={
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "observed_min": float(np.min(values)),
            "observed_max": float(np.max(values)),
        },
    )


def calibrate_histogram_percentile(
    values: np.ndarray,
    bins: int = 2048,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
    include_zero: bool = True,
) -> CalibrationResult:

    observed_min = float(np.min(values))
    observed_max = float(np.max(values))

    hist, edges = np.histogram(
        values,
        bins=bins,
        range=(observed_min, observed_max),
    )

    cdf = np.cumsum(hist).astype(np.float64)
    cdf = cdf / cdf[-1]

    lower_target = lower_percentile / 100.0
    upper_target = upper_percentile / 100.0

    lower_idx = int(np.searchsorted(cdf, lower_target))
    upper_idx = int(np.searchsorted(cdf, upper_target))

    lower_idx = int(np.clip(lower_idx, 0, len(edges) - 2))
    upper_idx = int(np.clip(upper_idx, 0, len(edges) - 2))

    rmin = float(edges[lower_idx])
    rmax = float(edges[upper_idx + 1])

    rmin, rmax = ensure_valid_range(
        rmin,
        rmax,
        include_zero=include_zero,
    )

    return CalibrationResult(
        method="histogram_percentile",
        rmin=rmin,
        rmax=rmax,
        details={
            "bins": bins,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "observed_min": observed_min,
            "observed_max": observed_max,
        },
    )


def calibrate_mean_std(
    values: np.ndarray,
    num_std: float = 3.0,
    include_zero: bool = True,
) -> CalibrationResult:

    mean = float(np.mean(values))
    std = float(np.std(values))

    observed_min = float(np.min(values))
    observed_max = float(np.max(values))

    rmin = mean - num_std * std
    rmax = mean + num_std * std

    rmin = max(rmin, observed_min)
    rmax = min(rmax, observed_max)

    rmin, rmax = ensure_valid_range(
        rmin,
        rmax,
        include_zero=include_zero,
    )

    return CalibrationResult(
        method="mean_std",
        rmin=rmin,
        rmax=rmax,
        details={
            "mean": mean,
            "std": std,
            "num_std": num_std,
            "observed_min": observed_min,
            "observed_max": observed_max,
        },
    )


# ============================================================
# MSE calibration
# ============================================================

def get_qrange(dtype: str) -> Tuple[int, int]:
    dtype = dtype.lower()

    if dtype == "int8":
        return -128, 127

    if dtype == "int16":
        return -32768, 32767

    raise ValueError("dtype must be int8 or int16")


def calc_asym_scale_zp_from_range(
    rmin: float,
    rmax: float,
    qmin: int,
    qmax: int,
) -> Tuple[float, int]:

    rmin, rmax = ensure_valid_range(rmin, rmax, include_zero=True)

    scale = (rmax - rmin) / float(qmax - qmin)

    if scale <= 0:
        raise ValueError("Scale must be positive.")

    zero_point = round(qmin - rmin / scale)
    zero_point = int(np.clip(zero_point, qmin, qmax))

    return float(scale), zero_point


def fake_quant_dequant(
    values: np.ndarray,
    rmin: float,
    rmax: float,
    dtype: str,
) -> np.ndarray:

    qmin, qmax = get_qrange(dtype)

    scale, zero_point = calc_asym_scale_zp_from_range(
        rmin,
        rmax,
        qmin,
        qmax,
    )

    q = np.round(values / scale + zero_point)
    q = np.clip(q, qmin, qmax)

    dq = scale * (q - zero_point)

    return dq.astype(np.float32)


def calibrate_mse(
    values: np.ndarray,
    dtype: str,
    num_candidates: int = 100,
    include_zero: bool = True,
) -> CalibrationResult:

    observed_min = float(np.min(values))
    observed_max = float(np.max(values))

    best_mse = float("inf")
    best_rmin = observed_min
    best_rmax = observed_max

    if observed_min >= 0.0:
        rmin = 0.0 if include_zero else observed_min
        candidates = np.linspace(99.0, 100.0, num_candidates)

        for p in candidates:
            rmax = float(np.percentile(values, p))
            rmin_v, rmax_v = ensure_valid_range(rmin, rmax, include_zero)

            dq = fake_quant_dequant(values, rmin_v, rmax_v, dtype)
            mse = float(np.mean((values - dq) ** 2))

            if mse < best_mse:
                best_mse = mse
                best_rmin = rmin_v
                best_rmax = rmax_v

    elif observed_max <= 0.0:
        rmax = 0.0 if include_zero else observed_max
        candidates = np.linspace(0.0, 1.0, num_candidates)

        for p in candidates:
            rmin = float(np.percentile(values, p))
            rmin_v, rmax_v = ensure_valid_range(rmin, rmax, include_zero)

            dq = fake_quant_dequant(values, rmin_v, rmax_v, dtype)
            mse = float(np.mean((values - dq) ** 2))

            if mse < best_mse:
                best_mse = mse
                best_rmin = rmin_v
                best_rmax = rmax_v

    else:
        candidates = np.linspace(0.0, 1.0, num_candidates)

        for p in candidates:
            rmin = float(np.percentile(values, p))
            rmax = float(np.percentile(values, 100.0 - p))
            rmin_v, rmax_v = ensure_valid_range(rmin, rmax, include_zero)

            dq = fake_quant_dequant(values, rmin_v, rmax_v, dtype)
            mse = float(np.mean((values - dq) ** 2))

            if mse < best_mse:
                best_mse = mse
                best_rmin = rmin_v
                best_rmax = rmax_v

    return CalibrationResult(
        method="mse",
        rmin=best_rmin,
        rmax=best_rmax,
        details={
            "dtype": dtype,
            "best_mse": best_mse,
            "num_candidates": num_candidates,
            "observed_min": observed_min,
            "observed_max": observed_max,
        },
    )


# ============================================================
# Calibration dispatcher
# ============================================================

def calibrate_values(
    values: np.ndarray,
    method: str,
    dtype: str,
) -> CalibrationResult:

    method = method.lower()

    if method == "minmax":
        return calibrate_minmax(
            values,
            include_zero=INCLUDE_ZERO,
        )

    if method == "percentile":
        return calibrate_percentile(
            values,
            lower_percentile=LOWER_PERCENTILE,
            upper_percentile=UPPER_PERCENTILE,
            include_zero=INCLUDE_ZERO,
        )

    if method == "histogram_percentile":
        return calibrate_histogram_percentile(
            values,
            bins=HISTOGRAM_BINS,
            lower_percentile=LOWER_PERCENTILE,
            upper_percentile=UPPER_PERCENTILE,
            include_zero=INCLUDE_ZERO,
        )

    if method == "mean_std":
        return calibrate_mean_std(
            values,
            num_std=NUM_STD,
            include_zero=INCLUDE_ZERO,
        )

    if method == "mse":
        return calibrate_mse(
            values,
            dtype=dtype,
            num_candidates=MSE_CANDIDATES,
            include_zero=INCLUDE_ZERO,
        )

    raise ValueError(
        "Unsupported calibration method. Use: "
        "minmax, percentile, histogram_percentile, mean_std, mse"
    )


# ============================================================
# Quant params
# ============================================================

def calculate_quant_params(
    calibration_result: CalibrationResult,
    dtype: str,
) -> QuantParams:

    qmin, qmax = get_qrange(dtype)

    scale, zero_point = calc_asym_scale_zp_from_range(
        calibration_result.rmin,
        calibration_result.rmax,
        qmin,
        qmax,
    )

    return QuantParams(
        dtype=dtype,
        qmin=qmin,
        qmax=qmax,
        scale=scale,
        zero_point=zero_point,
        rmin=calibration_result.rmin,
        rmax=calibration_result.rmax,
        calibration_method=calibration_result.method,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    values = load_all_raw_values(
        input_folder=INPUT_FOLDER,
        raw_dtype=RAW_DTYPE,
        divide_by_255=DIVIDE_BY_255,
    )

    print("========== Input Stats ==========")
    print(f"Total values : {values.size}")
    print(f"Observed min : {float(np.min(values))}")
    print(f"Observed max : {float(np.max(values))}")
    print(f"Values < 0   : {int(np.sum(values < 0.0))}")
    print(f"Values > 1   : {int(np.sum(values > 1.0))}")

    print("\n========== Calibration Method ==========")
    print(f"Method       : {CALIBRATION_METHOD}")

    calib_int8 = calibrate_values(
        values=values,
        method=CALIBRATION_METHOD,
        dtype="int8",
    )

    params_int8 = calculate_quant_params(
        calibration_result=calib_int8,
        dtype="int8",
    )

    calib_int16 = calibrate_values(
        values=values,
        method=CALIBRATION_METHOD,
        dtype="int16",
    )

    params_int16 = calculate_quant_params(
        calibration_result=calib_int16,
        dtype="int16",
    )

    print("\n========== Calibration Range ==========")
    print(f"rmin         : {calib_int8.rmin}")
    print(f"rmax         : {calib_int8.rmax}")
    print(f"details      : {calib_int8.details}")

    print("\n========== INT8 Asymmetric Params ==========")
    print(f"qmin         : {params_int8.qmin}")
    print(f"qmax         : {params_int8.qmax}")
    print(f"scale        : {params_int8.scale}")
    print(f"zero_point   : {params_int8.zero_point}")

    print("\n========== INT16 Asymmetric Params ==========")
    print(f"qmin         : {params_int16.qmin}")
    print(f"qmax         : {params_int16.qmax}")
    print(f"scale        : {params_int16.scale}")
    print(f"zero_point   : {params_int16.zero_point}")