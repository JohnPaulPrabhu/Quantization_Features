import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union


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
# Helper functions
# ============================================================

def get_signed_qrange(dtype: str) -> Tuple[int, int, np.dtype]:
    """
    Returns qmin, qmax, numpy dtype for signed int8/int16.
    """
    dtype = dtype.lower()

    if dtype == "int8":
        return -128, 127, np.int8

    if dtype == "int16":
        return -32768, 32767, np.int16

    raise ValueError("Only 'int8' and 'int16' are supported.")


def collect_calibration_values(
    calibration_data: Union[np.ndarray, Iterable[np.ndarray]],
    remove_nan_inf: bool = True,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Converts calibration data into one flat float64 array.

    calibration_data can be:
    - one numpy array
    - list/tuple/generator of numpy arrays
    """

    if isinstance(calibration_data, np.ndarray):
        x = calibration_data.astype(np.float64, copy=False).reshape(-1)
    else:
        arrays = []
        for item in calibration_data:
            arr = np.asarray(item, dtype=np.float64).reshape(-1)
            arrays.append(arr)

        if len(arrays) == 0:
            raise ValueError("Calibration data is empty.")

        x = np.concatenate(arrays, axis=0)

    if remove_nan_inf:
        x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("Calibration data has no valid finite values.")

    if max_samples is not None and x.size > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(x.size, size=max_samples, replace=False)
        x = x[indices]

    return x


def ensure_valid_range(
    rmin: float,
    rmax: float,
    include_zero: bool = True,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Ensures calibration range is valid.
    Also optionally includes zero inside the range.
    """

    rmin = float(rmin)
    rmax = float(rmax)

    if include_zero:
        rmin = min(rmin, 0.0)
        rmax = max(rmax, 0.0)

    if rmax < rmin:
        raise ValueError(f"Invalid range: rmin={rmin}, rmax={rmax}")

    if abs(rmax - rmin) < eps:
        rmin -= eps
        rmax += eps

    return rmin, rmax


# ============================================================
# Calibration techniques
# ============================================================

def calibrate_minmax(
    x: np.ndarray,
    include_zero: bool = True,
) -> CalibrationResult:
    """
    Uses true minimum and maximum from calibration data.
    Most accurate range coverage, but sensitive to outliers.
    """

    rmin = float(np.min(x))
    rmax = float(np.max(x))

    rmin, rmax = ensure_valid_range(rmin, rmax, include_zero)

    return CalibrationResult(
        method="minmax",
        rmin=rmin,
        rmax=rmax,
        details={
            "observed_min": float(np.min(x)),
            "observed_max": float(np.max(x)),
            "include_zero": include_zero,
        },
    )


def calibrate_percentile(
    x: np.ndarray,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
    include_zero: bool = True,
) -> CalibrationResult:
    """
    Clips extreme outliers using percentiles.

    Example:
    lower_percentile=0.1, upper_percentile=99.9
    means keep the central 99.8% of values.
    """

    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    rmin = float(np.percentile(x, lower_percentile))
    rmax = float(np.percentile(x, upper_percentile))

    rmin, rmax = ensure_valid_range(rmin, rmax, include_zero)

    return CalibrationResult(
        method="percentile",
        rmin=rmin,
        rmax=rmax,
        details={
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "include_zero": include_zero,
        },
    )


def calibrate_histogram_percentile(
    x: np.ndarray,
    bins: int = 2048,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
    include_zero: bool = True,
) -> CalibrationResult:
    """
    Histogram-based percentile calibration.

    This is useful when data is very large and you want a histogram-based
    approximation instead of directly calculating percentiles.
    """

    observed_min = float(np.min(x))
    observed_max = float(np.max(x))

    observed_min, observed_max = ensure_valid_range(
        observed_min,
        observed_max,
        include_zero=False,
    )

    hist, edges = np.histogram(
        x,
        bins=bins,
        range=(observed_min, observed_max),
    )

    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1]

    lower_target = lower_percentile / 100.0
    upper_target = upper_percentile / 100.0

    lower_idx = int(np.searchsorted(cdf, lower_target))
    upper_idx = int(np.searchsorted(cdf, upper_target))

    lower_idx = np.clip(lower_idx, 0, len(edges) - 2)
    upper_idx = np.clip(upper_idx, 0, len(edges) - 2)

    rmin = float(edges[lower_idx])
    rmax = float(edges[upper_idx + 1])

    rmin, rmax = ensure_valid_range(rmin, rmax, include_zero)

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
            "include_zero": include_zero,
        },
    )


def calibrate_mean_std(
    x: np.ndarray,
    num_std: float = 3.0,
    include_zero: bool = True,
) -> CalibrationResult:
    """
    Uses mean +/- N standard deviations.

    Useful when the data is roughly Gaussian-like.
    Not always ideal for image inputs, but useful for activations.
    """

    mean = float(np.mean(x))
    std = float(np.std(x))

    rmin = mean - num_std * std
    rmax = mean + num_std * std

    observed_min = float(np.min(x))
    observed_max = float(np.max(x))

    # Do not exceed true observed range.
    rmin = max(rmin, observed_min)
    rmax = min(rmax, observed_max)

    rmin, rmax = ensure_valid_range(rmin, rmax, include_zero)

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
            "include_zero": include_zero,
        },
    )


# ============================================================
# Quantization simulation used by MSE calibration
# ============================================================

def calculate_asymmetric_qparams_from_range(
    rmin: float,
    rmax: float,
    dtype: str,
    calibration_method: str = "manual",
) -> QuantParams:
    """
    Calculates asymmetric quantization parameters.

    Formula:
        scale = (rmax - rmin) / (qmax - qmin)
        zero_point = round(qmin - rmin / scale)

    Quantization:
        q = round(x / scale + zero_point)

    Dequantization:
        x_float = scale * (q - zero_point)
    """

    qmin, qmax, _ = get_signed_qrange(dtype)

    rmin, rmax = ensure_valid_range(rmin, rmax, include_zero=True)

    scale = (rmax - rmin) / float(qmax - qmin)

    if scale <= 0.0:
        raise ValueError("Scale must be positive.")

    zero_point_fp = qmin - (rmin / scale)
    zero_point = int(np.round(zero_point_fp))
    zero_point = int(np.clip(zero_point, qmin, qmax))

    return QuantParams(
        dtype=dtype,
        qmin=qmin,
        qmax=qmax,
        scale=float(scale),
        zero_point=zero_point,
        rmin=float(rmin),
        rmax=float(rmax),
        calibration_method=calibration_method,
    )


def fake_quant_dequant_float(
    x: np.ndarray,
    rmin: float,
    rmax: float,
    dtype: str,
) -> np.ndarray:
    """
    Simulates quantization + dequantization in float.
    Used to measure MSE without permanently casting to int8/int16.
    """

    params = calculate_asymmetric_qparams_from_range(
        rmin=rmin,
        rmax=rmax,
        dtype=dtype,
        calibration_method="mse_candidate",
    )

    q = np.round(x / params.scale + params.zero_point)
    q = np.clip(q, params.qmin, params.qmax)

    dq = params.scale * (q - params.zero_point)
    return dq


def calibrate_mse(
    x: np.ndarray,
    dtype: str,
    num_candidates: int = 100,
    include_zero: bool = True,
    max_samples: Optional[int] = 500_000,
    seed: int = 0,
) -> CalibrationResult:
    """
    Searches for the clipping range that gives minimum reconstruction MSE
    after fake quantization.

    This is useful when minmax range is affected by outliers.

    For mostly positive data like normalized images [0, 1], this usually
    searches rmin=0 and different rmax clipping thresholds.
    """

    if max_samples is not None and x.size > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(x.size, size=max_samples, replace=False)
        x_eval = x[indices]
    else:
        x_eval = x

    observed_min = float(np.min(x_eval))
    observed_max = float(np.max(x_eval))

    best_mse = np.inf
    best_rmin = observed_min
    best_rmax = observed_max

    # Case 1: mostly non-negative data, common for image input [0, 1]
    if observed_min >= 0.0:
        rmin_candidates = [0.0 if include_zero else observed_min]

        # Search high clipping thresholds between 99% and 100%.
        upper_percentiles = np.linspace(99.0, 100.0, num_candidates)

        for rmin in rmin_candidates:
            for p_high in upper_percentiles:
                rmax = float(np.percentile(x_eval, p_high))
                rmin_v, rmax_v = ensure_valid_range(rmin, rmax, include_zero)

                dq = fake_quant_dequant_float(x_eval, rmin_v, rmax_v, dtype)
                mse = float(np.mean((x_eval - dq) ** 2))

                if mse < best_mse:
                    best_mse = mse
                    best_rmin = rmin_v
                    best_rmax = rmax_v

    # Case 2: mostly non-positive data
    elif observed_max <= 0.0:
        rmax_candidates = [0.0 if include_zero else observed_max]

        lower_percentiles = np.linspace(0.0, 1.0, num_candidates)

        for rmax in rmax_candidates:
            for p_low in lower_percentiles:
                rmin = float(np.percentile(x_eval, p_low))
                rmin_v, rmax_v = ensure_valid_range(rmin, rmax, include_zero)

                dq = fake_quant_dequant_float(x_eval, rmin_v, rmax_v, dtype)
                mse = float(np.mean((x_eval - dq) ** 2))

                if mse < best_mse:
                    best_mse = mse
                    best_rmin = rmin_v
                    best_rmax = rmax_v

    # Case 3: mixed positive and negative data
    else:
        tail_percentiles = np.linspace(0.0, 1.0, num_candidates)

        for p_tail in tail_percentiles:
            rmin = float(np.percentile(x_eval, p_tail))
            rmax = float(np.percentile(x_eval, 100.0 - p_tail))

            rmin_v, rmax_v = ensure_valid_range(rmin, rmax, include_zero)

            dq = fake_quant_dequant_float(x_eval, rmin_v, rmax_v, dtype)
            mse = float(np.mean((x_eval - dq) ** 2))

            if mse < best_mse:
                best_mse = mse
                best_rmin = rmin_v
                best_rmax = rmax_v

    best_rmin, best_rmax = ensure_valid_range(best_rmin, best_rmax, include_zero)

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
            "include_zero": include_zero,
        },
    )


# ============================================================
# Main calibration dispatcher
# ============================================================

def calibrate_range(
    calibration_data: Union[np.ndarray, Iterable[np.ndarray]],
    method: str = "minmax",
    dtype: str = "int8",
    include_zero: bool = True,
    max_samples: Optional[int] = None,
    **kwargs,
) -> CalibrationResult:
    """
    Main calibration function.

    Supported methods:
    - minmax
    - percentile
    - histogram_percentile
    - mean_std
    - mse
    """

    x = collect_calibration_values(
        calibration_data,
        remove_nan_inf=True,
        max_samples=max_samples,
    )

    method = method.lower()

    if method == "minmax":
        return calibrate_minmax(
            x,
            include_zero=include_zero,
        )

    if method == "percentile":
        return calibrate_percentile(
            x,
            lower_percentile=kwargs.get("lower_percentile", 0.1),
            upper_percentile=kwargs.get("upper_percentile", 99.9),
            include_zero=include_zero,
        )

    if method == "histogram_percentile":
        return calibrate_histogram_percentile(
            x,
            bins=kwargs.get("bins", 2048),
            lower_percentile=kwargs.get("lower_percentile", 0.1),
            upper_percentile=kwargs.get("upper_percentile", 99.9),
            include_zero=include_zero,
        )

    if method == "mean_std":
        return calibrate_mean_std(
            x,
            num_std=kwargs.get("num_std", 3.0),
            include_zero=include_zero,
        )

    if method == "mse":
        return calibrate_mse(
            x,
            dtype=dtype,
            num_candidates=kwargs.get("num_candidates", 100),
            include_zero=include_zero,
            max_samples=kwargs.get("mse_max_samples", 500_000),
            seed=kwargs.get("seed", 0),
        )

    raise ValueError(
        "Unsupported calibration method. Use one of: "
        "'minmax', 'percentile', 'histogram_percentile', 'mean_std', 'mse'"
    )


# ============================================================
# Asymmetric quantization / dequantization
# ============================================================

def calculate_asymmetric_qparams(
    calibration_result: CalibrationResult,
    dtype: str,
) -> QuantParams:
    """
    Calculates asymmetric quantization parameters from calibration result.
    """

    return calculate_asymmetric_qparams_from_range(
        rmin=calibration_result.rmin,
        rmax=calibration_result.rmax,
        dtype=dtype,
        calibration_method=calibration_result.method,
    )


def overflow_report_before_cast(
    x: np.ndarray,
    params: QuantParams,
) -> Dict[str, Any]:
    """
    Checks overflow before casting to int8/int16.

    Important:
    Do this before astype(np.int8) or astype(np.int16),
    because direct casting can wrap values.
    """

    x_float = np.asarray(x, dtype=np.float64)

    q_raw = np.round(x_float / params.scale + params.zero_point)

    overflow_low = int(np.sum(q_raw < params.qmin))
    overflow_high = int(np.sum(q_raw > params.qmax))

    total = int(q_raw.size)

    return {
        "dtype": params.dtype,
        "qmin": params.qmin,
        "qmax": params.qmax,
        "scale": params.scale,
        "zero_point": params.zero_point,
        "raw_q_min": float(np.min(q_raw)),
        "raw_q_max": float(np.max(q_raw)),
        "overflow_low_count": overflow_low,
        "overflow_high_count": overflow_high,
        "total_values": total,
        "overflow_total": overflow_low + overflow_high,
        "overflow_percent": 100.0 * (overflow_low + overflow_high) / total,
    }


def quantize_asymmetric(
    x: np.ndarray,
    params: QuantParams,
    clip: bool = True,
) -> np.ndarray:
    """
    Quantizes float input to signed int8/int16 using asymmetric quantization.

    Formula:
        q = round(x / scale + zero_point)

    Important:
    Always clip before casting.
    """

    _, _, np_dtype = get_signed_qrange(params.dtype)

    x_float = np.asarray(x, dtype=np.float64)

    q = np.round(x_float / params.scale + params.zero_point)

    if clip:
        q = np.clip(q, params.qmin, params.qmax)
    else:
        report = overflow_report_before_cast(x, params)
        if report["overflow_total"] > 0:
            raise OverflowError(
                f"Overflow detected before casting: {report}"
            )

    return q.astype(np_dtype)


def dequantize_asymmetric(
    q: np.ndarray,
    params: QuantParams,
) -> np.ndarray:
    """
    Dequantizes int8/int16 tensor back to float.

    Formula:
        x = scale * (q - zero_point)
    """

    q_float = np.asarray(q, dtype=np.float64)
    x = params.scale * (q_float - params.zero_point)

    return x.astype(np.float32)


def quantize_with_calibration(
    input_data: np.ndarray,
    calibration_data: Union[np.ndarray, Iterable[np.ndarray]],
    dtype: str = "int8",
    calibration_method: str = "minmax",
    include_zero: bool = True,
    **calibration_kwargs,
) -> Tuple[np.ndarray, QuantParams, CalibrationResult, Dict[str, Any]]:
    """
    Full pipeline:
    1. Calibrate range using calibration data
    2. Calculate asymmetric qparams
    3. Check overflow before casting
    4. Quantize input data
    """

    calibration_result = calibrate_range(
        calibration_data=calibration_data,
        method=calibration_method,
        dtype=dtype,
        include_zero=include_zero,
        **calibration_kwargs,
    )

    params = calculate_asymmetric_qparams(
        calibration_result=calibration_result,
        dtype=dtype,
    )

    overflow_report = overflow_report_before_cast(
        input_data,
        params,
    )

    q = quantize_asymmetric(
        input_data,
        params,
        clip=True,
    )

    return q, params, calibration_result, overflow_report


# ============================================================
# Debug / validation utilities
# ============================================================

def print_quant_summary(
    x: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    params: QuantParams,
    calibration_result: CalibrationResult,
    overflow_report: Dict[str, Any],
) -> None:
    """
    Prints useful debugging information.
    """

    abs_error = np.abs(x.astype(np.float32) - dq)
    mse = float(np.mean((x.astype(np.float32) - dq) ** 2))

    print("\n========== Calibration Summary ==========")
    print(f"Calibration method : {calibration_result.method}")
    print(f"Calibration rmin   : {calibration_result.rmin}")
    print(f"Calibration rmax   : {calibration_result.rmax}")
    print(f"Details            : {calibration_result.details}")

    print("\n========== Quantization Params ==========")
    print(f"dtype       : {params.dtype}")
    print(f"qmin        : {params.qmin}")
    print(f"qmax        : {params.qmax}")
    print(f"scale       : {params.scale}")
    print(f"zero_point  : {params.zero_point}")

    print("\n========== Input Stats ==========")
    print(f"input min   : {float(np.min(x))}")
    print(f"input max   : {float(np.max(x))}")

    print("\n========== Raw Quant Overflow Check ==========")
    for k, v in overflow_report.items():
        print(f"{k}: {v}")

    print("\n========== Quantized Tensor Stats ==========")
    print(f"q dtype     : {q.dtype}")
    print(f"q min       : {int(np.min(q))}")
    print(f"q max       : {int(np.max(q))}")

    print("\n========== Dequant Error ==========")
    print(f"dequant min : {float(np.min(dq))}")
    print(f"dequant max : {float(np.max(dq))}")
    print(f"max abs err : {float(np.max(abs_error))}")
    print(f"mean abs err: {float(np.mean(abs_error))}")
    print(f"mse         : {mse}")


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Example calibration data
    # Replace this with your real calibration images.
    #
    # Example shape:
    # calibration_data = [img1, img2, img3, ...]
    # Each image can be NHWC or NCHW.
    # Values are assumed to be normalized between 0 and 1.
    # ------------------------------------------------------------

    np.random.seed(0)

    calibration_data = np.random.rand(20, 224, 224, 3).astype(np.float32)

    # Example input to quantize.
    # Usually this can be one image or a batch.
    input_data = calibration_data[:1]

    # ------------------------------------------------------------
    # Choose calibration method
    # ------------------------------------------------------------

    calibration_method = "minmax"
    # calibration_method = "percentile"
    # calibration_method = "histogram_percentile"
    # calibration_method = "mean_std"
    # calibration_method = "mse"

    # ------------------------------------------------------------
    # INT8 asymmetric quantization
    # ------------------------------------------------------------

    q_int8, params_int8, calib_int8, report_int8 = quantize_with_calibration(
        input_data=input_data,
        calibration_data=calibration_data,
        dtype="int8",
        calibration_method=calibration_method,
        include_zero=True,

        # Used only by percentile / histogram_percentile
        lower_percentile=0.1,
        upper_percentile=99.9,

        # Used only by histogram_percentile
        bins=2048,

        # Used only by mean_std
        num_std=3.0,

        # Used only by mse
        num_candidates=100,
        mse_max_samples=500_000,
    )

    dq_int8 = dequantize_asymmetric(q_int8, params_int8)

    print("\n\n================ INT8 RESULT ================")
    print_quant_summary(
        x=input_data,
        q=q_int8,
        dq=dq_int8,
        params=params_int8,
        calibration_result=calib_int8,
        overflow_report=report_int8,
    )

    # ------------------------------------------------------------
    # INT16 asymmetric quantization
    # ------------------------------------------------------------

    q_int16, params_int16, calib_int16, report_int16 = quantize_with_calibration(
        input_data=input_data,
        calibration_data=calibration_data,
        dtype="int16",
        calibration_method=calibration_method,
        include_zero=True,

        lower_percentile=0.1,
        upper_percentile=99.9,
        bins=2048,
        num_std=3.0,
        num_candidates=100,
        mse_max_samples=500_000,
    )

    dq_int16 = dequantize_asymmetric(q_int16, params_int16)

    print("\n\n================ INT16 RESULT ================")
    print_quant_summary(
        x=input_data,
        q=q_int16,
        dq=dq_int16,
        params=params_int16,
        calibration_result=calib_int16,
        overflow_report=report_int16,
    )