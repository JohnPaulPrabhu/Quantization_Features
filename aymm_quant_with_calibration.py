import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
from PIL import Image


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
# Supported input files
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".npy"
}


def get_input_files(input_folder: str) -> List[Path]:
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    files = []

    for path in input_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    files = sorted(files)

    if len(files) == 0:
        raise ValueError(f"No supported input files found in: {input_folder}")

    return files


# ============================================================
# Input loading
# ============================================================

def load_input_file(
    file_path: Path,
    image_mode: str = "RGB",
    resize_hw: Optional[Tuple[int, int]] = None,
    layout: str = "NHWC",
    add_batch_dim: bool = True,
) -> np.ndarray:
    """
    Loads one input file.

    image_mode:
        "RGB" for 3-channel image
        "L" for grayscale

    resize_hw:
        None means do not resize.
        (height, width) means resize image to that size.

    layout:
        "NHWC" gives shape [1, H, W, C]
        "NCHW" gives shape [1, C, H, W]

    add_batch_dim:
        True adds batch dimension.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        x = np.load(file_path).astype(np.float32)

    else:
        img = Image.open(file_path).convert(image_mode)

        if resize_hw is not None:
            height, width = resize_hw
            img = img.resize((width, height), Image.BILINEAR)

        x = np.asarray(img).astype(np.float32)

        # Normalize image input to [0, 1]
        x = x / 255.0

        # For grayscale image, PIL gives shape [H, W].
        # Convert to [H, W, 1].
        if x.ndim == 2:
            x = x[..., None]

    layout = layout.upper()

    if layout not in ["NHWC", "NCHW"]:
        raise ValueError("layout must be either 'NHWC' or 'NCHW'")

    # If input is image-like [H, W, C], convert layout if needed.
    if x.ndim == 3:
        if layout == "NCHW":
            x = np.transpose(x, (2, 0, 1))

    # Add batch dimension.
    if add_batch_dim and x.ndim in [3, 4]:
        # If already batched .npy, do not add again.
        if not (suffix == ".npy" and x.ndim == 4):
            x = np.expand_dims(x, axis=0)

    return x.astype(np.float32)


# ============================================================
# Quantization range helpers
# ============================================================

def get_signed_qrange(dtype: str) -> Tuple[int, int, np.dtype]:
    dtype = dtype.lower()

    if dtype == "int8":
        return -128, 127, np.int8

    if dtype == "int16":
        return -32768, 32767, np.int16

    raise ValueError("Only int8 and int16 are supported.")


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
# Collect values from folder for calibration
# ============================================================

def collect_values_from_folder(
    input_folder: str,
    image_mode: str = "RGB",
    resize_hw: Optional[Tuple[int, int]] = None,
    layout: str = "NHWC",
    max_values: Optional[int] = 2_000_000,
    seed: int = 0,
) -> np.ndarray:
    """
    Loads values from all files in a folder and returns one flat array.

    max_values:
        Limits how many values are used for calibration.
        This avoids memory explosion for large folders.

    For accurate minmax, you can set max_values=None.
    """

    files = get_input_files(input_folder)
    rng = np.random.default_rng(seed)

    collected = []

    for file_path in files:
        x = load_input_file(
            file_path=file_path,
            image_mode=image_mode,
            resize_hw=resize_hw,
            layout=layout,
            add_batch_dim=True,
        )

        values = x.reshape(-1)
        values = values[np.isfinite(values)]

        if values.size == 0:
            continue

        collected.append(values)

    if len(collected) == 0:
        raise ValueError("No valid finite calibration values found.")

    all_values = np.concatenate(collected, axis=0).astype(np.float32)

    if max_values is not None and all_values.size > max_values:
        indices = rng.choice(
            all_values.size,
            size=max_values,
            replace=False,
        )
        all_values = all_values[indices]

    return all_values


# ============================================================
# Calibration techniques
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

    observed_min, observed_max = ensure_valid_range(
        observed_min,
        observed_max,
        include_zero=False,
    )

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

    lower_idx = np.clip(lower_idx, 0, len(edges) - 2)
    upper_idx = np.clip(upper_idx, 0, len(edges) - 2)

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
            "observed_min": float(np.min(values)),
            "observed_max": float(np.max(values)),
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


def calibrate_folder_range(
    input_folder: str,
    method: str = "minmax",
    image_mode: str = "RGB",
    resize_hw: Optional[Tuple[int, int]] = None,
    layout: str = "NHWC",
    include_zero: bool = True,
    max_values: Optional[int] = 2_000_000,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
    bins: int = 2048,
    num_std: float = 3.0,
) -> CalibrationResult:

    values = collect_values_from_folder(
        input_folder=input_folder,
        image_mode=image_mode,
        resize_hw=resize_hw,
        layout=layout,
        max_values=max_values,
    )

    method = method.lower()

    print("========== Calibration Input Stats ==========")
    print(f"Total sampled values : {values.size}")
    print(f"Observed min         : {float(np.min(values))}")
    print(f"Observed max         : {float(np.max(values))}")
    print(f"Values < 0           : {int(np.sum(values < 0.0))}")
    print(f"Values > 1           : {int(np.sum(values > 1.0))}")

    if method == "minmax":
        return calibrate_minmax(values, include_zero)

    if method == "percentile":
        return calibrate_percentile(
            values,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            include_zero=include_zero,
        )

    if method == "histogram_percentile":
        return calibrate_histogram_percentile(
            values,
            bins=bins,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            include_zero=include_zero,
        )

    if method == "mean_std":
        return calibrate_mean_std(
            values,
            num_std=num_std,
            include_zero=include_zero,
        )

    raise ValueError(
        "Unsupported method. Use: minmax, percentile, histogram_percentile, mean_std"
    )


# ============================================================
# Asymmetric quantization
# ============================================================

def calculate_asymmetric_qparams(
    calibration_result: CalibrationResult,
    dtype: str,
) -> QuantParams:

    qmin, qmax, _ = get_signed_qrange(dtype)

    rmin, rmax = ensure_valid_range(
        calibration_result.rmin,
        calibration_result.rmax,
        include_zero=True,
    )

    scale = (rmax - rmin) / float(qmax - qmin)

    if scale <= 0:
        raise ValueError("Scale must be positive.")

    zero_point = int(round(qmin - rmin / scale))
    zero_point = int(np.clip(zero_point, qmin, qmax))

    return QuantParams(
        dtype=dtype,
        qmin=qmin,
        qmax=qmax,
        scale=float(scale),
        zero_point=zero_point,
        rmin=rmin,
        rmax=rmax,
        calibration_method=calibration_result.method,
    )


def overflow_report_before_cast(
    x: np.ndarray,
    params: QuantParams,
) -> Dict[str, Any]:

    q_raw = np.round(x.astype(np.float64) / params.scale + params.zero_point)

    overflow_low = int(np.sum(q_raw < params.qmin))
    overflow_high = int(np.sum(q_raw > params.qmax))

    return {
        "raw_q_min": float(np.min(q_raw)),
        "raw_q_max": float(np.max(q_raw)),
        "overflow_low_count": overflow_low,
        "overflow_high_count": overflow_high,
        "overflow_total": overflow_low + overflow_high,
        "total_values": int(q_raw.size),
        "overflow_percent": 100.0 * (overflow_low + overflow_high) / q_raw.size,
    }


def quantize_asymmetric(
    x: np.ndarray,
    params: QuantParams,
) -> np.ndarray:

    _, _, np_dtype = get_signed_qrange(params.dtype)

    q = np.round(x.astype(np.float64) / params.scale + params.zero_point)

    # Very important: clip before casting
    q = np.clip(q, params.qmin, params.qmax)

    return q.astype(np_dtype)


def dequantize_asymmetric(
    q: np.ndarray,
    params: QuantParams,
) -> np.ndarray:

    return (params.scale * (q.astype(np.float32) - params.zero_point)).astype(np.float32)


# ============================================================
# Quantize full folder
# ============================================================

def quantize_folder(
    input_folder: str,
    output_folder: str,
    dtype: str = "int16",
    calibration_method: str = "minmax",
    image_mode: str = "RGB",
    resize_hw: Optional[Tuple[int, int]] = None,
    layout: str = "NHWC",
    save_dequant: bool = False,
) -> QuantParams:

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = get_input_files(str(input_folder))

    calibration_result = calibrate_folder_range(
        input_folder=str(input_folder),
        method=calibration_method,
        image_mode=image_mode,
        resize_hw=resize_hw,
        layout=layout,
        include_zero=True,
        max_values=2_000_000,
        lower_percentile=0.1,
        upper_percentile=99.9,
        bins=2048,
        num_std=3.0,
    )

    params = calculate_asymmetric_qparams(
        calibration_result=calibration_result,
        dtype=dtype,
    )

    print("\n========== Quantization Params ==========")
    print(f"dtype       : {params.dtype}")
    print(f"qmin        : {params.qmin}")
    print(f"qmax        : {params.qmax}")
    print(f"rmin        : {params.rmin}")
    print(f"rmax        : {params.rmax}")
    print(f"scale       : {params.scale}")
    print(f"zero_point  : {params.zero_point}")

    total_overflow = 0
    total_values = 0

    for file_path in files:
        x = load_input_file(
            file_path=file_path,
            image_mode=image_mode,
            resize_hw=resize_hw,
            layout=layout,
            add_batch_dim=True,
        )

        report = overflow_report_before_cast(x, params)

        total_overflow += report["overflow_total"]
        total_values += report["total_values"]

        q = quantize_asymmetric(x, params)

        relative_path = file_path.relative_to(input_folder)
        output_path = output_folder / relative_path
        output_path = output_path.with_suffix(f".{dtype}.npy")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, q)

        if save_dequant:
            dq = dequantize_asymmetric(q, params)
            dq_output_path = output_path.with_suffix(f".{dtype}.dequant.npy")
            np.save(dq_output_path, dq)

        print(f"Saved: {output_path}")
        print(f"  input min/max       : {float(np.min(x))}, {float(np.max(x))}")
        print(f"  raw q min/max       : {report['raw_q_min']}, {report['raw_q_max']}")
        print(f"  overflow count      : {report['overflow_total']}")

    print("\n========== Folder Overflow Summary ==========")
    print(f"Total values      : {total_values}")
    print(f"Total overflow    : {total_overflow}")
    print(f"Overflow percent  : {100.0 * total_overflow / total_values:.6f}%")

    # Save quant params
    params_path = output_folder / f"quant_params_{dtype}.txt"
    with open(params_path, "w") as f:
        f.write(f"dtype={params.dtype}\n")
        f.write(f"qmin={params.qmin}\n")
        f.write(f"qmax={params.qmax}\n")
        f.write(f"rmin={params.rmin}\n")
        f.write(f"rmax={params.rmax}\n")
        f.write(f"scale={params.scale}\n")
        f.write(f"zero_point={params.zero_point}\n")
        f.write(f"calibration_method={params.calibration_method}\n")

    print(f"\nSaved quant params: {params_path}")

    return params


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    input_folder = "calibration_inputs"
    output_folder = "quantized_outputs"

    # Example 1: INT16 asymmetric quantization
    params_int16 = quantize_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        dtype="int16",
        calibration_method="minmax",

        # Use this if your model expects RGB input.
        image_mode="RGB",

        # Set this to your model input size.
        # Example: resize_hw=(224, 224)
        resize_hw=None,

        # Use "NHWC" or "NCHW" based on your model.
        layout="NHWC",

        save_dequant=True,
    )

    # Example 2: INT8 asymmetric quantization
    params_int8 = quantize_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        dtype="int8",
        calibration_method="minmax",
        image_mode="RGB",
        resize_hw=None,
        layout="NHWC",
        save_dequant=True,
    )