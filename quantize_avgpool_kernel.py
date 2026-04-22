import numpy as np

def quantize_weights_symmetric(weights: np.ndarray, num_bits: int = 8):
    qmax = (2 ** (num_bits - 1)) - 1
    max_abs = np.max(np.abs(weights))

    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / qmax

    qweights = np.round(weights / scale).astype(np.int8)
    qweights = np.clip(qweights, -qmax, qmax).astype(np.int8)

    return qweights, scale


if __name__ == "__main__":
    weights = np.full((4, 4, 8, 1), 1.0 / 16.0, dtype=np.float32)
    qweights, scale = quantize_weights_symmetric(weights)

    print("Quant scale:", scale)
    print("Unique quantized values:", np.unique(qweights))