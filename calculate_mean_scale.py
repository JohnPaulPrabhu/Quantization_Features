import numpy as np

npy_files = [
    "data1.npy",
    "data2.npy",
    "data3.npy",
]

count = 0
total_sum = 0.0
total_sq_sum = 0.0

for file in npy_files:
    data = np.load(file).astype(np.float64)

    count += data.size
    total_sum += np.sum(data)
    total_sq_sum += np.sum(data * data)

mean = total_sum / count

variance = (total_sq_sum / count) - (mean * mean)
std = np.sqrt(variance)

scale = 1.0 / std

print("Count :", count)
print("Mean  :", mean)
print("Std   :", std)
print("Scale :", scale)