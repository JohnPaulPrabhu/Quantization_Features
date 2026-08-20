import numpy as np
import matplotlib.pyplot as plt

# Example:
# data = np.load("data.npy")

# Flatten if data is multidimensional
x = data.flatten()

print("Shape :", data.shape)
print("Count :", x.size)
print("Min   :", np.min(x))
print("Max   :", np.max(x))
print("Mean  :", np.mean(x))
print("Median:", np.median(x))
print("Std   :", np.std(x))

plt.hist(x, bins=100)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Data Distribution")
plt.grid(alpha=0.3)
plt.show()