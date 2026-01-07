import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Time axis
# ----------------------------
T = 1000
t = np.arange(1, T + 1)

rng = np.random.default_rng(42)  # reproducibility

# ----------------------------
# 2) Data (shape + noise like the figure)
# ----------------------------

# Estimated error (blue, decays slowly)
estimated_error = (
    0.75 * np.exp(-t / 90)
    + 0.12 * np.exp(-t / 600)
    + 0.02
)
estimated_error += rng.normal(0, 0.01, T)
estimated_error = np.clip(estimated_error, 0, 1)

# True error (green dashed, decays faster)
true_error = (
    0.95 * np.exp(-t / 45)
    + 0.03
)
true_error += rng.normal(0, 0.008, T)
true_error = np.clip(true_error, 0, 1)

# Predictive accuracy (red dotted, rises to ~1)
predictive_accuracy = (
    1 - 0.75 * np.exp(-t / 80)
)
predictive_accuracy += rng.normal(0, 0.03, T)
predictive_accuracy = np.clip(predictive_accuracy, 0, 1)

# ----------------------------
# 3) Plot
# ----------------------------
fig = plt.figure(figsize=(8, 6), dpi=100)
ax = fig.add_subplot(111)

# Lines (MATLAB defaults)
ax.plot(t, estimated_error, color='b', linewidth=1.0, label='Estimated Error')
ax.plot(t, true_error, color='g', linewidth=1.0,
        linestyle='--', dashes=(6, 6), label='True Error')
ax.plot(t, predictive_accuracy, color='r', linewidth=1.0,
        linestyle=':', dashes=(1, 4), label='Predictive Accuracy')

# Titles and labels (verbatim)
ax.set_title("Learning performance against stationary opponent")
ax.set_xlabel("Time")
ax.set_ylabel("")

# Axes limits and ticks
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1)

ax.set_xticks(np.arange(0, 1001, 100))
ax.set_yticks(np.linspace(0, 1, 11))

# No grid (important)
ax.grid(False)

# Legend (boxed, upper right-ish)
legend = ax.legend(
    loc='center right',
    frameon=True,
    framealpha=1.0,
    fancybox=False,
    borderpad=0.8,
    handlelength=3
)
legend.get_frame().set_edgecolor('black')

plt.tight_layout()
plt.show()
