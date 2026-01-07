import numpy as np
import matplotlib.pyplot as plt

# --- 1) X axis: Cooperation level from -1 to 1
x = np.linspace(-1, 1, 201)

# --- 2) Y values: expected observed probability
# Replace this with YOUR real formula/data.
# This is a smooth S-curve similar to the screenshot.
def expected_prob(x):
    # logistic-ish curve, mapped to ~[0.18, 0.99] like the image
    y = 1 / (1 + np.exp(-3.2 * (x + 0.05)))
    y = 0.16 + 0.84 * y
    return np.clip(y, 0, 1)

y = expected_prob(x)

# --- 3) Plot (match style)
fig = plt.figure(figsize=(6.4, 4.8), dpi=100)  # 640x480 like the screenshot
ax = fig.add_subplot(111)

ax.plot(x, y, 'b-', linewidth=0.8)

ax.set_title("Expected Probability of Observed Move Given Cooperation Level")
ax.set_xlabel("Cooperation Level")
ax.set_ylabel("Expected Observed Probability")

ax.set_xlim(-1, 1)
ax.set_ylim(0, 1)

# Ticks similar to the screenshot
ax.set_xticks(np.linspace(-1, 1, 11))   # -1, -0.8, ..., 1
ax.set_yticks(np.linspace(0, 1, 11))    # 0, 0.1, ..., 1

ax.grid(False)
plt.tight_layout()
plt.show()
