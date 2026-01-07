import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Time axis
# ----------------------------
T = 1000
t = np.arange(1, T + 1)

rng = np.random.default_rng(7)  # deterministic noise

# ----------------------------
# 2) Data (match the visual behavior)
# ----------------------------

# Cooperation level: rises very fast from ~0.4 to ~1 and then stays there
coop = 0.4 + 0.6 * (1 - np.exp(-(t - 1) / 18.0))
coop = np.clip(coop, 0, 1)

# Average payoff: noisy dotted series that rises to ~0.8-0.88 and fluctuates
avg_payoff = 0.45 + 0.40 * (1 - np.exp(-(t - 1) / 45.0))
avg_payoff += rng.normal(0, 0.018, size=T)  # dotted “noise cloud”
avg_payoff = np.clip(avg_payoff, 0, 1)

# ----------------------------
# 3) Plot styling (match figure)
# ----------------------------
fig = plt.figure(figsize=(6.4, 4.8), dpi=100)  # ~640x480, like MATLAB
ax = fig.add_subplot(111)

# Blue dotted "Average payoff"
ax.plot(t, avg_payoff, color='b', linewidth=0.9,
        linestyle=':', dashes=(1, 4), label='Average payoff')

# Red solid "Cooperation level"
ax.plot(t, coop, color='r', linewidth=0.9,
        linestyle='-', label='Cooperation level')

# Title + labels (verbatim)
ax.set_title("Achieving cooperation using reciprocation")
ax.set_xlabel("Time")

# Limits and ticks (match layout)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1)

ax.set_xticks(np.arange(0, 1001, 100))
ax.set_yticks(np.linspace(0, 1, 11))

ax.grid(False)

# Legend: boxed, bottom-right-ish
leg = ax.legend(
    loc='lower right',
    bbox_to_anchor=(0.94, 0.12),  # nudges it inward like the screenshot
    frameon=True,
    framealpha=1.0,
    fancybox=False,
    borderpad=0.8,
    handlelength=3
)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.show()
