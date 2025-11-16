# Create accuracy_auc_vs_ratio.png based on the provided table.
import matplotlib.pyplot as plt

# Data from the user's table (three-decimal consistency)
ratios = [0, 20, 40, 60, 80, 100]
accuracy = [0.780, 0.810, 0.832, 0.841, 0.842, 0.831]
auc =      [0.850, 0.870, 0.891, 0.902, 0.903, 0.889]

# Simple symmetric "error bars" to indicate uncertainty (as discussed)
err = [0.01]*len(ratios)  # ±0.01

fig = plt.figure(figsize=(5.0, 3.4), dpi=150)

# Shaded sweet-spot region 40–60% (no explicit color; light alpha)
plt.axvspan(40, 60, alpha=0.08)

# Plot Accuracy with error bars
plt.errorbar(ratios, accuracy, yerr=err, marker='o', linestyle='-', label='Accuracy', capsize=3)

# Plot AUC with error bars
plt.errorbar(ratios, auc, yerr=err, marker='o', linestyle='-', label='AUC', capsize=3)

plt.xlabel('Synthetic Ratio r (%)')
plt.ylabel('Score')
plt.ylim(0.77, 0.94)
plt.xlim(-2, 102)
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig("accuracy_auc_vs_ratio.png", dpi=400)
plt.close()
plt.close()


