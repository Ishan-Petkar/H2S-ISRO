import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

# Generate bivariate normal distribution
r = 0.225
mean = [0, 0]
cov = [[1, r], [r, 1]]
x, y = np.random.multivariate_normal(mean, cov, 1000000).T

plt.figure(figsize=(8, 8))
plt.hist2d(x, y, bins=100, cmap='Blues', density=True)
plt.colorbar(label='Density')

# Highlight the anomaly zone (x >= 2.0, y <= -2.0)
rect = patches.Rectangle((2.0, -4.5), 2.5, 2.5, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
plt.gca().add_patch(rect)

# Draw dashed threshold lines
plt.axvline(2.0, color='red', linestyle='--', alpha=0.8)
plt.axhline(-2.0, color='red', linestyle='--', alpha=0.8)

plt.xlabel(r'CPR Anomaly ($z_{\mathrm{CPR}}$)', fontsize=14)
plt.ylabel(r'DOP Anomaly ($z_{\mathrm{DOP}}$)', fontsize=14)
plt.title(f'Bivariate Distribution of Dry Regolith ($r = {r}$)', fontsize=16)

plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('../results/bivariate_anomaly_space.pdf', format='pdf', bbox_inches='tight')
print("Successfully generated bivariate_anomaly_space.pdf")
