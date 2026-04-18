import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'

# Parse the top2000 results file
formation_energies_gt = []
formation_energies_pred = []
band_gaps_gt = []
band_gaps_pred = []

with open('top2000_element_and_property_match_results.txt', 'r') as f:
    content = f.read()

    # Extract formation energy and band gap values
    fe_pattern = r'Formation Energy: GT=([-\d.]+), Pred=([-\d.]+)'
    bg_pattern = r'Band Gap: GT=([-\d.]+), Pred=([-\d.]+)'

    for match in re.finditer(fe_pattern, content):
        gt_val = float(match.group(1))
        pred_val = float(match.group(2))
        # Filter outliers (abs > 10 for FE)
        if abs(pred_val) < 10:
            formation_energies_gt.append(gt_val)
            formation_energies_pred.append(pred_val)

    for match in re.finditer(bg_pattern, content):
        gt_val = float(match.group(1))
        pred_val = float(match.group(2))
        # Filter outliers (> 10 for BG)
        if pred_val < 10:
            band_gaps_gt.append(gt_val)
            band_gaps_pred.append(pred_val)

fe_gt = np.array(formation_energies_gt)
fe_pred = np.array(formation_energies_pred)
bg_gt = np.array(band_gaps_gt)
bg_pred = np.array(band_gaps_pred)

print(f"Found {len(fe_gt)} formation energy samples")
print(f"Found {len(bg_gt)} band gap samples")

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Formation Energy
ax = axes[0]
if len(fe_gt) > 0 and len(fe_pred) > 0:
    kde_gt = gaussian_kde(fe_gt)
    kde_pred = gaussian_kde(fe_pred)

    x_range = np.linspace(min(fe_gt.min(), fe_pred.min()),
                          max(fe_gt.max(), fe_pred.max()), 200)

    ax.plot(x_range, kde_gt(x_range), label='Ground Truth', linewidth=3)
    ax.plot(x_range, kde_pred(x_range), label='Predicted', linewidth=3, linestyle='--')
    ax.set_xlabel('Formation Energy (eV/atom)', fontsize=24)
    ax.set_ylabel('Density', fontsize=24)
    ax.set_title('Formation Energy Distribution', fontsize=21, fontweight='bold')
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)

# Plot Band Gap
ax = axes[1]
if len(bg_gt) > 0 and len(bg_pred) > 0:
    kde_gt = gaussian_kde(bg_gt)
    kde_pred = gaussian_kde(bg_pred)

    x_range = np.linspace(0, max(bg_gt.max(), bg_pred.max()), 200)

    ax.plot(x_range, kde_gt(x_range), label='Ground Truth', linewidth=3)
    ax.plot(x_range, kde_pred(x_range), label='Predicted', linewidth=3, linestyle='--')
    ax.set_xlabel('Band Gap (eV)', fontsize=24)
    ax.set_ylabel('Density', fontsize=24)
    ax.set_title('Band Gap Distribution', fontsize=21, fontweight='bold')
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('property_distributions.png', dpi=300, bbox_inches='tight')
print("\nSaved plot to property_distributions.png")
