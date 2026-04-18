"""
Plot KDE distributions of formation energy and band gap for Top 2000 samples
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Read the Top 2000 samples
df = pd.read_csv('data_text/mp_20/test.csv')
top2000_file = 'top2000_element_and_property_match_results.txt'

# Read the Top 2000 material IDs
top2000_ids = []
with open(top2000_file, 'r') as f:
    for line in f:
        if line.startswith('material_id_'):
            material_id = line.split(':')[0].strip()
            top2000_ids.append(material_id)

print(f"Found {len(top2000_ids)} material IDs")

# Filter the dataframe
df_top2000 = df[df['material_id'].isin(top2000_ids)].copy()
print(f"Matched {len(df_top2000)} samples in test.csv")

# Get formation energy and band gap
formation_energy = df_top2000['formation_energy_per_atom'].values
band_gap = df_top2000['band_gap'].values

print(f"\n=== Formation Energy Statistics ===")
print(f"Mean: {formation_energy.mean():.4f} eV/atom")
print(f"Std: {formation_energy.std():.4f} eV/atom")
print(f"Min: {formation_energy.min():.4f} eV/atom")
print(f"Max: {formation_energy.max():.4f} eV/atom")
print(f"Median: {np.median(formation_energy):.4f} eV/atom")

print(f"\n=== Band Gap Statistics ===")
print(f"Mean: {band_gap.mean():.4f} eV")
print(f"Std: {band_gap.std():.4f} eV")
print(f"Min: {band_gap.min():.4f} eV")
print(f"Max: {band_gap.max():.4f} eV")
print(f"Median: {np.median(band_gap):.4f} eV")
print(f"Metals (BG ≈ 0): {(band_gap < 0.01).sum()} ({(band_gap < 0.01).sum()/len(band_gap)*100:.2f}%)")
print(f"Semiconductors (BG > 0): {(band_gap >= 0.01).sum()} ({(band_gap >= 0.01).sum()/len(band_gap)*100:.2f}%)")

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Formation Energy KDE
ax1 = axes[0]
kde_fe = gaussian_kde(formation_energy)
x_fe = np.linspace(formation_energy.min(), formation_energy.max(), 1000)
ax1.fill_between(x_fe, kde_fe(x_fe), alpha=0.6, color='steelblue')
ax1.plot(x_fe, kde_fe(x_fe), color='darkblue', linewidth=2)
ax1.axvline(formation_energy.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {formation_energy.mean():.3f}')
ax1.axvline(np.median(formation_energy), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(formation_energy):.3f}')
ax1.set_xlabel('Formation Energy (eV/atom)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Formation Energy Distribution (Top 2000 Samples)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Band Gap KDE
ax2 = axes[1]
kde_bg = gaussian_kde(band_gap)
x_bg = np.linspace(band_gap.min(), band_gap.max(), 1000)
ax2.fill_between(x_bg, kde_bg(x_bg), alpha=0.6, color='coral')
ax2.plot(x_bg, kde_bg(x_bg), color='darkred', linewidth=2)
ax2.axvline(band_gap.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {band_gap.mean():.3f}')
ax2.axvline(np.median(band_gap), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(band_gap):.3f}')
ax2.set_xlabel('Band Gap (eV)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Band Gap Distribution (Top 2000 Samples)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('top2000_property_distributions.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to: top2000_property_distributions.png")

# Also create a histogram version for better visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Formation Energy Histogram + KDE
ax1 = axes[0]
ax1.hist(formation_energy, bins=50, density=True, alpha=0.5, color='steelblue', edgecolor='black')
sns.kdeplot(formation_energy, ax=ax1, color='darkblue', linewidth=2.5)
ax1.axvline(formation_energy.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {formation_energy.mean():.3f}')
ax1.axvline(np.median(formation_energy), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(formation_energy):.3f}')
ax1.set_xlabel('Formation Energy (eV/atom)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Formation Energy Distribution (Histogram + KDE)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Band Gap Histogram + KDE
ax2 = axes[1]
ax2.hist(band_gap, bins=50, density=True, alpha=0.5, color='coral', edgecolor='black')
sns.kdeplot(band_gap, ax=ax2, color='darkred', linewidth=2.5)
ax2.axvline(band_gap.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {band_gap.mean():.3f}')
ax2.axvline(np.median(band_gap), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(band_gap):.3f}')
ax2.set_xlabel('Band Gap (eV)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Band Gap Distribution (Histogram + KDE)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('top2000_property_distributions_hist.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved plot to: top2000_property_distributions_hist.png")
