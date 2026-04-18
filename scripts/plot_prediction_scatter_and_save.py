"""
Plot prediction scatter plots and save results to TXT and CSV files
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'

# Parse the top2000 results file
material_ids = []
formation_energies_gt = []
formation_energies_pred = []
band_gaps_gt = []
band_gaps_pred = []
element_scores = []

with open('top2000_element_and_property_match_results.txt', 'r') as f:
    content = f.read()

    # Split by material entries
    entries = content.split('Material ID: ')[1:]  # Skip header

    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) < 3:
            continue

        # Extract material ID
        mat_id = lines[0].strip()

        # Extract element score
        elem_match = re.search(r'Element Score: ([\d.]+)', entry)
        elem_score = float(elem_match.group(1)) if elem_match else 0.0

        # Extract formation energy
        fe_match = re.search(r'Formation Energy: GT=([-\d.]+), Pred=([-\d.]+)', entry)
        if fe_match:
            gt_fe = float(fe_match.group(1))
            pred_fe = float(fe_match.group(2))

            # Filter outliers (abs > 10 for FE)
            if abs(pred_fe) < 10:
                material_ids.append(mat_id)
                element_scores.append(elem_score)
                formation_energies_gt.append(gt_fe)
                formation_energies_pred.append(pred_fe)

                # Extract band gap for the same material
                bg_match = re.search(r'Band Gap: GT=([-\d.]+), Pred=([-\d.]+)', entry)
                if bg_match:
                    gt_bg = float(bg_match.group(1))
                    pred_bg = float(bg_match.group(2))
                    band_gaps_gt.append(gt_bg)
                    band_gaps_pred.append(pred_bg)
                else:
                    band_gaps_gt.append(np.nan)
                    band_gaps_pred.append(np.nan)

# Convert to numpy arrays
fe_gt = np.array(formation_energies_gt)
fe_pred = np.array(formation_energies_pred)
bg_gt = np.array(band_gaps_gt)
bg_pred = np.array(band_gaps_pred)

print(f"Total samples (after outlier removal): {len(material_ids)}")
print(f"Formation Energy samples: {len(fe_gt)}")
print(f"Band Gap samples: {np.sum(~np.isnan(bg_gt))}")

# Calculate errors
fe_errors = np.abs(fe_gt - fe_pred)
bg_errors = np.abs(bg_gt - bg_pred)

print(f"\nFormation Energy MAE: {np.mean(fe_errors):.4f} eV/atom")
print(f"Band Gap MAE: {np.nanmean(bg_errors):.4f} eV")

# Create scatter plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Formation Energy scatter
ax1 = axes[0]
scatter1 = ax1.scatter(fe_gt, fe_pred, c=fe_errors, cmap='viridis',
                       alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
# Add diagonal line (perfect prediction)
min_val = min(fe_gt.min(), fe_pred.min())
max_val = max(fe_gt.max(), fe_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='Perfect Prediction')
ax1.set_xlabel('Ground Truth Formation Energy (eV/atom)', fontsize=24)
ax1.set_ylabel('Predicted Formation Energy (eV/atom)', fontsize=24)
ax1.set_title('Formation Energy: Prediction vs Ground Truth', fontsize=21, fontweight='bold')
ax1.legend(fontsize=22)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Absolute Error (eV/atom)', fontsize=20)
cbar1.ax.tick_params(labelsize=18)

# Plot 2: Band Gap scatter
ax2 = axes[1]
valid_mask = ~np.isnan(bg_gt)
scatter2 = ax2.scatter(bg_gt[valid_mask], bg_pred[valid_mask],
                       c=bg_errors[valid_mask], cmap='plasma',
                       alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
# Add diagonal line
min_val = 0
max_val = max(bg_gt[valid_mask].max(), bg_pred[valid_mask].max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='Perfect Prediction')
ax2.set_xlabel('Ground Truth Band Gap (eV)', fontsize=24)
ax2.set_ylabel('Predicted Band Gap (eV)', fontsize=24)
ax2.set_title('Band Gap: Prediction vs Ground Truth', fontsize=21, fontweight='bold')
ax2.legend(fontsize=22)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Absolute Error (eV)', fontsize=20)
cbar2.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig('prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved scatter plots to: prediction_scatter_plots.png")

# Save results to CSV
df = pd.DataFrame({
    'material_id': material_ids,
    'element_score': element_scores,
    'formation_energy_gt': fe_gt,
    'formation_energy_pred': fe_pred,
    'formation_energy_error': fe_errors,
    'band_gap_gt': bg_gt,
    'band_gap_pred': bg_pred,
    'band_gap_error': bg_errors
})

csv_file = 'top2000_predictions.csv'
df.to_csv(csv_file, index=False, float_format='%.6f')
print(f"✓ Saved predictions to: {csv_file}")

# Save results to TXT (formatted)
txt_file = 'top2000_predictions.txt'
with open(txt_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Top 2000 Samples - Prediction Results (After Outlier Removal)\n")
    f.write("="*80 + "\n\n")

    f.write(f"Total Samples: {len(material_ids)}\n")
    f.write(f"Formation Energy MAE: {np.mean(fe_errors):.4f} eV/atom\n")
    f.write(f"Band Gap MAE: {np.nanmean(bg_errors):.4f} eV\n\n")

    f.write("="*80 + "\n")
    f.write("Detailed Predictions:\n")
    f.write("="*80 + "\n\n")

    for i, mat_id in enumerate(material_ids):
        f.write(f"Material ID: {mat_id}\n")
        f.write(f"  Element Score: {element_scores[i]:.4f}\n")
        f.write(f"  Formation Energy:\n")
        f.write(f"    Ground Truth: {fe_gt[i]:.6f} eV/atom\n")
        f.write(f"    Predicted:    {fe_pred[i]:.6f} eV/atom\n")
        f.write(f"    Error:        {fe_errors[i]:.6f} eV/atom\n")
        f.write(f"  Band Gap:\n")
        if not np.isnan(bg_gt[i]):
            f.write(f"    Ground Truth: {bg_gt[i]:.6f} eV\n")
            f.write(f"    Predicted:    {bg_pred[i]:.6f} eV\n")
            f.write(f"    Error:        {bg_errors[i]:.6f} eV\n")
        else:
            f.write(f"    N/A\n")
        f.write("\n")

print(f"✓ Saved predictions to: {txt_file}")

# Print summary statistics
print("\n" + "="*80)
print("Summary Statistics:")
print("="*80)
print(f"\nFormation Energy:")
print(f"  MAE:    {np.mean(fe_errors):.4f} eV/atom")
print(f"  RMSE:   {np.sqrt(np.mean(fe_errors**2)):.4f} eV/atom")
print(f"  Median: {np.median(fe_errors):.4f} eV/atom")
print(f"  Std:    {np.std(fe_errors):.4f} eV/atom")

print(f"\nBand Gap:")
valid_bg_errors = bg_errors[~np.isnan(bg_errors)]
print(f"  MAE:    {np.mean(valid_bg_errors):.4f} eV")
print(f"  RMSE:   {np.sqrt(np.mean(valid_bg_errors**2)):.4f} eV")
print(f"  Median: {np.median(valid_bg_errors):.4f} eV")
print(f"  Std:    {np.std(valid_bg_errors):.4f} eV")
