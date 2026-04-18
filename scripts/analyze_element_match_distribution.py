"""
Analyze element composition match distribution for top 2000 samples
"""
import re
import numpy as np
import pandas as pd

# Parse the top2000 results file
element_scores = []

with open('top2000_element_and_property_match_results.txt', 'r') as f:
    content = f.read()

    # Split by material entries
    entries = content.split('Material ID: ')[1:]  # Skip header

    for entry in entries:
        # Extract element score
        elem_match = re.search(r'Element Score: ([\d.]+)', entry)
        if elem_match:
            elem_score = float(elem_match.group(1))
            element_scores.append(elem_score)

element_scores = np.array(element_scores)

print(f"Total samples analyzed: {len(element_scores)}")
print(f"Average element match score: {np.mean(element_scores):.4f}\n")

# Categorize by match level
perfect_match = np.sum(element_scores == 1.0)
partial_match = np.sum((element_scores > 0) & (element_scores < 1.0))
no_match = np.sum(element_scores == 0.0)

total = len(element_scores)

# Create distribution table
print("="*70)
print("Element Composition Match Distribution")
print("="*70)
print(f"{'Match Category':<30} {'Count':<10} {'Percentage':<15} {'Score Range':<15}")
print("-"*70)

# Perfect match (score = 1.0)
pct = (perfect_match / total) * 100
print(f"{'Complete Match':<30} {perfect_match:<10} {pct:>6.2f}%{'':<8} {'[1.0]':<15}")

# Partial match (0 < score < 1.0)
pct = (partial_match / total) * 100
print(f"{'Partial Match':<30} {partial_match:<10} {pct:>6.2f}%{'':<8} {'(0.0, 1.0)':<15}")

# No match (score = 0.0)
pct = (no_match / total) * 100
print(f"{'No Match':<30} {no_match:<10} {pct:>6.2f}%{'':<8} {'[0.0]':<15}")

print("-"*70)

# Overall average
avg_score = np.mean(element_scores)
print(f"{'Overall Average Match Score':<30} {total:<10} {100.0:>6.2f}%{'':<8} {f'{avg_score:.4f}':<15}")

print("="*70)

# More detailed breakdown for partial matches
print("\nDetailed Partial Match Breakdown:")
print("="*70)

# Get all unique scores and their counts
unique_scores = np.unique(element_scores)
print(f"{'Score':<10} {'Count':<10} {'Percentage':<15} {'Description':<40}")
print("-"*70)

score_descriptions = {
    1.0: "Complete (all elements)",
    0.8: "80% match (4/5 elements)",
    0.75: "75% match (3/4 elements)",
    0.667: "66.7% match (2/3 elements)",
    0.5: "50% match (1/2 or 2/4 elements)",
    0.333: "33.3% match (1/3 elements)",
    0.25: "25% match (1/4 elements)",
    0.0: "No match (0 elements)"
}

detailed_data = []
for score in sorted(unique_scores, reverse=True):
    count = np.sum(np.abs(element_scores - score) < 0.01)
    pct = (count / total) * 100

    # Find closest description
    desc = "Unknown"
    for ref_score, ref_desc in score_descriptions.items():
        if abs(score - ref_score) < 0.01:
            desc = ref_desc
            break

    print(f"{score:<10.4f} {count:<10} {pct:>6.2f}%{'':<8} {desc:<40}")
    detailed_data.append({
        'Score': f"{score:.4f}",
        'Count': count,
        'Percentage': f"{pct:.2f}%",
        'Description': desc
    })

print("="*70)

# Save to CSV
df = pd.DataFrame({
    'Match_Category': ['Complete Match', 'Partial Match', 'No Match', 'Overall Average'],
    'Count': [perfect_match, partial_match, no_match, total],
    'Percentage': [
        f"{(perfect_match/total)*100:.2f}%",
        f"{(partial_match/total)*100:.2f}%",
        f"{(no_match/total)*100:.2f}%",
        "100.00%"
    ],
    'Score_Range': ['[1.0]', '(0.0, 1.0)', '[0.0]', f'{avg_score:.4f}']
})

csv_file = 'element_match_distribution.csv'
df.to_csv(csv_file, index=False)
print(f"\n✓ Saved distribution table to: {csv_file}")

# Save detailed breakdown
df_detailed = pd.DataFrame(detailed_data)
csv_detailed = 'element_match_detailed_distribution.csv'
df_detailed.to_csv(csv_detailed, index=False)
print(f"✓ Saved detailed distribution to: {csv_detailed}")

# Save to TXT
txt_file = 'element_match_distribution.txt'
with open(txt_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("Element Composition Match Distribution (Top 2000 Samples)\n")
    f.write("="*70 + "\n\n")

    f.write(f"Total samples analyzed: {len(element_scores)}\n")
    f.write(f"Average element match score: {np.mean(element_scores):.4f}\n\n")

    f.write("="*70 + "\n")
    f.write(f"{'Match Category':<30} {'Count':<10} {'Percentage':<15} {'Score Range':<15}\n")
    f.write("-"*70 + "\n")

    pct = (perfect_match / total) * 100
    f.write(f"{'Complete Match':<30} {perfect_match:<10} {pct:>6.2f}%{'':<8} {'[1.0]':<15}\n")

    pct = (partial_match / total) * 100
    f.write(f"{'Partial Match':<30} {partial_match:<10} {pct:>6.2f}%{'':<8} {'(0.0, 1.0)':<15}\n")

    pct = (no_match / total) * 100
    f.write(f"{'No Match':<30} {no_match:<10} {pct:>6.2f}%{'':<8} {'[0.0]':<15}\n")

    f.write("-"*70 + "\n")
    f.write(f"{'Overall Average Match Score':<30} {total:<10} {100.0:>6.2f}%{'':<8} {f'{avg_score:.4f}':<15}\n")
    f.write("="*70 + "\n")

print(f"✓ Saved distribution summary to: {txt_file}")
