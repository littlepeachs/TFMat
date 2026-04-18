#!/bin/bash
# cleanup_for_github.sh
# Script to organize the repository before pushing to GitHub
# Moves results to all_results/ instead of deleting

set -e

echo "=========================================="
echo "TFMat Repository Organization Script"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo ""

# Create all_results directory if it doesn't exist
if [ ! -d "all_results" ]; then
    echo "Creating all_results/ directory..."
    mkdir -p all_results
    echo "✓ Created all_results/"
else
    echo "✓ all_results/ directory already exists"
fi
echo ""

# Function to safely move files/directories
safe_move() {
    if [ -e "$1" ]; then
        echo "Moving: $1 -> all_results/"
        mv "$1" all_results/
    else
        echo "Not found: $1"
    fi
}

echo "Step 1: Moving large generated files to all_results/..."
echo "--------------------------------------------------------"
safe_move "generated_crystals"
safe_move "matplotlib-cache"
safe_move "cgcnn_models"
echo ""

echo "Step 2: Moving preview images and PDFs to all_results/..."
echo "--------------------------------------------------------"
safe_move "AgPbSO2_preview.png"
safe_move "C_preview.png"
safe_move "CsBr_preview.png"
safe_move "CsMnNOF_preview.png"
safe_move "K2Te_preview.png"
safe_move "PbBrF_preview.png"
safe_move "Pr3In_preview.png"
safe_move "Tb2TlHg_preview.png"
safe_move "TePdN3_preview.png"
safe_move "TmIn2Sn_preview.png"
safe_move "TmMg_preview.png"
safe_move "Yb2TlIn_preview.png"
safe_move "Zr2VCo3_preview.png"
safe_move "2503.00522v1 (1).pdf"
safe_move "image-1.png"
echo ""

echo "Step 3: Moving analysis result files to all_results/..."
echo "--------------------------------------------------------"
safe_move "element_composition_match_results_5000.txt"
safe_move "element_match_detailed_distribution.csv"
safe_move "element_match_distribution.csv"
safe_move "element_match_distribution.txt"
safe_move "FINAL_RESULTS_TOP2000.txt"
safe_move "prediction_scatter_plots.png"
safe_move "property_distributions.png"
echo ""

echo "Step 4: Moving batch generation scripts to all_results/..."
echo "--------------------------------------------------------"
mkdir -p all_results/scripts
safe_move "scripts/generate_1000_materials_gpu1.sh"
safe_move "scripts/generate_materials_gpu1.sh"
safe_move "scripts/run_csp_ckpt_targeted_sweep.sh"
safe_move "scripts/run_inference_sweep_text.sh"
[ -e "scripts/run_table4_co" ] && safe_move "scripts/run_table4_co"
echo ""

echo "Step 5: Checking git status..."
echo "--------------------------------------------------------"
git status --short
echo ""

echo "Step 6: Verifying .gitignore..."
echo "--------------------------------------------------------"
if [ -f ".gitignore" ]; then
    echo "✓ .gitignore exists"
    echo "Key patterns:"
    grep -E "(all_results|hydra|data_text)" .gitignore || echo "  (patterns may need updating)"
else
    echo "✗ .gitignore not found!"
fi
echo ""

echo "Step 7: Checking for large files in tracked area..."
echo "--------------------------------------------------------"
echo "Files larger than 10MB (excluding all_results and .git):"
find . -type f -size +10M ! -path "./.git/*" ! -path "./all_results/*" ! -path "./hydra/*" -exec ls -lh {} \; | awk '{print $9, $5}' || echo "  None found"
echo ""

echo "Step 8: Summary..."
echo "--------------------------------------------------------"
echo "Untracked files (excluding all_results):"
git status --porcelain | grep "^??" | grep -v "all_results" | wc -l | xargs echo "  Count:"
echo ""

echo "=========================================="
echo "Organization Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Results moved to: all_results/"
echo "  - Hydra outputs: kept in place (will be ignored)"
echo "  - Data files: data_text/ (will be ignored)"
echo ""
echo "Next steps:"
echo "1. Review the changes: git status"
echo "2. Add new files: git add README_TFMAT.md docs/ .gitignore cleanup_for_github.sh"
echo "3. Commit changes: git commit -m 'Prepare TFMat for GitHub release'"
echo "4. Push to GitHub: git push origin main"
echo ""
