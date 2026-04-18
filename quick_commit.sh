#!/bin/bash
# quick_commit.sh
# Quick script to stage and commit TFMat files for GitHub

set -e

echo "=========================================="
echo "TFMat Quick Commit Script"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

echo "Step 1: Staging new documentation..."
echo "----------------------------------------"
git add README.md
git add README_TFMAT.md
git add GITHUB_READY.md
git add COMPLETE.md
git add cleanup_for_github.sh
git add docs/PROJECT_STRUCTURE.md
git add docs/RELEASE_CHECKLIST.md
git add docs/TRAINING_EVALUATION_GUIDE.md
git add docs/TABLE4_EXPERIMENT_RESULTS.md
git add docs/TABLE4_README.md
git add docs/table4_experiment_guide.md
git add docs/main.tex
git add docs/sample.bib
git add docs/nature_writer.md
git add docs/*.png
echo "✓ Documentation staged"
echo ""

echo "Step 2: Staging updated .gitignore..."
echo "----------------------------------------"
git add .gitignore
echo "✓ .gitignore staged"
echo ""

echo "Step 3: Staging analysis scripts..."
echo "----------------------------------------"
git add scripts/analyze_element_match_distribution.py
git add scripts/analyze_prediction_errors.py
git add scripts/compute_dng_gen_test_metrics.py
git add scripts/compute_element_composition_match.py
git add scripts/compute_text_prompt_match.py
git add scripts/compute_text_prompt_match_eval_gen.py
git add scripts/compute_text_prompt_match_with_cgcnn.py
git add scripts/convert_eval_diff_to_cached_text.py
git add scripts/convert_eval_gen_to_cached_text.py
git add scripts/detailed_error_analysis.py
git add scripts/plot_distributions_from_results.py
git add scripts/plot_dng_generation_compare_tsne.py
git add scripts/plot_prediction_scatter_and_save.py
git add scripts/plot_property_distributions.py
git add scripts/plot_text_embedding_tsne.py
git add scripts/render_tfmat_top8_ase.py
git add scripts/select_csp_ase_examples.py
git add scripts/select_tfmat_examples.py
git add scripts/select_top2000_and_predict_properties.py
git add scripts/train_cgcnn_for_properties.py
echo "✓ Analysis scripts staged"
echo ""

echo "Step 4: Staging shell scripts..."
echo "----------------------------------------"
git add scripts/run_table4_complete_pipeline.sh
git add scripts/run_table4_generation.sh
git add scripts/run_train_dng_text_lattice5_periodic_last.sh
git add scripts/run_train_dng_text_periodic_ckpt.sh
git add scripts/train_cgcnn_bg_gpu3.sh
git add scripts/train_cgcnn_fe_gpu2.sh
echo "✓ Shell scripts staged"
echo ""

echo "Step 5: Staging modified files..."
echo "----------------------------------------"
git add conf/train/default.yaml
git add diffcsp/run.py
git add result/results-1.csv
git add scripts/eval_utils.py
git add scripts/evaluate.py
echo "✓ Modified files staged"
echo ""

echo "Step 6: Showing what will be committed..."
echo "----------------------------------------"
git status --short
echo ""

echo "Step 7: Ready to commit!"
echo "----------------------------------------"
echo ""
echo "Commit message preview:"
echo "---"
cat << 'EOF'
Prepare TFMat for GitHub release

Major changes:
- Add comprehensive TFMat documentation (README_TFMAT.md)
- Add project structure guide (docs/PROJECT_STRUCTURE.md)
- Add release checklist and GitHub ready guide
- Update .gitignore to exclude large files and results
- Add cleanup script for repository maintenance
- Organize all results into all_results/ directory
- Add analysis and visualization scripts
- Add training shell scripts for reproducibility
- Update training and evaluation configurations

Performance highlights:
- Perov-5: 91.33% match rate (single-sample)
- Carbon-24: 46.40% match rate (single-sample)
- MP-20: 77.80% match rate (single-sample)
- MP-20: 92.04% match rate (num_eval=20)
- Best distributional fidelity on MP-20 DNG

Documentation:
- Complete installation guide
- Training and evaluation examples
- Analysis pipeline documentation
- Project structure overview
EOF
echo "---"
echo ""

read -p "Do you want to commit now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "Prepare TFMat for GitHub release

Major changes:
- Add comprehensive TFMat documentation (README_TFMAT.md)
- Add project structure guide (docs/PROJECT_STRUCTURE.md)
- Add release checklist and GitHub ready guide
- Update .gitignore to exclude large files and results
- Add cleanup script for repository maintenance
- Organize all results into all_results/ directory
- Add analysis and visualization scripts
- Add training shell scripts for reproducibility
- Update training and evaluation configurations

Performance highlights:
- Perov-5: 91.33% match rate (single-sample)
- Carbon-24: 46.40% match rate (single-sample)
- MP-20: 77.80% match rate (single-sample)
- MP-20: 92.04% match rate (num_eval=20)
- Best distributional fidelity on MP-20 DNG

Documentation:
- Complete installation guide
- Training and evaluation examples
- Analysis pipeline documentation
- Project structure overview"

    echo ""
    echo "✓ Committed successfully!"
    echo ""
    echo "Next steps:"
    echo "  git push origin main"
    echo "  or"
    echo "  git push origin <your-branch>"
else
    echo ""
    echo "Commit cancelled. Files are staged and ready."
    echo "You can commit manually with:"
    echo "  git commit"
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
