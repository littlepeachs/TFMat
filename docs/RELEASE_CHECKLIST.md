# TFMat GitHub Release Checklist

This document provides a step-by-step guide to prepare the TFMat repository for GitHub release.

## Pre-Release Checklist

### 1. Code Organization

- [x] Core model code in `diffcsp/`
- [x] Evaluation scripts in `scripts/`
- [x] Configuration files in `conf/`
- [x] Documentation in `docs/`
- [ ] Remove debug/experimental code
- [ ] Add docstrings to key functions

### 2. Documentation

- [x] `README_TFMAT.md` - Main project README
- [x] `docs/PROJECT_STRUCTURE.md` - Project structure guide
- [x] `docs/main.tex` - Paper manuscript
- [x] `docs/TABLE4_EXPERIMENT_RESULTS.md` - Experimental results
- [ ] Add code comments where needed
- [ ] Update installation instructions if needed

### 3. Data and Checkpoints

- [ ] Ensure `data_text/` is in `.gitignore` (large datasets)
- [ ] Ensure checkpoint files (`.ckpt`, `.pt`, `.pth`) are in `.gitignore`
- [ ] Keep only essential result CSVs (`result/results-1.csv`, `result/results-20.csv`)
- [ ] Remove temporary analysis outputs

### 4. Configuration

- [x] `.env.template` exists with placeholder values
- [x] `.env` is in `.gitignore`
- [x] `.gitignore` covers all large files
- [ ] Verify all config files in `conf/` are correct

### 5. Clean Up

- [ ] Remove generated crystal files (`generated_crystals/`)
- [ ] Remove preview images (`*_preview.png`)
- [ ] Remove large PDFs
- [ ] Remove temporary analysis results
- [ ] Remove batch generation scripts (or move to `scripts/batch/`)

## Release Steps

### Step 1: Run Cleanup Script

```bash
cd /ssd/liwentao/GenAI/CrystalMF/CrystalFlow
./cleanup_for_github.sh
```

This will:
- Remove large generated files
- Remove preview images
- Remove temporary analysis results
- Show git status

### Step 2: Review Changes

```bash
git status
git diff
```

Check:
- No large files (>10MB) are staged
- No sensitive data (API keys, credentials)
- No personal paths in config files

### Step 3: Stage Files

```bash
# Add new documentation
git add README_TFMAT.md
git add docs/PROJECT_STRUCTURE.md
git add docs/RELEASE_CHECKLIST.md

# Add updated .gitignore
git add .gitignore

# Add cleanup script
git add cleanup_for_github.sh

# Review what will be committed
git status
```

### Step 4: Commit Changes

```bash
git commit -m "Prepare TFMat for GitHub release

- Add comprehensive README_TFMAT.md with installation and usage
- Add PROJECT_STRUCTURE.md documenting codebase organization
- Update .gitignore to exclude large files and generated outputs
- Add cleanup script for repository maintenance
- Clean up temporary files and analysis outputs
"
```

### Step 5: Create Release Branch (Optional)

```bash
git checkout -b release/v1.0.0
git push origin release/v1.0.0
```

### Step 6: Push to GitHub

```bash
git push origin main
```

### Step 7: Create GitHub Release

1. Go to GitHub repository
2. Click "Releases" → "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: "TFMat v1.0.0 - Initial Release"
5. Description:

```markdown
# TFMat: Text-Conditioned Flow Matching for Crystal Structure Generation

First public release of TFMat, a text-conditioned extension of CrystalFlow for crystal structure generation.

## Highlights

- **91.33%** match rate on Perov-5 (single-sample)
- **46.40%** match rate on Carbon-24 (single-sample)
- **77.80%** match rate on MP-20 (single-sample)
- **92.04%** match rate on MP-20 (num_eval=20)
- Best distributional fidelity on MP-20 de novo generation

## What's Included

- Complete model implementation with text conditioning
- Training and evaluation scripts
- Analysis and visualization tools
- Comprehensive documentation
- Example configurations

## Getting Started

See [README_TFMAT.md](README_TFMAT.md) for installation and usage instructions.

## Requirements

- Python 3.11+
- PyTorch 2.3+
- CUDA 12.1+ (for GPU training)

## Note on Data

Due to size constraints, the text-annotated datasets (`data_text/`) and model checkpoints are not included in this release. Please contact the authors for access or prepare your own datasets following the format described in the documentation.
```

## Post-Release Tasks

### 1. Add Badges to README

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3+-red.svg)](https://pytorch.org/)
```

### 2. Set Up GitHub Pages (Optional)

For documentation hosting:

```bash
# In docs/ directory
# Convert markdown to HTML or use Jekyll
```

### 3. Add Issue Templates

Create `.github/ISSUE_TEMPLATE/`:
- `bug_report.md`
- `feature_request.md`
- `question.md`

### 4. Add Contributing Guidelines

Create `CONTRIBUTING.md`:
- Code style guidelines
- Pull request process
- Testing requirements

### 5. Add License

Ensure `LICENSE` file exists (MIT License recommended).

## Verification Checklist

Before announcing the release:

- [ ] Repository is public
- [ ] README renders correctly on GitHub
- [ ] All links in documentation work
- [ ] Installation instructions are accurate
- [ ] Example commands run without errors
- [ ] No sensitive information exposed
- [ ] License file is present
- [ ] Citation information is correct

## Common Issues

### Large Files Rejected

If git rejects large files:

```bash
# Remove from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch <large_file>" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (use with caution!)
git push origin --force --all
```

### Sensitive Data Exposed

If sensitive data was committed:

```bash
# Use BFG Repo-Cleaner
java -jar bfg.jar --delete-files <sensitive_file> .git
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Wrong Files Committed

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

## Maintenance

### Regular Updates

- Update dependencies in `requirements.txt`
- Update paper citation when published
- Add new features to documentation
- Respond to issues and pull requests

### Version Numbering

Follow semantic versioning:
- `v1.0.0` - Initial release
- `v1.0.1` - Bug fixes
- `v1.1.0` - New features (backward compatible)
- `v2.0.0` - Breaking changes

## Contact

For questions about the release process, contact the repository maintainers.
