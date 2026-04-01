# python scripts/compute_metrics.py \
#   --root_path /ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/crystalmf-imf-mp20-baseline \
#   --tasks csp \
#   --gt_file data/mp_20/test.csv \
#   --label imf_1step

python scripts/compute_metrics.py \
  --root_path /ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/CSP-mp20 \
  --tasks csp \
  --gt_file data/mp_20/test.csv \
  --label csp1
