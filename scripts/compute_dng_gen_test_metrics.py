"""
Compute coverage / distributional metrics between generated and test splits
for the two DNG t-SNE CSVs (unconditional baseline vs text-guided).

Metrics (all computed in 2-D t-SNE space):
  JSD          -- Jensen-Shannon Divergence (histogram-based, symmetric KL, lower ↓)
  Coverage     -- fraction of test points within radius r of ≥1 generated point (higher ↑)
  Precision    -- fraction of generated points within radius r of ≥1 test point (higher ↑)
  MMD          -- Maximum Mean Discrepancy with RBF kernel (lower ↓)

For Coverage & Precision the radius r is set to the mean k-NN distance in the test set
(adaptive, dataset-dependent), following the Improved P&R framework.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.special import rel_entr

ROOT = Path("/ssd/liwentao/GenAI/CrystalMF/CrystalFlow")

CSVS = {
    "Unconditional": ROOT / "generated_crystals/dng_embedding_tsne_compare_10kgen/mp_20_dng/mp_20_dng_tsne_points.csv",
    "Text-guided":   ROOT / "generated_crystals/dng_embedding_tsne_compare_10kgen/mp_20_dng_text/mp_20_dng_text_tsne_points.csv",
}

K_NEIGHBOR = 5   # k for adaptive radius
JSD_BINS   = 50  # bins per axis for histogram-based JSD


# ── helpers ──────────────────────────────────────────────────────────────────

def jsd_2d(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    """Jensen-Shannon Divergence via 2-D histogram (base-2, in [0,1])."""
    xmin = min(a[:, 0].min(), b[:, 0].min())
    xmax = max(a[:, 0].max(), b[:, 0].max())
    ymin = min(a[:, 1].min(), b[:, 1].min())
    ymax = max(a[:, 1].max(), b[:, 1].max())
    edges = [np.linspace(xmin, xmax, bins + 1),
             np.linspace(ymin, ymax, bins + 1)]
    ha, _ = np.histogramdd(a, bins=edges)
    hb, _ = np.histogramdd(b, bins=edges)
    pa = (ha.astype(float) + 1e-10)          # add small eps to avoid log(0)
    pb = (hb.astype(float) + 1e-10)
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    jsd = 0.5 * rel_entr(pa, m).sum() + 0.5 * rel_entr(pb, m).sum()
    jsd /= np.log(2)                         # convert nats → bits
    return float(np.clip(jsd, 0.0, 1.0))


def adaptive_radius(points: np.ndarray, k: int) -> float:
    """Mean distance to k-th nearest neighbour within `points`."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)   # +1 because first hit is self
    return float(dists[:, k].mean())


def coverage_precision(gen: np.ndarray, test: np.ndarray,
                        k: int = K_NEIGHBOR):
    """
    Coverage  (recall-like): fraction of test pts with ≥1 gen pt within r_test.
    Precision (quality):     fraction of gen  pts with ≥1 test pt within r_gen.
    Radii are adaptive (mean k-NN distance within the respective reference set).
    """
    r_test = adaptive_radius(test, k)
    r_gen  = adaptive_radius(gen,  k)

    tree_gen  = cKDTree(gen)
    tree_test = cKDTree(test)

    # coverage: for each test point, is there a gen point within r_test?
    hits_cov = tree_gen.query_ball_point(test, r=r_test, return_length=True)
    coverage = float((hits_cov > 0).sum()) / len(test)

    # precision: for each gen point, is there a test point within r_gen?
    hits_pre = tree_test.query_ball_point(gen, r=r_gen, return_length=True)
    precision = float((hits_pre > 0).sum()) / len(gen)

    return coverage, precision


def mmd_rbf(a: np.ndarray, b: np.ndarray, max_pts: int = 2000,
            sigma: float | None = None) -> float:
    """
    Unbiased MMD² with RBF kernel.
    Subsample for speed when point clouds are large.
    """
    rng = np.random.default_rng(42)
    if len(a) > max_pts:
        a = a[rng.choice(len(a), max_pts, replace=False)]
    if len(b) > max_pts:
        b = b[rng.choice(len(b), max_pts, replace=False)]

    if sigma is None:
        # median heuristic on pooled data
        pooled = np.concatenate([a, b], axis=0)
        diffs = pooled[:, None, :] - pooled[None, :, :]
        sq_dists = (diffs ** 2).sum(-1)
        sigma = float(np.sqrt(np.median(sq_dists[sq_dists > 0]) / 2))

    def k(x, y):
        d = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        return np.exp(-d / (2 * sigma ** 2))

    kxx = k(a, a)
    kyy = k(b, b)
    kxy = k(a, b)
    na, nb = len(a), len(b)
    # unbiased estimates
    mmd2 = (kxx.sum() - np.trace(kxx)) / (na * (na - 1)) \
         + (kyy.sum() - np.trace(kyy)) / (nb * (nb - 1)) \
         - 2 * kxy.mean()
    return float(max(mmd2, 0.0))


# ── main ─────────────────────────────────────────────────────────────────────

def compute(name: str, csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    coords = df[["tsne_x", "tsne_y"]].values.astype(np.float32)
    gen  = coords[df["split"] == "generated"]
    test = coords[df["split"] == "test"]
    print(f"\n[{name}]  gen={len(gen):,}  test={len(test):,}")

    jsd            = jsd_2d(gen, test, bins=JSD_BINS)
    cov, prec      = coverage_precision(gen, test)
    mmd            = mmd_rbf(gen, test)

    print(f"  JSD (↓ better):       {jsd:.4f}")
    print(f"  Coverage / COV-R (↑): {cov * 100:.2f}%")
    print(f"  Precision / COV-P (↑):{prec * 100:.2f}%")
    print(f"  MMD² (↓ better):      {mmd:.6f}")

    return {"name": name, "JSD": jsd, "Coverage": cov, "Precision": prec, "MMD2": mmd}


def main():
    results = []
    for name, path in CSVS.items():
        results.append(compute(name, path))

    # summary table
    print("\n" + "=" * 65)
    print(f"{'Metric':<22} {'Unconditional':>18} {'Text-guided':>18}")
    print("-" * 65)
    r0, r1 = results[0], results[1]
    metrics = [
        ("JSD ↓",          "JSD",      False),
        ("Coverage ↑ (%)", "Coverage", True),
        ("Precision ↑ (%)", "Precision", True),
        ("MMD² ↓",         "MMD2",     False),
    ]
    for label, key, pct in metrics:
        v0 = r0[key] * (100 if pct else 1)
        v1 = r1[key] * (100 if pct else 1)
        fmt = ".2f" if pct else ".4f" if key == "JSD" else ".6f"
        print(f"  {label:<20} {v0:>{18}{fmt}} {v1:>{18}{fmt}}")
    print("=" * 65)
    print("\nNotes:")
    print("  JSD in [0,1] (bits); Coverage/Precision from adaptive k-NN radius (k=5).")
    print("  MMD² uses RBF kernel with median heuristic (subsample ≤2000 pts).")


if __name__ == "__main__":
    main()
