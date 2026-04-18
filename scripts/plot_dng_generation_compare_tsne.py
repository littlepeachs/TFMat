import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from plot_text_embedding_tsne import (
    DATASET_SPECS,
    ROOT,
    SPLIT_STYLES,
    build_dataset,
    encode_dataset,
    load_model,
    run_tsne,
    sample_dataset,
)


def parse_args():
    baseline_spec = DATASET_SPECS["mp_20_dng"]
    parser = argparse.ArgumentParser(
        description="Build a shared t-SNE comparison where train/val/test are fixed and only generated sets differ."
    )
    parser.add_argument(
        "--run-dir",
        default=str(baseline_spec.run_dir),
        help="Encoder model directory. Defaults to the unconditional MP-20 DNG baseline run.",
    )
    parser.add_argument(
        "--baseline-generated-pt",
        default=str(ROOT / "data/mp_20/generated_dng_mp20_baseline_s100_a10_gen10k.pt"),
        help="Cached baseline generated dataset.",
    )
    parser.add_argument(
        "--text-generated-pt",
        default=str(ROOT / "data/mp_20/generated_dng_mp20_text_guided_s100_a10_gen10k.pt"),
        help="Cached text-guided generated dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "generated_crystals/dng_embedding_tsne_shared_compare_10kgen"),
        help="Directory for the shared t-SNE outputs.",
    )
    parser.add_argument("--max-train-points", type=int, default=-1)
    parser.add_argument("--max-val-points", type=int, default=-1)
    parser.add_argument("--max-test-points", type=int, default=-1)
    parser.add_argument("--max-generated-points", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-value", type=float, default=1.0)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for embedding extraction.",
    )
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="Path to .npz cache for group embeddings and selected indices. "
             "Defaults to <output-dir>/embeddings_cache.npz. "
             "Skip encoding if cache exists.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing embedding cache and recompute from model.",
    )
    return parser.parse_args()


def build_sampled_datasets(model, cfg, spec, generated_paths, split_limits, seed):
    raw = {
        "train": build_dataset(
            cfg,
            model,
            split="train",
            save_path=spec.train_pt,
            csv_path=spec.train_csv,
            embedding_path=None,
            name="train",
        ),
        "val": build_dataset(
            cfg,
            model,
            split="val",
            save_path=spec.val_pt,
            csv_path=spec.val_csv,
            embedding_path=None,
            name="val",
        ),
        "test": build_dataset(
            cfg,
            model,
            split="test",
            save_path=spec.test_pt,
            csv_path=spec.test_csv,
            embedding_path=None,
            name="test",
        ),
        "generated_baseline": build_dataset(
            cfg,
            model,
            split="test",
            save_path=generated_paths["generated_baseline"],
            csv_path=spec.test_csv,
            embedding_path=None,
            name="generated_baseline",
            propless=True,
        ),
        "generated_text": build_dataset(
            cfg,
            model,
            split="test",
            save_path=generated_paths["generated_text"],
            csv_path=spec.test_csv,
            embedding_path=None,
            name="generated_text",
            propless=True,
        ),
    }

    sampled = {}
    selected_indices = {}
    ordered_groups = ("train", "val", "test", "generated_baseline", "generated_text")
    for offset, group in enumerate(ordered_groups):
        limit = split_limits["generated"] if group.startswith("generated") else split_limits[group]
        sampled[group], selected_indices[group] = sample_dataset(
            raw[group],
            max_points=limit,
            seed=seed + offset,
        )
    return sampled, selected_indices


def build_panel_records_per_tsne(
    sampled_datasets,
    selected_indices,
    group_embeddings,
    test_group: str,
    gen_group: str,
    perplexity: float,
    pca_dim: int,
    seed: int,
):
    """Run per-panel t-SNE on only test + one generated group and return a DataFrame."""
    test_emb = group_embeddings[test_group]
    gen_emb = group_embeddings[gen_group]
    merged = np.concatenate([test_emb, gen_emb], axis=0)
    tsne_coords, used_perplexity = run_tsne(merged, perplexity=perplexity, pca_dim=pca_dim, seed=seed)

    rows = []
    cursor = 0
    for group, split_label in [(test_group, "test"), (gen_group, "generated")]:
        dataset = sampled_datasets[group]
        base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
        subset_indices = selected_indices[group]
        count = group_embeddings[group].shape[0]
        group_coords = tsne_coords[cursor: cursor + count]
        cursor += count
        for local_idx, base_idx in enumerate(subset_indices):
            cached = base_dataset.cached_data[int(base_idx)]
            rows.append(
                {
                    "group": group,
                    "split": split_label,
                    "source_index": int(base_idx),
                    "mp_id": str(cached.get("mp_id", "")),
                    "tsne_x": float(group_coords[local_idx, 0]),
                    "tsne_y": float(group_coords[local_idx, 1]),
                }
            )
    return pd.DataFrame(rows), used_perplexity


def _jsd_2d(a: np.ndarray, b: np.ndarray, bins: int = 60, eps: float = 1e-8) -> float:
    """Jensen-Shannon Divergence in 2-D t-SNE space (base-2, in [0, 1]).

    bins=60 and eps=1e-8 match the original compute_dng_gen_test_metrics.py.
    """
    xmin = min(a[:, 0].min(), b[:, 0].min())
    xmax = max(a[:, 0].max(), b[:, 0].max())
    ymin = min(a[:, 1].min(), b[:, 1].min())
    ymax = max(a[:, 1].max(), b[:, 1].max())
    edges = [np.linspace(xmin, xmax, bins + 1), np.linspace(ymin, ymax, bins + 1)]
    ha, _ = np.histogramdd(a, bins=edges)
    hb, _ = np.histogramdd(b, bins=edges)
    pa = ha.astype(float) + eps
    pb = hb.astype(float) + eps
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    jsd = 0.5 * (pa * np.log(pa / m) + pb * np.log(pb / m)).sum()
    jsd /= np.log(2)
    return float(np.clip(jsd, 0.0, 1.0))


# Visual styles for the two-split (test / generated) single-panel plots.
# Saturation is intentionally kept moderate so the overlap region is readable.
_PANEL_STYLES = {
    "test":      {"color": "#7fc37f", "marker": "^", "alpha": 0.60, "size": 45, "zorder": 3, "linewidths": 0.5, "edgecolors": "#2e7d32"},
    "generated": {"color": "#e88a8c", "marker": "X", "alpha": 0.60, "size": 55, "zorder": 4, "linewidths": 0.5, "edgecolors": "#b03030"},
}
_PANEL_LABELS = {"test": "Test", "generated": "Generated"}


def plot_single_panel(df: pd.DataFrame, title: str, output_png: Path) -> None:
    """Draw a single t-SNE figure with only test and generated points."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 20})

    gen_xy  = df[df["split"] == "generated"][["tsne_x", "tsne_y"]].values.astype(np.float32)
    test_xy = df[df["split"] == "test"][["tsne_x", "tsne_y"]].values.astype(np.float32)
    jsd_val = (
        _jsd_2d(gen_xy, test_xy)
        if len(gen_xy) > 0 and len(test_xy) > 0
        else float("nan")
    )

    fig, ax = plt.subplots(figsize=(9, 8), dpi=220)
    handles, labels = [], []
    for split in ("test", "generated"):
        split_df = df[df["split"] == split]
        sty = _PANEL_STYLES[split]
        sc = ax.scatter(
            split_df["tsne_x"],
            split_df["tsne_y"],
            s=sty["size"],
            alpha=sty["alpha"],
            c=sty["color"],
            marker=sty["marker"],
            linewidths=sty["linewidths"],
            edgecolors=sty["edgecolors"],
            zorder=sty["zorder"],
            label=_PANEL_LABELS[split],
        )
        handles.append(sc)
        labels.append(_PANEL_LABELS[split])

    counts = df["split"].value_counts().to_dict()
    ax.text(
        0.03, 0.98,
        "test={}\ngen={}".format(counts.get("test", 0), counts.get("generated", 0)),
        transform=ax.transAxes, va="top", ha="left", fontsize=16,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.text(
        0.97, 0.98,
        f"JSD = {jsd_val:.4f}",
        transform=ax.transAxes, va="top", ha="right",
        fontsize=20, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fff8e7", "alpha": 0.95, "edgecolor": "#c8a000", "linewidth": 1.5},
    )

    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=20)
    ax.set_ylabel("t-SNE 2", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(False)
    ax.legend(
        handles, labels, fontsize=20, markerscale=1.5,
        frameon=True, framealpha=0.9, edgecolor="none",
    )
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    spec = DATASET_SPECS["mp_20_dng"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = {
        "generated_baseline": Path(args.baseline_generated_pt).resolve(),
        "generated_text": Path(args.text_generated_pt).resolve(),
    }
    for key, path in generated_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key} cache: {path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── embedding cache ───────────────────────────────────────────────────────
    cache_path = Path(
        args.embedding_cache
        if args.embedding_cache
        else str(output_dir / "embeddings_cache.npz")
    )
    _GROUPS = ("train", "val", "test", "generated_baseline", "generated_text")

    if cache_path.exists() and not args.force_recompute:
        print(f"Loading embeddings from cache: {cache_path}")
        raw_cache = np.load(cache_path, allow_pickle=True)
        group_embeddings = {g: raw_cache[g] for g in _GROUPS}
        selected_indices = {g: raw_cache[f"{g}_indices"].tolist() for g in _GROUPS}
        # Still need sampled_datasets for metadata lookup (fast – no encoding)
        split_limits = {
            "train": int(args.max_train_points),
            "val": int(args.max_val_points),
            "test": int(args.max_test_points),
            "generated": int(args.max_generated_points),
        }
        device = torch.device(args.device)
        model, _, cfg = load_model(Path(args.run_dir).resolve(), load_data=False)
        model.eval()
        model.to(device)
        sampled_datasets, _ = build_sampled_datasets(
            model, cfg, spec, generated_paths, split_limits, seed=args.seed
        )
    else:
        device = torch.device(args.device)
        model, _, cfg = load_model(Path(args.run_dir).resolve(), load_data=False)
        model.eval()
        model.to(device)

        split_limits = {
            "train": int(args.max_train_points),
            "val": int(args.max_val_points),
            "test": int(args.max_test_points),
            "generated": int(args.max_generated_points),
        }
        sampled_datasets, selected_indices = build_sampled_datasets(
            model, cfg, spec, generated_paths, split_limits, seed=args.seed,
        )

        group_embeddings = {}
        for group in _GROUPS:
            group_embeddings[group] = encode_dataset(
                model,
                sampled_datasets[group],
                batch_size=args.batch_size,
                device=device,
                time_value=args.time_value,
            )

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            **{g: group_embeddings[g] for g in _GROUPS},
            **{f"{g}_indices": np.array(selected_indices[g]) for g in _GROUPS},
        )
        print(f"Saved embeddings cache: {cache_path}")

    # ── per-panel t-SNE (test + one generated group each) ─────────────────────
    baseline_df, perp_baseline = build_panel_records_per_tsne(
        sampled_datasets, selected_indices, group_embeddings,
        test_group="test", gen_group="generated_baseline",
        perplexity=args.perplexity, pca_dim=args.pca_dim, seed=args.seed,
    )
    text_df, perp_text = build_panel_records_per_tsne(
        sampled_datasets, selected_indices, group_embeddings,
        test_group="test", gen_group="generated_text",
        perplexity=args.perplexity, pca_dim=args.pca_dim, seed=args.seed,
    )

    png_baseline = output_dir / "mp_20_dng_baseline_tsne.png"
    png_text     = output_dir / "mp_20_dng_text_tsne.png"
    csv_baseline = output_dir / "mp_20_dng_baseline_points.csv"
    csv_text     = output_dir / "mp_20_dng_text_points.csv"
    meta_path    = output_dir / "mp_20_dng_per_panel_meta.json"

    plot_single_panel(baseline_df, "Unconditional DNG baseline", png_baseline)
    plot_single_panel(text_df,     "Text-guided DNG",            png_text)
    baseline_df.to_csv(csv_baseline, index=False)
    text_df.to_csv(csv_text, index=False)

    metadata = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "embedding_cache": str(cache_path),
        "tsne_note": "Each panel uses its own per-panel t-SNE (test + gen only), matching the original compute_dng_gen_test_metrics approach.",
        "jsd_note": "JSD computed with bins=60 and eps=1e-8 to match compute_dng_gen_test_metrics.py.",
        "generated_sources": {key: str(path) for key, path in generated_paths.items()},
        "split_limits": split_limits,
        "batch_size": int(args.batch_size),
        "time_value": float(args.time_value),
        "perplexity": {"baseline": float(perp_baseline), "text": float(perp_text)},
        "counts": {
            group: int(group_embeddings[group].shape[0]) for group in _GROUPS
        },
        "outputs": {
            "png_baseline": str(png_baseline),
            "png_text": str(png_text),
            "csv_baseline": str(csv_baseline),
            "csv_text": str(csv_text),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"png_baseline={png_baseline}")
    print(f"png_text={png_text}")
    print(f"csv_baseline={csv_baseline}")
    print(f"csv_text={csv_text}")
    print(f"meta={meta_path}")
    print(f"counts={metadata['counts']}")


if __name__ == "__main__":
    main()