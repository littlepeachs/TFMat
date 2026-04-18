import argparse
import copy
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter

from diffcsp.common.data_utils import lattice_params_to_matrix_torch, lattice_polar_build_torch
from eval_utils import load_model


ROOT = Path(__file__).resolve().parents[1]

SPLIT_STYLES = {
    "train": {"color": "#93a8bf", "marker": "o", "alpha": 0.16, "size": 10, "zorder": 1},
    "val": {"color": "#f28e2b", "marker": "s", "alpha": 0.82, "size": 22, "zorder": 3},
    "test": {"color": "#59a14f", "marker": "^", "alpha": 0.84, "size": 22, "zorder": 4},
    "generated": {"color": "#e15759", "marker": "X", "alpha": 0.88, "size": 28, "zorder": 5},
}


@dataclass(frozen=True)
class DatasetSpec:
    plot_title: str
    output_stem: str
    model_family: str
    conditioning: str
    run_dir: Path
    train_pt: Path
    val_pt: Path
    test_pt: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    train_embedding_pt: Path | None
    val_embedding_pt: Path | None
    test_embedding_pt: Path | None
    generated_embedding_pt: Path | None
    generated_propless: bool
    generated_candidates: tuple[Path, ...]


DATASET_SPECS = {
    "carbon_24": DatasetSpec(
        plot_title="carbon_24 text-model crystal embeddings",
        output_stem="carbon_24_text",
        model_family="CrystalFlow text model",
        conditioning="precomputed text_embedding from data_text/precomputed_embeddings",
        run_dir=ROOT / "hydra_jobs/singlerun/CSP-carbon24-text-gpu1-offline-emb",
        train_pt=ROOT / "data_text/carbon_24/train_text.pt",
        val_pt=ROOT / "data_text/carbon_24/val_text.pt",
        test_pt=ROOT / "data_text/carbon_24/test_text.pt",
        train_csv=ROOT / "data_text/carbon_24/train.csv",
        val_csv=ROOT / "data_text/carbon_24/val.csv",
        test_csv=ROOT / "data_text/carbon_24/test.csv",
        train_embedding_pt=ROOT / "data_text/carbon_24/precomputed_embeddings/train_text2_matscibert_mean.pt",
        val_embedding_pt=ROOT / "data_text/carbon_24/precomputed_embeddings/val_text2_matscibert_mean.pt",
        test_embedding_pt=ROOT / "data_text/carbon_24/precomputed_embeddings/test_text2_matscibert_mean.pt",
        generated_embedding_pt=ROOT / "data_text/carbon_24/precomputed_embeddings/tsne_text2_matscibert_mean_10k_from_all.pt",
        generated_propless=False,
        generated_candidates=(
            ROOT / "data_text/carbon_24/generated_text_gen10k_tsne.pt",
            ROOT / "data_text/carbon_24/test_text_eval_epoch2637_numeval1_s200_a5_coords_gfauto.pt",
            ROOT / "data_text/carbon_24/test_text_eval_epoch2637_numeval1.pt",
            ROOT / "data_text/carbon_24/test_text_eval.pt",
        ),
    ),
    "perov_5": DatasetSpec(
        plot_title="perov_5 text-model crystal embeddings",
        output_stem="perov_5_text",
        model_family="CrystalFlow text model",
        conditioning="precomputed text_embedding from data_text/precomputed_embeddings",
        run_dir=ROOT / "hydra_jobs/singlerun/CSP-perov5-text-gpu7-offline-emb",
        train_pt=ROOT / "data_text/perov_5/train_text.pt",
        val_pt=ROOT / "data_text/perov_5/val_text.pt",
        test_pt=ROOT / "data_text/perov_5/test_text.pt",
        train_csv=ROOT / "data_text/perov_5/train.csv",
        val_csv=ROOT / "data_text/perov_5/val.csv",
        test_csv=ROOT / "data_text/perov_5/test.csv",
        train_embedding_pt=ROOT / "data_text/perov_5/precomputed_embeddings/train_text2_matscibert_mean.pt",
        val_embedding_pt=ROOT / "data_text/perov_5/precomputed_embeddings/val_text2_matscibert_mean.pt",
        test_embedding_pt=ROOT / "data_text/perov_5/precomputed_embeddings/test_text2_matscibert_mean.pt",
        generated_embedding_pt=ROOT / "data_text/perov_5/precomputed_embeddings/tsne_text2_matscibert_mean_10k_from_all.pt",
        generated_propless=False,
        generated_candidates=(
            ROOT / "data_text/perov_5/generated_text_gen10k_tsne.pt",
            ROOT / "data_text/perov_5/test_text_eval.pt",
            ROOT / "data_text/perov_5/test_text_eval_20260331_142520.pt",
        ),
    ),
    "mp_20": DatasetSpec(
        plot_title="mp_20 text-model crystal embeddings",
        output_stem="mp_20_text",
        model_family="CrystalFlow text model",
        conditioning="precomputed text_embedding from data_text/precomputed_embeddings",
        run_dir=ROOT / "hydra_jobs/singlerun/CSP-mp20-text-gpu3-offline-emb",
        train_pt=ROOT / "data_text/mp_20/train_text.pt",
        val_pt=ROOT / "data_text/mp_20/val_text.pt",
        test_pt=ROOT / "data_text/mp_20/test_text.pt",
        train_csv=ROOT / "data_text/mp_20/train.csv",
        val_csv=ROOT / "data_text/mp_20/val.csv",
        test_csv=ROOT / "data_text/mp_20/test.csv",
        train_embedding_pt=ROOT / "data_text/mp_20/precomputed_embeddings/train_text2_matscibert_mean.pt",
        val_embedding_pt=ROOT / "data_text/mp_20/precomputed_embeddings/val_text2_matscibert_mean.pt",
        test_embedding_pt=ROOT / "data_text/mp_20/precomputed_embeddings/test_text2_matscibert_mean.pt",
        generated_embedding_pt=ROOT / "data_text/mp_20/precomputed_embeddings/tsne_text2_matscibert_mean_10k_from_all.pt",
        generated_propless=False,
        generated_candidates=(
            ROOT / "data_text/mp_20/generated_text_gen10k_tsne.pt",
            ROOT / "data_text/mp_20/test_offline_eval_epoch1385_numeval1_text8000_s100_a10_coords_gfauto.pt",
            ROOT / "data_text/mp_20/test_text_eval.pt",
        ),
    ),
    "mp_20_dng": DatasetSpec(
        plot_title="mp_20 DNG crystal embeddings",
        output_stem="mp_20_dng",
        model_family="CrystalFlow DNG baseline model",
        conditioning="none during embedding extraction; unguided de novo generation without composition input",
        run_dir=ROOT / "hydra_jobs/singlerun/DNG-mp20-baseline-gpu2",
        train_pt=ROOT / "data/mp_20/train_ori.pt",
        val_pt=ROOT / "data/mp_20/val_ori.pt",
        test_pt=ROOT / "data/mp_20/test_ori.pt",
        train_csv=ROOT / "data/mp_20/train.csv",
        val_csv=ROOT / "data/mp_20/val.csv",
        test_csv=ROOT / "data/mp_20/test.csv",
        train_embedding_pt=None,
        val_embedding_pt=None,
        test_embedding_pt=None,
        generated_embedding_pt=None,
        generated_propless=True,
        generated_candidates=(
            ROOT / "data/mp_20/generated_dng_mp20_baseline_s100_a10_gen10k.pt",
        ),
    ),
    "mp_20_dng_text": DatasetSpec(
        plot_title="mp_20 DNG text-guided crystal embeddings",
        output_stem="mp_20_dng_text",
        model_family="CrystalFlow DNG text-guided model",
        conditioning="precomputed text_embedding from data_text/mp_20/test.csv prompts; generated structures are loaded from targeted s100_a10_coords_gfauto CIF exports",
        run_dir=ROOT / "hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5",
        train_pt=ROOT / "data_text/mp_20/train_text.pt",
        val_pt=ROOT / "data_text/mp_20/val_text.pt",
        test_pt=ROOT / "data_text/mp_20/test_text.pt",
        train_csv=ROOT / "data_text/mp_20/train.csv",
        val_csv=ROOT / "data_text/mp_20/val.csv",
        test_csv=ROOT / "data_text/mp_20/test.csv",
        train_embedding_pt=ROOT / "data_text/mp_20/precomputed_embeddings/train_text2_matscibert_mean.pt",
        val_embedding_pt=ROOT / "data_text/mp_20/precomputed_embeddings/val_text2_matscibert_mean.pt",
        test_embedding_pt=ROOT / "data_text/mp_20/precomputed_embeddings/test_text2_matscibert_mean.pt",
        generated_embedding_pt=ROOT / "data_text/mp_20/precomputed_embeddings/test_text2_matscibert_mean.pt",
        generated_propless=False,
        generated_candidates=(
            ROOT / "data_text/mp_20/generated_dng_text_lattice5_epoch1834_s100_a10_coords_gfauto_gen10k.pt",
        ),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract text-model crystal embeddings and plot t-SNE.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["carbon_24", "perov_5", "mp_20"],
        choices=list(DATASET_SPECS.keys()),
        help="Datasets to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "generated_crystals/text_embedding_tsne"),
        help="Output directory for figures and coordinate tables.",
    )
    parser.add_argument(
        "--max-train-points",
        type=int,
        default=-1,
        help="Train split cap. Use <=0 to keep the full train set.",
    )
    parser.add_argument(
        "--max-val-points",
        type=int,
        default=-1,
        help="Validation split cap. Use <=0 to keep the full validation set.",
    )
    parser.add_argument(
        "--max-test-points",
        type=int,
        default=-1,
        help="Test split cap. Use <=0 to keep the full test set.",
    )
    parser.add_argument(
        "--max-generated-points",
        type=int,
        default=10000,
        help="Generated split cap. Use <=0 to keep all generated points.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding extraction batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split sampling and t-SNE.")
    parser.add_argument("--time-value", type=float, default=1.0, help="Flow time used when encoding clean crystals.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="Target t-SNE perplexity.")
    parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimension before t-SNE.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for embedding extraction.",
    )
    return parser.parse_args()


def choose_generated_path(spec: DatasetSpec) -> Path:
    for candidate in spec.generated_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No generated text pt found for {spec.run_dir}")


def choose_generated_embedding_path(spec: DatasetSpec, generated_pt: Path) -> Path:
    if spec.generated_embedding_pt is None:
        return None
    if generated_pt == spec.generated_candidates[0] and spec.generated_embedding_pt.exists():
        return spec.generated_embedding_pt
    return spec.test_embedding_pt


def get_dataset_cfg(cfg, split: str):
    if split == "train":
        return copy.deepcopy(cfg.data.datamodule.datasets.train)
    if split == "val":
        return copy.deepcopy(cfg.data.datamodule.datasets.val[0])
    if split == "test":
        return copy.deepcopy(cfg.data.datamodule.datasets.test[0])
    raise ValueError(f"Unsupported split: {split}")


def build_dataset(cfg, model, split: str, save_path: Path, csv_path: Path, embedding_path: Path | None, name: str, propless: bool = False):
    dataset_cfg = get_dataset_cfg(cfg, split)
    dataset_cfg.name = name
    dataset_cfg.path = str(csv_path)
    dataset_cfg.save_path = str(save_path)
    if propless:
        dataset_cfg.prop = None
        dataset_cfg.properties = []
    if embedding_path is not None:
        dataset_cfg.precomputed_text_embedding_path = str(embedding_path)
    dataset = hydra.utils.instantiate(dataset_cfg)
    dataset.lattice_scaler = model.lattice_scaler
    dataset.scaler = model.scaler
    dataset.scalers = model.scalers
    return dataset


def sample_dataset(dataset, max_points: int, seed: int):
    if max_points <= 0 or len(dataset) <= max_points:
        indices = np.arange(len(dataset), dtype=int)
        return dataset, indices

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_points, replace=False))
    return Subset(dataset, indices.tolist()), indices


def get_hook_module(model):
    if getattr(model.decoder, "ln", False) and hasattr(model.decoder, "final_layer_norm"):
        return model.decoder.final_layer_norm
    return model.decoder._modules[f"csp_layer_{model.decoder.num_layers - 1}"]


def encode_batch(model, batch, device: torch.device, time_value: float):
    batch = batch.to(device)
    captured = {}

    def hook(_module, _inputs, output):
        captured["node_features"] = output.detach()

    hook_handle = get_hook_module(model).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            times = torch.full((batch.num_graphs,), time_value, device=device)
            time_emb = model.time_embedding(times)

            if model.guide_threshold is None:
                cemb = None
                guide_indicator = None
            else:
                cemb = model._build_condition_embedding(batch)
                guide_indicator = torch.ones(batch.num_graphs, device=device)

            if model.lattice_polar:
                lattices_rep = batch.lattice_polar
                lattices_mat = lattice_polar_build_torch(lattices_rep)
            else:
                lattices_rep = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
                lattices_mat = lattices_rep

            if model.pred_type:
                if model.type_encoding is None:
                    input_atom_types = F.one_hot(
                        batch.atom_types - 1,
                        num_classes=model.decoder.node_embedding.in_features,
                    ).float()
                else:
                    input_atom_types = model.type_encoding(batch.atom_types).float()
            else:
                input_atom_types = batch.atom_types

            _ = model.decoder(
                t=time_emb,
                atom_types=input_atom_types,
                frac_coords=batch.frac_coords,
                lattices_rep=lattices_rep,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
                lattices_mat=lattices_mat,
                cemb=cemb,
                guide_indicator=guide_indicator,
            )
    finally:
        hook_handle.remove()

    if "node_features" not in captured:
        raise RuntimeError("Failed to capture decoder node features for embedding extraction.")

    graph_features = scatter(captured["node_features"], batch.batch, dim=0, reduce="mean")
    return graph_features.cpu().numpy()


def encode_dataset(model, dataset, batch_size: int, device: torch.device, time_value: float):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in loader:
        embeddings.append(encode_batch(model, batch, device=device, time_value=time_value))
    return np.concatenate(embeddings, axis=0)


def build_split_datasets(spec: DatasetSpec, model, cfg, seed: int, split_limits: dict[str, int]):
    generated_pt = choose_generated_path(spec)
    generated_embedding_pt = choose_generated_embedding_path(spec, generated_pt)
    raw_datasets = {
        "train": build_dataset(
            cfg,
            model,
            split="train",
            save_path=spec.train_pt,
            csv_path=spec.train_csv,
            embedding_path=spec.train_embedding_pt,
            name="train",
        ),
        "val": build_dataset(
            cfg,
            model,
            split="val",
            save_path=spec.val_pt,
            csv_path=spec.val_csv,
            embedding_path=spec.val_embedding_pt,
            name="val",
        ),
        "test": build_dataset(
            cfg,
            model,
            split="test",
            save_path=spec.test_pt,
            csv_path=spec.test_csv,
            embedding_path=spec.test_embedding_pt,
            name="test",
        ),
        "generated": build_dataset(
            cfg,
            model,
            split="test",
            save_path=generated_pt,
            csv_path=spec.test_csv,
            embedding_path=generated_embedding_pt,
            name="generated",
            propless=spec.generated_propless,
        ),
    }

    sampled = {}
    selected_indices = {}
    for offset, split in enumerate(("train", "val", "test", "generated")):
        sampled[split], selected_indices[split] = sample_dataset(
            raw_datasets[split],
            max_points=split_limits[split],
            seed=seed + offset,
        )
    return sampled, selected_indices, generated_pt


def run_tsne(embeddings: np.ndarray, perplexity: float, pca_dim: int, seed: int):
    if embeddings.shape[0] < 3:
        raise ValueError("Need at least 3 samples for t-SNE.")

    reduced = embeddings
    if pca_dim > 0 and embeddings.shape[1] > pca_dim and embeddings.shape[0] > pca_dim:
        reduced = PCA(n_components=pca_dim, random_state=seed).fit_transform(embeddings)

    max_perplexity = max(5.0, min(perplexity, float((embeddings.shape[0] - 1) // 3)))
    tsne_kwargs = {
        "n_components": 2,
        "init": "pca",
        "learning_rate": "auto",
        "perplexity": max_perplexity,
        "random_state": seed,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_kwargs["max_iter"] = 1500
    else:
        tsne_kwargs["n_iter"] = 1500

    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(reduced), max_perplexity


def build_records(sampled_datasets, selected_indices, split_embeddings, tsne_coords):
    rows = []
    cursor = 0
    for split in ("train", "val", "test", "generated"):
        dataset = sampled_datasets[split]
        base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        subset_indices = selected_indices[split]
        count = split_embeddings[split].shape[0]
        split_coords = tsne_coords[cursor:cursor + count]
        cursor += count
        for local_idx, base_idx in enumerate(subset_indices):
            cached = base_dataset.cached_data[int(base_idx)]
            rows.append(
                {
                    "split": split,
                    "source_index": int(base_idx),
                    "mp_id": str(cached.get("mp_id", "")),
                    "text2": str(cached.get("text2", "")),
                    "tsne_x": float(split_coords[local_idx, 0]),
                    "tsne_y": float(split_coords[local_idx, 1]),
                }
            )
    return pd.DataFrame(rows)


def plot_dataset(df: pd.DataFrame, title: str, output_png: Path):
    plt.figure(figsize=(10, 8), dpi=180)
    for split in ("train", "val", "test", "generated"):
        split_df = df[df["split"] == split]
        style = SPLIT_STYLES[split]
        plt.scatter(
            split_df["tsne_x"],
            split_df["tsne_y"],
            s=style["size"],
            alpha=style["alpha"],
            c=style["color"],
            marker=style["marker"],
            linewidths=0,
            zorder=style["zorder"],
            label=f"{split} (n={len(split_df)})",
        )
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()


def embedding_description(spec: DatasetSpec, time_value: float):
    return {
        "model_family": spec.model_family,
        "conditioning": spec.conditioning,
        "decoder_input": f"clean crystal graph encoded at fixed flow time t={time_value}",
        "node_feature_capture": "decoder final_layer_norm output; if layer norm is absent, use the last CSP layer output",
        "graph_pooling": "mean pool node features over node2graph with torch_scatter.scatter(..., reduce='mean')",
        "embedding_dim": 512,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    summary = {}
    split_limits = {
        "train": int(args.max_train_points),
        "val": int(args.max_val_points),
        "test": int(args.max_test_points),
        "generated": int(args.max_generated_points),
    }

    for dataset_name in args.datasets:
        spec = DATASET_SPECS[dataset_name]
        model, _, cfg = load_model(spec.run_dir.resolve(), load_data=False)
        model.eval()
        model.to(device)

        sampled_datasets, selected_indices, generated_pt = build_split_datasets(
            spec,
            model,
            cfg,
            seed=args.seed,
            split_limits=split_limits,
        )

        split_embeddings = {}
        for split in ("train", "val", "test", "generated"):
            split_embeddings[split] = encode_dataset(
                model,
                sampled_datasets[split],
                batch_size=args.batch_size,
                device=device,
                time_value=args.time_value,
            )

        merged_embeddings = np.concatenate(
            [split_embeddings[split] for split in ("train", "val", "test", "generated")],
            axis=0,
        )
        tsne_coords, used_perplexity = run_tsne(
            merged_embeddings,
            perplexity=args.perplexity,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )

        records = build_records(sampled_datasets, selected_indices, split_embeddings, tsne_coords)

        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = dataset_output_dir / f"{spec.output_stem}_tsne_points.csv"
        png_path = dataset_output_dir / f"{spec.output_stem}_tsne.png"
        meta_path = dataset_output_dir / f"{spec.output_stem}_tsne_meta.json"

        records.to_csv(csv_path, index=False)
        plot_dataset(records, title=spec.plot_title, output_png=png_path)

        metadata = {
            "dataset": dataset_name,
            "run_dir": str(spec.run_dir),
            "generated_pt": str(generated_pt),
            "train_pt": str(spec.train_pt),
            "val_pt": str(spec.val_pt),
            "test_pt": str(spec.test_pt),
            "split_limits": split_limits,
            "batch_size": int(args.batch_size),
            "time_value": float(args.time_value),
            "perplexity": float(used_perplexity),
            "embedding_method": embedding_description(spec, args.time_value),
            "counts": {
                split: int(split_embeddings[split].shape[0])
                for split in ("train", "val", "test", "generated")
            },
        }
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        summary[dataset_name] = metadata

        print(f"[{dataset_name}] generated_pt={generated_pt}")
        print(f"[{dataset_name}] counts={metadata['counts']}")
        print(f"[{dataset_name}] png={png_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()