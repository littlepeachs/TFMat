import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from compute_metrics import Crystal


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sanitize_token(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "sample"


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def ensure_eval_axis(value, sample_ndim: int):
    array = tensor_to_numpy(value)
    if array.ndim == sample_ndim:
        return np.expand_dims(array, axis=0)
    return array


def structure_to_crystal_dict(structure: Structure):
    atom_types = np.array([site.specie.Z for site in structure], dtype=int)
    return {
        "frac_coords": np.array(structure.frac_coords),
        "atom_types": atom_types,
        "lengths": np.array(structure.lattice.abc),
        "angles": np.array(structure.lattice.angles),
    }


def build_pred_dict(frac_coords_array, atom_types_array, lengths_array, angles_array, eval_idx: int, index: int, start: int, n_atoms: int):
    frac_coords = frac_coords_array[eval_idx][start:start + n_atoms]
    atom_types = atom_types_array[eval_idx][start:start + n_atoms]
    lengths = lengths_array[eval_idx][index]
    angles = angles_array[eval_idx][index]
    return {
        "frac_coords": frac_coords,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
    }

def compute_lattice_metrics(gt_structure: Structure, pred_structure: Structure):
    gt_lengths = np.array(gt_structure.lattice.abc, dtype=float)
    pred_lengths = np.array(pred_structure.lattice.abc, dtype=float)
    gt_angles = np.array(gt_structure.lattice.angles, dtype=float)
    pred_angles = np.array(pred_structure.lattice.angles, dtype=float)

    length_abs_diff = np.abs(pred_lengths - gt_lengths)
    safe_gt_lengths = np.where(np.abs(gt_lengths) < 1e-8, 1.0, gt_lengths)
    length_rel_diff = length_abs_diff / safe_gt_lengths
    angle_abs_diff = np.abs(pred_angles - gt_angles)
    length_rmse = np.sqrt(np.mean(np.square(pred_lengths - gt_lengths)))
    length_rel_rmse = np.sqrt(np.mean(np.square(length_rel_diff)))
    angle_rmse = np.sqrt(np.mean(np.square(pred_angles - gt_angles)))

    return {
        "lattice_length_mae": float(length_abs_diff.mean()),
        "lattice_length_rel_mae": float(length_rel_diff.mean()),
        "lattice_length_rmse": float(length_rmse),
        "lattice_length_rel_rmse": float(length_rel_rmse),
        "lattice_angle_mae": float(angle_abs_diff.mean()),
        "lattice_angle_rmse": float(angle_rmse),
    }

def metric_or_inf(value):
    if value is None or value == "":
        return float("inf")
    return float(value)


def score_rows_by_dense_ranks(rows, fields):
    if not rows:
        return []

    scored_rows = [dict(row) for row in rows]
    rank_arrays = []
    for field in fields:
        if field == "gt_num_atoms":
            values = pd.Series([float(row[field]) for row in rows], dtype=float)
        else:
            values = pd.Series([metric_or_inf(row[field]) for row in rows], dtype=float)
        rank_arrays.append(values.rank(method="dense", ascending=True).to_numpy())

    for idx, row in enumerate(scored_rows):
        row["selection_score"] = float(sum(rank_array[idx] for rank_array in rank_arrays))

    return scored_rows


def get_text_field(row: pd.Series):
    for key in ("text2", "text", "description", "prompt"):
        if key in row.index and pd.notna(row[key]):
            return str(row[key])
    return ""


def get_material_id(payload, index: int):
    material_ids = payload.get("material_id")
    if material_ids is None or len(material_ids) <= index:
        return ""
    return str(material_ids[index])


def write_manifest(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Path to eval_diff_*.pt file")
    parser.add_argument("--test-csv", required=True, help="Path to aligned test csv")
    parser.add_argument("--output-dir", required=True, help="Directory for manifests and CIF outputs")
    parser.add_argument("--top-k", type=int, default=4, help="Number of examples to export")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to inspect")
    parser.add_argument(
        "--selection-mode",
        choices=("high_atoms", "lattice_match", "small_atoms_low_error"),
        default="high_atoms",
        help="How to rank selected examples",
    )
    args = parser.parse_args()

    pt_path = Path(args.pt)
    test_csv = Path(args.test_csv)
    output_dir = Path(args.output_dir)
    gt_dir = output_dir / "selected_gt_cif"
    pred_dir = output_dir / "selected_generated_cif"

    ensure_dir(output_dir)
    ensure_dir(gt_dir)
    ensure_dir(pred_dir)

    test_df = pd.read_csv(test_csv).reset_index(drop=True)
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)

    frac_coords_array = ensure_eval_axis(payload["frac_coords"], sample_ndim=2)
    atom_types_array = ensure_eval_axis(payload["atom_types"], sample_ndim=1)
    lengths_array = ensure_eval_axis(payload["lengths"], sample_ndim=2)
    angles_array = ensure_eval_axis(payload["angles"], sample_ndim=2)
    num_atoms_array = ensure_eval_axis(payload["num_atoms"], sample_ndim=1)

    n_evals, n_pred = num_atoms_array.shape
    limit = min(len(test_df), n_pred) if args.limit is None else min(args.limit, len(test_df), n_pred)
    if limit <= 0:
        raise ValueError("No aligned samples available")

    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    all_rows = []
    matched_rows = []
    valid_rows = []
    best_pred_by_index = {}

    cursors = [0 for _ in range(n_evals)]
    for index in range(limit):
        row = test_df.iloc[index]
        gt_structure = Structure.from_str(row["cif"], fmt="cif")
        gt = Crystal(structure_to_crystal_dict(gt_structure), compute_fp=False)

        best_key = None
        best_pred = None
        best_pred_n_atoms = 0
        best_rms_dist = None
        best_is_match = False
        best_eval_idx = -1
        for eval_idx in range(n_evals):
            pred_n_atoms = int(num_atoms_array[eval_idx, index])
            pred_dict = build_pred_dict(
                frac_coords_array,
                atom_types_array,
                lengths_array,
                angles_array,
                eval_idx,
                index,
                cursors[eval_idx],
                pred_n_atoms,
            )
            cursors[eval_idx] += pred_n_atoms

            pred = Crystal(pred_dict, compute_fp=False)
            rms_dist = None
            is_match = False
            if pred.valid and gt.valid:
                try:
                    rms = matcher.get_rms_dist(pred.structure, gt_structure)
                    if rms is not None:
                        rms_dist = float(rms[0])
                        is_match = True
                except Exception:
                    rms_dist = None

            candidate_key = (
                0 if is_match else 1,
                0 if pred.valid else 1,
                float(rms_dist) if rms_dist is not None else float("inf"),
                eval_idx,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_pred = pred
                best_pred_n_atoms = pred_n_atoms
                best_rms_dist = rms_dist
                best_is_match = is_match
                best_eval_idx = eval_idx

        if best_pred is None:
            raise RuntimeError(f"Failed to decode predictions for sample index {index}")

        best_pred_by_index[index] = best_pred

        gt_n_atoms = len(gt_structure)
        atom_count_match = best_pred_n_atoms == gt_n_atoms
        lattice_metrics = {
            "lattice_length_mae": None,
            "lattice_length_rel_mae": None,
            "lattice_length_rmse": None,
            "lattice_length_rel_rmse": None,
            "lattice_angle_mae": None,
            "lattice_angle_rmse": None,
        }
        if best_pred.constructed:
            lattice_metrics = compute_lattice_metrics(gt_structure, best_pred.structure)

        material_id = str(row["material_id"]) if "material_id" in row.index else f"idx{index:05d}"
        gen_material_id = get_material_id(payload, index)
        pred_formula = best_pred.structure.composition.reduced_formula if best_pred.constructed else ""
        gt_formula = gt_structure.composition.reduced_formula
        record = {
            "rank_key_num_atoms": gt_n_atoms,
            "index": index,
            "material_id": material_id,
            "gen_material_id": gen_material_id,
            "aligned_material_id": material_id == gen_material_id if gen_material_id else "",
            "gt_formula": gt_formula,
            "pred_formula": pred_formula,
            "gt_num_atoms": gt_n_atoms,
            "pred_num_atoms": best_pred_n_atoms,
            "atom_count_match": bool(atom_count_match),
            "best_eval_idx": best_eval_idx,
            "comp_valid": bool(best_pred.comp_valid),
            "struct_valid": bool(best_pred.struct_valid),
            "valid": bool(best_pred.valid),
            "match": bool(best_is_match),
            "rms_dist": best_rms_dist,
            "lattice_length_mae": lattice_metrics["lattice_length_mae"],
            "lattice_length_rel_mae": lattice_metrics["lattice_length_rel_mae"],
            "lattice_length_rmse": lattice_metrics["lattice_length_rmse"],
            "lattice_length_rel_rmse": lattice_metrics["lattice_length_rel_rmse"],
            "lattice_angle_mae": lattice_metrics["lattice_angle_mae"],
            "lattice_angle_rmse": lattice_metrics["lattice_angle_rmse"],
            "text": get_text_field(row),
        }
        all_rows.append(record)

        if best_pred.valid:
            valid_rows.append(record)
        if best_pred.valid and best_is_match:
            matched_rows.append(record)

    if args.selection_mode == "lattice_match":
        matched_rows.sort(
            key=lambda item: (
                0 if item["atom_count_match"] else 1,
                metric_or_inf(item["lattice_length_rel_mae"]),
                metric_or_inf(item["rms_dist"]),
                metric_or_inf(item["lattice_angle_mae"]),
                item["index"],
            )
        )
        valid_rows.sort(
            key=lambda item: (
                0 if item["atom_count_match"] else 1,
                metric_or_inf(item["lattice_length_rel_mae"]),
                metric_or_inf(item["lattice_angle_mae"]),
                item["index"],
            )
        )
        matched_pool = [item for item in matched_rows if item["atom_count_match"]]
        valid_pool = [item for item in valid_rows if item["atom_count_match"]]
    elif args.selection_mode == "small_atoms_low_error":
        matched_pool = [item for item in matched_rows if item["atom_count_match"]]
        valid_pool = [item for item in valid_rows if item["atom_count_match"]]
        matched_pool = score_rows_by_dense_ranks(
            matched_pool,
            ("gt_num_atoms", "rms_dist", "lattice_length_rel_rmse"),
        )
        valid_pool = score_rows_by_dense_ranks(
            valid_pool,
            ("gt_num_atoms", "lattice_length_rel_rmse", "lattice_angle_rmse"),
        )
        matched_pool.sort(
            key=lambda item: (
                item["selection_score"],
                item["gt_num_atoms"],
                metric_or_inf(item["rms_dist"]),
                metric_or_inf(item["lattice_length_rel_rmse"]),
                metric_or_inf(item["lattice_angle_rmse"]),
                item["index"],
            )
        )
        valid_pool.sort(
            key=lambda item: (
                item["selection_score"],
                item["gt_num_atoms"],
                metric_or_inf(item["lattice_length_rel_rmse"]),
                metric_or_inf(item["lattice_angle_rmse"]),
                item["index"],
            )
        )
    else:
        matched_rows.sort(key=lambda item: (-item["gt_num_atoms"], metric_or_inf(item["rms_dist"]), item["index"]))
        valid_rows.sort(key=lambda item: (-item["gt_num_atoms"], item["index"]))
        matched_pool = matched_rows
        valid_pool = valid_rows

    selected_rows = matched_pool[:args.top_k]
    if len(selected_rows) < args.top_k:
        selected_indices = {item["index"] for item in selected_rows}
        for item in valid_pool:
            if item["index"] in selected_indices:
                continue
            selected_rows.append(item)
            selected_indices.add(item["index"])
            if len(selected_rows) == args.top_k:
                break

        if len(selected_rows) < args.top_k:
            for item in valid_rows:
                if item["index"] in selected_indices:
                    continue
                selected_rows.append(item)
                selected_indices.add(item["index"])
                if len(selected_rows) == args.top_k:
                    break

    if not selected_rows:
        raise RuntimeError("No valid samples were selected")

    selected_rows.sort(
        key=lambda item: (
            0 if item["atom_count_match"] else 1,
            metric_or_inf(item["lattice_length_rel_mae"]),
            metric_or_inf(item["rms_dist"]),
            metric_or_inf(item["lattice_angle_mae"]),
            item["index"],
        ) if args.selection_mode == "lattice_match" else (
            item.get("selection_score", float("inf")),
            item["gt_num_atoms"],
            metric_or_inf(item["rms_dist"]),
            metric_or_inf(item["lattice_length_rel_rmse"]),
            metric_or_inf(item["lattice_angle_rmse"]),
            item["index"],
        ) if args.selection_mode == "small_atoms_low_error" else (
            -item["gt_num_atoms"],
            metric_or_inf(item["rms_dist"]),
            item["index"],
        )
    )

    write_manifest(output_dir / "all_manifest.csv", all_rows)
    write_manifest(output_dir / "matched_manifest.csv", matched_rows)
    write_manifest(output_dir / "selected_manifest.csv", selected_rows)

    for rank, item in enumerate(selected_rows, start=1):
        index = int(item["index"])
        row = test_df.iloc[index]
        gt_structure = Structure.from_str(row["cif"], fmt="cif")
        pred = best_pred_by_index[index]
        if not pred.constructed:
            continue

        material_id = sanitize_token(item["material_id"])
        gt_formula = sanitize_token(item["gt_formula"])
        pred_formula = sanitize_token(item["pred_formula"] or "pred")
        rms_token = sanitize_token("nomatch" if item["rms_dist"] is None else f"{item['rms_dist']:.6f}")
        stem = f"{rank:02d}_{index:05d}_{material_id}_na{item['gt_num_atoms']}_rms{rms_token}"

        CifWriter(gt_structure).write_file(gt_dir / f"{stem}_{gt_formula}.cif")
        CifWriter(pred.structure).write_file(pred_dir / f"{stem}_{pred_formula}.cif")

    print(f"aligned_rows={limit}")
    print(f"matched_rows={len(matched_rows)}")
    print(f"selected_rows={len(selected_rows)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()