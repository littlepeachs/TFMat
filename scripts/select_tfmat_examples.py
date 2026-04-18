import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from compute_metrics import Crystal


def build_crystal_dict(frac_coords, atom_types, lengths, angles):
    return {
        "frac_coords": frac_coords,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
    }


def structure_to_crystal_dict(structure: Structure):
    atom_types = np.array([site.specie.Z for site in structure], dtype=int)
    return {
        "frac_coords": np.array(structure.frac_coords),
        "atom_types": atom_types,
        "lengths": np.array(structure.lattice.abc),
        "angles": np.array(structure.lattice.angles),
    }


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Path to eval_gen_*.pt")
    parser.add_argument("--test-csv", required=True, help="Path to MP-20 test.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for manifests and shortlisted CIFs")
    parser.add_argument("--limit", type=int, default=8000, help="Number of aligned test samples to inspect")
    parser.add_argument("--max-elements", type=int, default=3, help="Maximum number of distinct elements for shortlist")
    parser.add_argument("--top-k", type=int, default=8, help="Number of final examples to keep")
    args = parser.parse_args()

    pt_path = Path(args.pt)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    shortlist_gen_dir = out_dir / "top8_generated_cif"
    shortlist_gt_dir = out_dir / "top8_gt_cif"
    ensure_dir(shortlist_gen_dir)
    ensure_dir(shortlist_gt_dir)

    test_df = pd.read_csv(args.test_csv).reset_index(drop=True)
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)

    if len(test_df) < args.limit:
        raise ValueError(f"test.csv has only {len(test_df)} rows, limit={args.limit} is too large")
    if len(payload["num_atoms"]) < args.limit:
        raise ValueError(f"generated file has only {len(payload['num_atoms'])} crystals, limit={args.limit} is too large")

    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    frac_coords = payload["frac_coords"]
    atom_types = payload["atom_types"]
    lengths = payload["lengths"]
    angles = payload["angles"]
    num_atoms = payload["num_atoms"]
    material_ids = list(map(str, payload["material_id"]))

    valid_rows = []
    matched_rows = []
    all_rows = []

    start = 0
    for idx in range(args.limit):
        n_atoms = int(num_atoms[idx].item())
        pred_dict = build_crystal_dict(
            frac_coords[start:start + n_atoms].numpy(),
            atom_types[start:start + n_atoms].numpy(),
            lengths[idx].numpy(),
            angles[idx].numpy(),
        )
        start += n_atoms

        pred = Crystal(pred_dict, compute_fp=False)
        row = test_df.iloc[idx]
        gt_mid = str(row["material_id"])
        gen_mid = material_ids[idx]
        gt_structure = Structure.from_str(row["cif"], fmt="cif")
        gt = Crystal(structure_to_crystal_dict(gt_structure), compute_fp=False)

        pred_formula = pred.structure.composition.reduced_formula if pred.constructed else ""
        gt_formula = gt.structure.composition.reduced_formula if gt.constructed else str(row["pretty_formula"])
        num_elements = len(pred.elems)
        rms_dist = None
        is_match = False
        if pred.valid and gt.valid:
            try:
                rms = matcher.get_rms_dist(pred.structure, gt.structure)
                if rms is not None:
                    rms_dist = float(rms[0])
                    is_match = True
            except Exception:
                rms_dist = None

        record = {
            "index": idx,
            "material_id": gt_mid,
            "gen_material_id": gen_mid,
            "aligned_material_id": gt_mid == gen_mid,
            "gt_formula": gt_formula,
            "pred_formula": pred_formula,
            "num_elements": num_elements,
            "comp_valid": bool(pred.comp_valid),
            "struct_valid": bool(pred.struct_valid),
            "valid": bool(pred.valid),
            "match": bool(is_match),
            "rms_dist": "" if rms_dist is None else f"{rms_dist:.6f}",
            "text2": str(row["text2"]),
        }
        all_rows.append(record)

        if pred.valid and num_elements <= args.max_elements:
            valid_rows.append(record)
        if pred.valid and num_elements <= args.max_elements and is_match:
            matched_rows.append(record)

    matched_rows.sort(key=lambda x: (float(x["rms_dist"]), x["index"]))
    valid_rows.sort(key=lambda x: (x["num_elements"], x["index"]))

    shortlist_rows = matched_rows[:args.top_k]

    write_manifest(out_dir / "all_first8000_manifest.csv", all_rows)
    write_manifest(out_dir / "valid_leq3_manifest.csv", valid_rows)
    write_manifest(out_dir / "matched_leq3_manifest.csv", matched_rows)
    write_manifest(out_dir / "top8_matched_manifest.csv", shortlist_rows)

    for rank, item in enumerate(shortlist_rows, start=1):
        idx = int(item["index"])
        gen_formula = item["pred_formula"].replace(" ", "")
        gt_formula = item["gt_formula"].replace(" ", "")
        gen_src = out_dir.parent / "all_cif" / f"{idx:05d}_{item['material_id']}_{gen_formula}.cif"
        gen_dst = shortlist_gen_dir / f"{rank:02d}_{idx:05d}_{item['material_id']}_{gen_formula}.cif"
        if not gen_src.exists():
            raise FileNotFoundError(f"Missing generated CIF: {gen_src}")
        gen_dst.write_bytes(gen_src.read_bytes())

        gt_row = test_df.iloc[idx]
        gt_structure = Structure.from_str(gt_row["cif"], fmt="cif")
        gt_dst = shortlist_gt_dir / f"{rank:02d}_{idx:05d}_{item['material_id']}_{gt_formula}.cif"
        gt_structure.to(filename=str(gt_dst))

    print(f"aligned_rows={args.limit}")
    print(f"valid_leq{args.max_elements}={len(valid_rows)}")
    print(f"matched_leq{args.max_elements}={len(matched_rows)}")
    print(f"topk_saved={len(shortlist_rows)}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
