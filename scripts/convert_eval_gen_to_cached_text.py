import argparse
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure

from diffcsp.common.data_utils import add_scaled_lattice_prop, preprocess_tensors
from eval_utils import get_crystals_list


def parse_args():
    parser = argparse.ArgumentParser(description="Convert eval_gen*.pt into cached list-format data.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--eval-gen", help="Path to eval_gen*.pt produced by scripts/generation.py")
    source_group.add_argument("--cif-dir", help="Directory containing generated CIF files named as <index>_<material_id>_<formula>.cif")
    parser.add_argument("--output", required=True, help="Output path for cached list-format .pt")
    parser.add_argument(
        "--lookup-pt",
        nargs="+",
        help="Optional cached dataset pt files used to recover text2 and optional properties by mp_id.",
    )
    parser.add_argument(
        "--mp-id-prefix",
        default="generated",
        help="Synthetic mp_id prefix used when eval_gen payload does not contain material_id.",
    )
    parser.add_argument("--graph-method", default="crystalnn")
    parser.add_argument("--lattice-scale-method", default="scale_length")
    parser.add_argument("--niggli", action="store_true", default=True)
    parser.add_argument("--primitive", action="store_true", default=False)
    return parser.parse_args()


def load_from_eval_gen(path: str, mp_id_prefix: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    crystal_array_list = get_crystals_list(
        payload["frac_coords"],
        payload["atom_types"],
        payload["lengths"],
        payload["angles"],
        payload["num_atoms"],
    )

    material_ids = payload.get("material_id")
    if material_ids is None:
        material_ids = [f"{mp_id_prefix}_{idx:05d}" for idx in range(len(crystal_array_list))]
    else:
        material_ids = [
            str(item) if item not in (None, "") else f"{mp_id_prefix}_{idx:05d}"
            for idx, item in enumerate(material_ids)
        ]

    return crystal_array_list, material_ids


def load_from_cif_dir(path: str):
    cif_paths = sorted(Path(path).glob("*.cif"))
    if not cif_paths:
        raise FileNotFoundError(f"No CIF files found in {path}")

    crystal_array_list = []
    material_ids = []
    for cif_path in cif_paths:
        structure = Structure.from_file(cif_path)
        crystal_array_list.append(
            {
                "frac_coords": np.asarray(structure.frac_coords, dtype=float),
                "atom_types": np.asarray(structure.atomic_numbers, dtype=int),
                "lengths": np.asarray(structure.lattice.abc, dtype=float),
                "angles": np.asarray(structure.lattice.angles, dtype=float),
            }
        )

        stem_parts = cif_path.stem.split("_", 2)
        if len(stem_parts) >= 2 and stem_parts[1]:
            material_ids.append(stem_parts[1])
        else:
            material_ids.append(cif_path.stem)

    return crystal_array_list, material_ids


def main():
    args = parse_args()

    if args.eval_gen:
        crystal_array_list, material_ids = load_from_eval_gen(args.eval_gen, args.mp_id_prefix)
    else:
        crystal_array_list, material_ids = load_from_cif_dir(args.cif_dir)

    lookup = {}
    if args.lookup_pt:
        for path in args.lookup_pt:
            rows = torch.load(path, map_location="cpu", weights_only=False)
            for row in rows:
                lookup[str(row.get("mp_id", ""))] = row

    cached_data = preprocess_tensors(
        crystal_array_list,
        niggli=args.niggli,
        primitive=args.primitive,
        graph_method=args.graph_method,
    )
    add_scaled_lattice_prop(cached_data, args.lattice_scale_method)

    for idx, (row, material_id) in enumerate(zip(cached_data, material_ids, strict=True)):
        source = lookup.get(material_id, {})
        row["mp_id"] = material_id
        if "text2" in source:
            row["text2"] = source["text2"]
        if "formation_energy_per_atom" in source:
            row["formation_energy_per_atom"] = source["formation_energy_per_atom"]
        row["source_index"] = idx

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached_data, output_path)

    print(f"cached_rows={len(cached_data)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()