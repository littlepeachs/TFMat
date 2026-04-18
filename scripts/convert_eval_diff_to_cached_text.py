import argparse
from pathlib import Path

import torch

from diffcsp.common.data_utils import add_scaled_lattice_prop, preprocess_tensors
from eval_utils import get_crystals_list


def parse_args():
    parser = argparse.ArgumentParser(description="Convert eval_diff*.pt to cached list-format predicted crystals.")
    parser.add_argument("--eval-diff", required=True, help="Path to eval_diff*.pt")
    parser.add_argument("--source-cache", required=True, help="Cached pt used as the source evaluation set")
    parser.add_argument("--output", required=True, help="Output cached list-format pt")
    parser.add_argument("--graph-method", default="crystalnn")
    parser.add_argument("--lattice-scale-method", default="scale_length")
    parser.add_argument("--niggli", action="store_true", default=True)
    parser.add_argument("--primitive", action="store_true", default=False)
    return parser.parse_args()


def select_first_eval(tensor_or_obj):
    if isinstance(tensor_or_obj, torch.Tensor) and tensor_or_obj.dim() >= 1:
        return tensor_or_obj[0]
    return tensor_or_obj


def main():
    args = parse_args()

    payload = torch.load(args.eval_diff, map_location="cpu", weights_only=False)
    source_cache = torch.load(args.source_cache, map_location="cpu", weights_only=False)

    crystal_array_list = get_crystals_list(
        select_first_eval(payload["frac_coords"]),
        select_first_eval(payload["atom_types"]),
        select_first_eval(payload["lengths"]),
        select_first_eval(payload["angles"]),
        select_first_eval(payload["num_atoms"]),
    )

    if len(crystal_array_list) != len(source_cache):
        raise ValueError(f"Length mismatch: predicted={len(crystal_array_list)} source={len(source_cache)}")

    cached_data = preprocess_tensors(
        crystal_array_list,
        niggli=args.niggli,
        primitive=args.primitive,
        graph_method=args.graph_method,
    )
    add_scaled_lattice_prop(cached_data, args.lattice_scale_method)

    for row, source in zip(cached_data, source_cache, strict=True):
        row["mp_id"] = str(source.get("mp_id", ""))
        if "text2" in source:
            row["text2"] = source["text2"]
        if "formation_energy_per_atom" in source:
            row["formation_energy_per_atom"] = source["formation_energy_per_atom"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached_data, output_path)
    print(f"cached_rows={len(cached_data)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()