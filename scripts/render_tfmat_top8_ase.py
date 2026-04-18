import argparse
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ase.build.tools import niggli_reduce
from ase.io import read
from ase.data.colors import jmol_colors
from ase.data import atomic_numbers
from ase.visualize.plot import plot_atoms
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Arial"

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:
    StructureMatcher = None
    AseAtomsAdaptor = None


ROTATION = "14x,0y,0z"
DISPLAY_CORNER = (0.08, 0.08, 0.08)
ORIENTATION_MODE = "normal"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return None
    return vector / norm


def compute_display_shift(atoms, anchor=DISPLAY_CORNER):
    scaled = atoms.get_scaled_positions(wrap=True)
    shift = np.zeros(3, dtype=float)

    for axis in range(3):
        coords = np.mod(scaled[:, axis], 1.0)
        if len(coords) == 0:
            continue

        if len(coords) == 1:
            start = float(coords[0])
            span = 0.0
        else:
            sorted_coords = np.sort(coords)
            gaps = np.empty_like(sorted_coords)
            gaps[:-1] = sorted_coords[1:] - sorted_coords[:-1]
            gaps[-1] = sorted_coords[0] + 1.0 - sorted_coords[-1]
            split_idx = int(np.argmax(gaps))
            start = float(sorted_coords[(split_idx + 1) % len(sorted_coords)])
            shifted = np.mod(coords - start, 1.0)
            span = float(shifted.max() - shifted.min())

        offset = float(anchor[axis])
        if span + offset > 0.98:
            offset = max(0.0, 0.98 - span)
        shift[axis] = -start + offset

    return shift


def apply_display_shift(atoms, shift):
    atoms = atoms.copy()
    scaled = atoms.get_scaled_positions(wrap=True)
    scaled = np.mod(scaled + shift, 1.0)
    atoms.set_scaled_positions(scaled)
    atoms.wrap()
    return atoms


def canonicalize_atoms(atoms, shift=None):
    atoms = atoms.copy()
    try:
        niggli_reduce(atoms)
    except Exception:
        pass
    atoms.wrap()
    if shift is None:
        shift = compute_display_shift(atoms)
    return apply_display_shift(atoms, shift)


def compute_orientation_basis(atoms):
    cell = np.asarray(atoms.cell.array, dtype=float)
    normalized_vectors = [normalize_vector(vector) for vector in cell]
    normalized_vectors = [vector for vector in normalized_vectors if vector is not None]

    target_x = None
    if normalized_vectors:
        target_x = normalize_vector(np.sum(normalized_vectors, axis=0))
    if target_x is None:
        target_x = normalize_vector(cell[0])
    if target_x is None:
        target_x = np.array([1.0, 0.0, 0.0])

    helper = np.cross(cell[0], cell[1]) + np.cross(cell[1], cell[2]) + np.cross(cell[2], cell[0])
    helper = helper - target_x * np.dot(helper, target_x)
    target_z = normalize_vector(helper)
    if target_z is None:
        for candidate in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
            candidate = candidate - target_x * np.dot(candidate, target_x)
            target_z = normalize_vector(candidate)
            if target_z is not None:
                break

    target_y = normalize_vector(np.cross(target_z, target_x))
    target_z = normalize_vector(np.cross(target_x, target_y))
    return np.vstack([target_x, target_y, target_z])


def orient_atoms_for_display(atoms, basis):
    atoms = atoms.copy()
    rotated_positions = atoms.positions @ basis.T
    rotated_cell = atoms.cell.array @ basis.T
    atoms.set_positions(rotated_positions)
    atoms.set_cell(rotated_cell, scale_atoms=False)
    atoms.wrap()
    return atoms


def align_generated_to_gt(gt_atoms, gen_atoms, matcher):
    if matcher is None or AseAtomsAdaptor is None:
        return gt_atoms, gen_atoms

    try:
        gt_structure = AseAtomsAdaptor.get_structure(gt_atoms)
        gen_structure = AseAtomsAdaptor.get_structure(gen_atoms)
        gen_like_gt = matcher.get_s2_like_s1(gt_structure, gen_structure)
        if gen_like_gt is None:
            return gt_atoms, gen_atoms
        return gt_atoms, AseAtomsAdaptor.get_atoms(gen_like_gt)
    except Exception:
        return gt_atoms, gen_atoms


def make_legend_handles(atoms):
    symbols = sorted(set(atoms.get_chemical_symbols()), key=lambda s: atomic_numbers[s])
    handles = []
    for sym in symbols:
        color = jmol_colors[atomic_numbers[sym]]
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=20, markeredgecolor="black", markeredgewidth=1.8, label=sym)
        )
    return handles


def style_axis(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_facecolor("white")


def render_single(atoms, out_path: Path, rotation: str, scale: float = 0.95):
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    plot_atoms(
        atoms,
        ax=ax,
        rotation=rotation,
        show_unit_cell=2,
        radii=0.38,
        scale=scale,
    )
    style_axis(ax)
    handles = make_legend_handles(atoms)
    ax.legend(handles=handles, loc="lower center", frameon=False,
              fontsize=20, handletextpad=0.3, borderpad=0.3, ncol=len(set(atoms.get_chemical_symbols())),
              bbox_to_anchor=(0.5, -0.25), columnspacing=1.0)
    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def render_pair(gt_atoms, gen_atoms, out_path: Path, rotation: str, scale: float = 0.95):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
    plot_atoms(
        gt_atoms,
        ax=axes[0],
        rotation=rotation,
        show_unit_cell=2,
        radii=0.38,
        scale=scale,
    )
    plot_atoms(
        gen_atoms,
        ax=axes[1],
        rotation=rotation,
        show_unit_cell=2,
        radii=0.38,
        scale=scale,
    )

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xmins.append(min(x0, x1))
        xmaxs.append(max(x0, x1))
        ymins.append(min(y0, y1))
        ymaxs.append(max(y0, y1))

    xpad = 0.04 * (max(xmaxs) - min(xmins))
    ypad = 0.04 * (max(ymaxs) - min(ymins))
    xlim = (min(xmins) - xpad, max(xmaxs) + xpad)
    ylim = (min(ymins) - ypad, max(ymaxs) + ypad)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        style_axis(ax)

    # Add legend centered below both subplots
    all_symbols = set(gt_atoms.get_chemical_symbols()) | set(gen_atoms.get_chemical_symbols())
    from ase.atoms import Atoms
    combined = Atoms(symbols=sorted(all_symbols, key=lambda s: atomic_numbers[s]))
    handles = make_legend_handles(combined)
    fig.legend(handles=handles, loc="lower center", frameon=False,
               fontsize=20, handletextpad=0.3, borderpad=0.3, ncol=len(all_symbols),
               bbox_to_anchor=(0.5, -0.08), columnspacing=1.0)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def render_grid(pair_paths, out_path: Path, ncols: int = 2):
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in pair_paths]
    if not images:
        return

    widths = [im.width for im in images]
    heights = [im.height for im in images]
    tile_w = max(widths)
    tile_h = max(heights)
    nrows = (len(images) + ncols - 1) // ncols

    canvas = Image.new("RGB", (tile_w * ncols, tile_h * nrows), color="white")
    for idx, im in enumerate(images):
        row = idx // ncols
        col = idx % ncols
        x = col * tile_w + (tile_w - im.width) // 2
        y = row * tile_h + (tile_h - im.height) // 2
        canvas.paste(im, (x, y))
    canvas.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--gen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rotation", default=ROTATION)
    parser.add_argument(
        "--orientation",
        choices=("normal", "rightward"),
        default=ORIENTATION_MODE,
        help="Whether to preserve the original lattice orientation or rotate it toward the right for display.",
    )
    parser.add_argument("--grid-name", default="top8_pairs_grid.png")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--pairs-only", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    gt_dir = Path(args.gt_dir)
    gen_dir = Path(args.gen_dir)
    out_dir = Path(args.output_dir)

    gt_img_dir = out_dir / "gt"
    gen_img_dir = out_dir / "generated"
    pair_img_dir = out_dir / "pairs"
    ensure_dir(gt_img_dir)
    ensure_dir(gen_img_dir)
    ensure_dir(pair_img_dir)

    matcher = None
    if StructureMatcher is not None:
        matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    pair_paths = []
    for rank, row in enumerate(manifest.itertuples(index=False), start=1):
        pattern = f"{rank:02d}_*.cif"
        gt_matches = sorted(gt_dir.glob(pattern))
        gen_matches = sorted(gen_dir.glob(pattern))
        if len(gt_matches) != 1 or len(gen_matches) != 1:
            raise FileNotFoundError(f"Expected exactly one GT and one generated CIF for rank {rank:02d}")

        gt_path = gt_matches[0]
        gen_path = gen_matches[0]
        gt_atoms_raw = read(gt_path)
        gen_atoms_raw = read(gen_path)
        gt_atoms_raw, gen_atoms_raw = align_generated_to_gt(gt_atoms_raw, gen_atoms_raw, matcher)

        display_shift = compute_display_shift(gt_atoms_raw)
        if matcher is None:
            gt_atoms = canonicalize_atoms(gt_atoms_raw, shift=display_shift)
            gen_atoms = canonicalize_atoms(gen_atoms_raw, shift=display_shift)
        else:
            gt_atoms = apply_display_shift(gt_atoms_raw, display_shift)
            gen_atoms = apply_display_shift(gen_atoms_raw, display_shift)

        if args.orientation == "rightward":
            orientation_basis = compute_orientation_basis(gt_atoms)
            gt_atoms = orient_atoms_for_display(gt_atoms, orientation_basis)
            gen_atoms = orient_atoms_for_display(gen_atoms, orientation_basis)

        stem = gt_path.stem
        gt_png = gt_img_dir / f"{stem}.png"
        gen_png = gen_img_dir / f"{stem}.png"
        pair_png = pair_img_dir / f"{stem}.png"

        if not args.pairs_only:
            render_single(gt_atoms, gt_png, rotation=args.rotation)
            render_single(gen_atoms, gen_png, rotation=args.rotation)
        render_pair(gt_atoms, gen_atoms, pair_png, rotation=args.rotation)
        pair_paths.append(pair_png)

    if not args.skip_grid:
        render_grid(pair_paths, out_dir / args.grid_name, ncols=2)
    print(f"rendered_pairs={len(pair_paths)}")
    print(f"{out_dir}")


if __name__ == "__main__":
    main()
