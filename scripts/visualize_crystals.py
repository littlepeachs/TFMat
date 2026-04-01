"""
Visualize generated crystals from eval_diff_*.pt files.
- Convert to CIF files
- Render structure images using matplotlib + pymatgen
"""

import sys
sys.path.append('.')

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

# Chemical symbols lookup (index -> element symbol)
chemical_symbols = [
    'X',
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og',
]

# Element colors for visualization (CPK-like)
ELEMENT_COLORS = {
    'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00',
    'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
    'F': '#90E050', 'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
    'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30',
    'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
    'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7',
    'Mn': '#9C7AC7', 'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050',
    'Cu': '#C88033', 'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F',
    'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929', 'Kr': '#5CB8D1',
    'default': '#FF1493',
}

ELEMENT_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
    'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Ti': 1.60, 'Fe': 1.52, 'Co': 1.50, 'Ni': 1.24,
    'Cu': 1.32, 'Zn': 1.22, 'default': 1.20,
}


def extract_crystals(data, num_crystals=20, eval_idx=0):
    """Extract individual crystal structures from a batch .pt file."""
    frac_coords = data['frac_coords']
    atom_types = data['atom_types']
    lengths = data['lengths']
    angles = data['angles']
    num_atoms = data['num_atoms']

    # Handle multi-eval dimension: shape could be (num_evals, total_atoms, 3) or (total_atoms, 3)
    if frac_coords.dim() == 3:
        frac_coords = frac_coords[eval_idx]
        atom_types = atom_types[eval_idx]
        lengths = lengths[eval_idx]
        angles = angles[eval_idx]
        num_atoms = num_atoms[eval_idx]

    crystals = []
    start_idx = 0
    total = min(num_crystals, len(num_atoms))

    for i in range(total):
        n = int(num_atoms[i].item())
        fc = frac_coords[start_idx:start_idx + n].numpy()
        at = atom_types[start_idx:start_idx + n].numpy().astype(int)
        l = lengths[i].numpy()
        a = angles[i].numpy()

        species = [chemical_symbols[t] for t in at]

        try:
            lattice = Lattice.from_parameters(
                a=float(l[0]), b=float(l[1]), c=float(l[2]),
                alpha=float(a[0]), beta=float(a[1]), gamma=float(a[2]),
            )
            structure = Structure(
                lattice=lattice,
                species=species,
                coords=fc,
                coords_are_cartesian=False,
            )
            crystals.append({
                'structure': structure,
                'species': species,
                'frac_coords': fc,
                'lengths': l,
                'angles': a,
                'index': i,
            })
        except Exception as e:
            print(f"  Warning: Crystal {i} failed to build: {e}")

        start_idx += n

    return crystals


def save_cif(crystal, filepath):
    """Save a crystal structure as CIF file."""
    crystal['structure'].to(filename=str(filepath))


def plot_crystal_3d(crystal, filepath, title=None):
    """Render a crystal structure as a 3D plot and save as image."""
    structure = crystal['structure']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get cartesian coordinates
    cart_coords = structure.cart_coords
    lattice = structure.lattice

    # Draw atoms
    for i, site in enumerate(structure):
        elem = site.specie.symbol
        color = ELEMENT_COLORS.get(elem, ELEMENT_COLORS['default'])
        radius = ELEMENT_RADII.get(elem, ELEMENT_RADII['default'])
        ax.scatter(
            cart_coords[i, 0], cart_coords[i, 1], cart_coords[i, 2],
            c=color, s=radius * 200, edgecolors='black', linewidth=0.5,
            alpha=0.9, depthshade=True,
        )
        ax.text(
            cart_coords[i, 0], cart_coords[i, 1], cart_coords[i, 2] + 0.3,
            elem, fontsize=7, ha='center', va='bottom', color='black',
        )

    # Draw unit cell edges
    origin = np.array([0, 0, 0])
    a_vec = lattice.matrix[0]
    b_vec = lattice.matrix[1]
    c_vec = lattice.matrix[2]

    vertices = [
        origin, a_vec, b_vec, c_vec,
        a_vec + b_vec, a_vec + c_vec, b_vec + c_vec,
        a_vec + b_vec + c_vec,
    ]

    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (2, 4), (2, 6),
        (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
    ]

    for e0, e1 in edges:
        v0, v1 = vertices[e0], vertices[e1]
        ax.plot3D(
            [v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]],
            'b-', alpha=0.3, linewidth=1,
        )

    # Formatting
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')

    if title:
        ax.set_title(title, fontsize=10, pad=10)

    # Equal aspect ratio
    all_pts = np.array(vertices + [cart_coords[i] for i in range(len(cart_coords))])
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.savefig(str(filepath), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize generated crystals from eval_diff_*.pt files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('pt_files', nargs='+', help='Path(s) to eval_diff_*.pt file(s)')
    parser.add_argument('-n', '--num_crystals', type=int, default=20, help='Number of crystals to extract')
    parser.add_argument('-o', '--output_dir', type=str, default='crystal_vis', help='Output directory')
    parser.add_argument('--eval_idx', type=int, default=0, help='Which eval sample to use (for multi-eval)')
    args = parser.parse_args()

    for pt_file in args.pt_files:
        pt_path = Path(pt_file)
        if not pt_path.exists():
            print(f"File not found: {pt_path}")
            continue

        label = pt_path.stem  # e.g. eval_diff_csp1
        out_dir = Path(args.output_dir) / label
        cif_dir = out_dir / 'cif'
        img_dir = out_dir / 'images'
        cif_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {pt_path}")
        print(f"Output dir: {out_dir}")
        print(f"{'='*60}")

        data = torch.load(pt_path, weights_only=False, map_location='cpu')

        # Print summary
        print(f"  Keys: {list(data.keys())}")
        if 'num_atoms' in data:
            na = data['num_atoms']
            if na.dim() >= 2:
                print(f"  num_evals={na.shape[0]}, num_crystals={na.shape[1]}")
            else:
                print(f"  num_crystals={na.shape[0]}")
        if 'time' in data:
            print(f"  Generation time: {data['time']:.1f}s")

        crystals = extract_crystals(data, num_crystals=args.num_crystals, eval_idx=args.eval_idx)
        print(f"  Extracted {len(crystals)} crystals")

        for cryst in crystals:
            idx = cryst['index']
            struct = cryst['structure']
            formula = struct.composition.reduced_formula
            sg = "?"
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sg = SpacegroupAnalyzer(struct, symprec=0.1).get_space_group_symbol()
            except Exception:
                pass

            # Save CIF
            cif_path = cif_dir / f'{idx:03d}_{formula}.cif'
            save_cif(cryst, cif_path)

            # Save image
            img_path = img_dir / f'{idx:03d}_{formula}.png'
            title = f'#{idx} {formula} (SG: {sg})\n' \
                    f'a={cryst["lengths"][0]:.2f} b={cryst["lengths"][1]:.2f} c={cryst["lengths"][2]:.2f}\n' \
                    f'α={cryst["angles"][0]:.1f} β={cryst["angles"][1]:.1f} γ={cryst["angles"][2]:.1f}'
            plot_crystal_3d(cryst, img_path, title=title)

            print(f"  [{idx:03d}] {formula:12s}  SG={sg:10s}  "
                  f"a={cryst['lengths'][0]:6.2f} b={cryst['lengths'][1]:6.2f} c={cryst['lengths'][2]:6.2f}  "
                  f"α={cryst['angles'][0]:6.1f} β={cryst['angles'][1]:6.1f} γ={cryst['angles'][2]:6.1f}")

        print(f"\n  CIF files: {cif_dir}")
        print(f"  Images:    {img_dir}")
        print(f"  Done! ({len(crystals)} crystals)")


if __name__ == '__main__':
    main()
