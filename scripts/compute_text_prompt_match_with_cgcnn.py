"""
Compute correctness of generated materials matching conditions specified by textual prompts.
This script replicates Table 4 from the TGDMat paper.

Uses CGCNN to predict formation energy and band gap for generated structures,
then checks if the predicted sign/type matches the ground truth.
"""

import argparse
import torch
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def extract_formula_from_text(text):
    """Extract chemical formula from text description."""
    patterns = [
        r"chemical formula is ([A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*)",
        r"formula:\s*([A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*)",
        r"formula is ([A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_space_group_from_text(text):
    """Extract space group number from text description."""
    patterns = [
        r"space group number is (\d+)",
        r"space group:\s*(\d+)",
        r"space group number:\s*(\d+)",
        r"belongs to space group (\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_crystal_system_from_text(text):
    """Extract crystal system from text description."""
    systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']

    text_lower = text.lower()
    for system in systems:
        if f"crystal system is {system}" in text_lower or f"crystal system: {system}" in text_lower:
            return system
    return None


def check_formula_match(generated_structure, target_formula):
    """Check if generated structure matches target formula."""
    if target_formula is None:
        return None

    try:
        gen_comp = generated_structure.composition.reduced_composition
        target_comp = Composition(target_formula).reduced_composition
        return gen_comp == target_comp
    except:
        return False


def check_space_group_match(generated_structure, target_sg_number, symprec=0.1):
    """Check if generated structure matches target space group."""
    if target_sg_number is None:
        return None

    try:
        sga = SpacegroupAnalyzer(generated_structure, symprec=symprec)
        gen_sg_number = sga.get_space_group_number()
        return gen_sg_number == target_sg_number
    except:
        return False


def check_crystal_system_match(generated_structure, target_system, symprec=0.1):
    """Check if generated structure matches target crystal system."""
    if target_system is None:
        return None

    try:
        sga = SpacegroupAnalyzer(generated_structure, symprec=symprec)
        gen_system = sga.get_crystal_system()
        return gen_system.lower() == target_system.lower()
    except:
        return False


def check_formation_energy_sign_match(predicted_fe, gt_fe):
    """Check if predicted formation energy sign matches ground truth sign."""
    if predicted_fe is None or gt_fe is None:
        return None

    pred_sign = "positive" if predicted_fe > 0 else "negative"
    gt_sign = "positive" if gt_fe > 0 else "negative"

    return pred_sign == gt_sign


def check_band_gap_type_match(predicted_bg, gt_bg, threshold=0.01):
    """Check if predicted band gap type (zero/nonzero) matches ground truth."""
    if predicted_bg is None or gt_bg is None:
        return None

    pred_type = "zero" if abs(predicted_bg) < threshold else "nonzero"
    gt_type = "zero" if abs(gt_bg) < threshold else "nonzero"

    return pred_type == gt_type


def load_cgcnn_model(fe_model_path, bg_model_path):
    """
    Load trained CGCNN models for property prediction.

    Args:
        fe_model_path: Path to formation energy model
        bg_model_path: Path to band gap model

    Returns:
        Tuple of (fe_model, bg_model)
    """
    import sys
    from pathlib import Path

    # Import CGCNN model from train script
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from train_cgcnn_for_properties import SimpleCGCNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load formation energy model
    fe_checkpoint = torch.load(fe_model_path, map_location=device, weights_only=False)
    fe_args = fe_checkpoint['args']
    fe_model = SimpleCGCNN(
        atom_fea_len=fe_args['atom_fea_len'],
        h_fea_len=fe_args['h_fea_len'],
        n_conv=fe_args['n_conv'],
        n_h=fe_args['n_h']
    ).to(device)
    fe_model.load_state_dict(fe_checkpoint['model_state_dict'])
    fe_model.eval()

    # Load band gap model
    bg_checkpoint = torch.load(bg_model_path, map_location=device, weights_only=False)
    bg_args = bg_checkpoint['args']
    bg_model = SimpleCGCNN(
        atom_fea_len=bg_args['atom_fea_len'],
        h_fea_len=bg_args['h_fea_len'],
        n_conv=bg_args['n_conv'],
        n_h=bg_args['n_h']
    ).to(device)
    bg_model.load_state_dict(bg_checkpoint['model_state_dict'])
    bg_model.eval()

    print(f"Loaded formation energy model (Val MAE: {fe_checkpoint['val_mae']:.4f})")
    print(f"Loaded band gap model (Val MAE: {bg_checkpoint['val_mae']:.4f})")

    return fe_model, bg_model


def structure_to_graph(structure, radius=8.0):
    """
    Convert pymatgen Structure to PyG Data object.

    Args:
        structure: pymatgen Structure
        radius: cutoff radius for neighbors

    Returns:
        PyG Data object
    """
    from torch_geometric.data import Data

    # Get atom features (one-hot encoding of atomic numbers)
    atom_types = [site.specie.Z for site in structure]
    x = torch.zeros(len(atom_types), 92)
    for i, z in enumerate(atom_types):
        x[i, z - 1] = 1.0

    # Get edges (neighbors within radius)
    neighbors = structure.get_all_neighbors(radius, include_index=True)
    edge_index = []
    edge_attr = []

    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            j = neighbor.index
            distance = neighbor.nn_distance
            edge_index.append([i, j])
            edge_attr.append([distance])

    if len(edge_index) == 0:
        # No neighbors found, add self-loops
        edge_index = [[i, i] for i in range(len(atom_types))]
        edge_attr = [[0.0] for _ in range(len(atom_types))]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def predict_properties_with_cgcnn(structures, cgcnn_models, radius=8.0, batch_size=32):
    """
    Predict formation energy and band gap for structures using CGCNN.

    Args:
        structures: List of pymatgen Structure objects
        cgcnn_models: Tuple of (fe_model, bg_model)
        radius: Cutoff radius for neighbors
        batch_size: Batch size for prediction

    Returns:
        formation_energies: List of predicted formation energies
        band_gaps: List of predicted band gaps
    """
    from torch_geometric.data import DataLoader

    fe_model, bg_model = cgcnn_models
    device = next(fe_model.parameters()).device

    # Convert structures to graphs
    data_list = []
    for structure in structures:
        try:
            data = structure_to_graph(structure, radius=radius)
            data_list.append(data)
        except Exception as e:
            print(f"Error converting structure to graph: {e}")
            data_list.append(None)

    # Predict
    formation_energies = []
    band_gaps = []

    loader = DataLoader([d for d in data_list if d is not None], batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        # Predict formation energies
        fe_preds = []
        for batch in loader:
            batch = batch.to(device)
            out = fe_model(batch)
            fe_preds.extend(out.cpu().numpy())

        # Predict band gaps
        loader = DataLoader([d for d in data_list if d is not None], batch_size=batch_size, shuffle=False)
        bg_preds = []
        for batch in loader:
            batch = batch.to(device)
            out = bg_model(batch)
            bg_preds.extend(out.cpu().numpy())

    # Map back to original order
    pred_idx = 0
    for data in data_list:
        if data is not None:
            formation_energies.append(fe_preds[pred_idx])
            band_gaps.append(bg_preds[pred_idx])
            pred_idx += 1
        else:
            formation_energies.append(None)
            band_gaps.append(None)

    return formation_energies, band_gaps


def compute_match_statistics(eval_diff_path, cgcnn_model=None, symprec=0.1):
    """
    Compute matching statistics for generated materials.

    Args:
        eval_diff_path: Path to eval_diff.pt file
        cgcnn_model: Trained CGCNN model for property prediction (optional)
        symprec: Symmetry precision for space group analysis

    Returns:
        Dictionary with match statistics for each global feature
    """
    # Load evaluation results
    data = torch.load(eval_diff_path, map_location='cpu', weights_only=False)

    frac_coords = data['frac_coords']  # [num_evals, num_samples, max_atoms, 3]
    atom_types = data['atom_types']    # [num_evals, num_samples, max_atoms]
    lattices = data['lattices']        # [num_evals, num_samples, 3, 3]
    num_atoms = data['num_atoms']      # [num_evals, num_samples]
    input_data_batch = data['input_data_batch']

    num_evals, num_samples = frac_coords.shape[:2]

    # Extract ground truth properties from input data
    text_descriptions = []
    gt_formulas = []
    gt_space_groups = []
    gt_formation_energies = []
    gt_band_gaps = []

    for i in range(num_samples):
        # Text description
        if hasattr(input_data_batch, 'text'):
            text_descriptions.append(input_data_batch.text[i] if isinstance(input_data_batch.text, list) else "")
        else:
            text_descriptions.append("")

        # Ground truth formula
        if hasattr(input_data_batch, 'pretty_formula'):
            gt_formulas.append(input_data_batch.pretty_formula[i] if isinstance(input_data_batch.pretty_formula, list) else None)
        else:
            gt_formulas.append(None)

        # Ground truth space group
        if hasattr(input_data_batch, 'spacegroup'):
            sg = input_data_batch.spacegroup[i]
            gt_space_groups.append(int(sg.item()) if torch.is_tensor(sg) else int(sg))
        else:
            gt_space_groups.append(None)

        # Ground truth formation energy
        if hasattr(input_data_batch, 'y') and input_data_batch.y is not None:
            gt_formation_energies.append(input_data_batch.y[i].item() if len(input_data_batch.y) > i else None)
        else:
            gt_formation_energies.append(None)

        # Ground truth band gap
        if hasattr(input_data_batch, 'band_gap') and input_data_batch.band_gap is not None:
            gt_band_gaps.append(input_data_batch.band_gap[i].item() if len(input_data_batch.band_gap) > i else None)
        else:
            gt_band_gaps.append(None)

    # Statistics counters
    stats = {
        'formula': {'total': 0, 'matched': 0},
        'space_group': {'total': 0, 'matched': 0},
        'crystal_system': {'total': 0, 'matched': 0},
        'formation_energy': {'total': 0, 'matched': 0},
        'band_gap': {'total': 0, 'matched': 0},
    }

    # Build structures and collect for CGCNN prediction
    structures = []
    valid_indices = []

    for sample_idx in range(num_samples):
        eval_idx = 0  # Use first evaluation

        coords = frac_coords[eval_idx, sample_idx]
        types = atom_types[eval_idx, sample_idx]
        lattice = lattices[eval_idx, sample_idx]
        n_atoms = num_atoms[eval_idx, sample_idx].item()

        try:
            coords = coords[:n_atoms].numpy()
            types = types[:n_atoms].numpy()
            lattice_matrix = lattice.numpy()

            structure = Structure(
                lattice=Lattice(lattice_matrix),
                species=types,
                coords=coords,
                coords_are_cartesian=False
            )

            structures.append(structure)
            valid_indices.append(sample_idx)

        except Exception as e:
            print(f"Error building structure for sample {sample_idx}: {e}")
            structures.append(None)
            valid_indices.append(None)

    # Predict properties using CGCNN if model is provided
    predicted_fes = [None] * num_samples
    predicted_bgs = [None] * num_samples

    if cgcnn_model is not None:
        print("Predicting properties with CGCNN...")
        valid_structures = [s for s in structures if s is not None]
        valid_idx_list = [i for i in valid_indices if i is not None]

        if len(valid_structures) > 0:
            pred_fes, pred_bgs = predict_properties_with_cgcnn(valid_structures, cgcnn_model)

            for i, sample_idx in enumerate(valid_idx_list):
                predicted_fes[sample_idx] = pred_fes[i]
                predicted_bgs[sample_idx] = pred_bgs[i]

    # Process each sample
    for sample_idx in range(num_samples):
        if structures[sample_idx] is None:
            continue

        structure = structures[sample_idx]
        text = text_descriptions[sample_idx]

        # Extract target properties from text (if available)
        target_formula_text = extract_formula_from_text(text)
        target_sg_text = extract_space_group_from_text(text)
        target_system_text = extract_crystal_system_from_text(text)

        # Use ground truth as target if text doesn't specify
        target_formula = target_formula_text if target_formula_text else gt_formulas[sample_idx]
        target_sg = target_sg_text if target_sg_text else gt_space_groups[sample_idx]
        target_system = target_system_text

        try:
            # Check structural properties
            formula_match = check_formula_match(structure, target_formula)
            if formula_match is not None:
                stats['formula']['total'] += 1
                if formula_match:
                    stats['formula']['matched'] += 1

            sg_match = check_space_group_match(structure, target_sg, symprec)
            if sg_match is not None:
                stats['space_group']['total'] += 1
                if sg_match:
                    stats['space_group']['matched'] += 1

            system_match = check_crystal_system_match(structure, target_system, symprec)
            if system_match is not None:
                stats['crystal_system']['total'] += 1
                if system_match:
                    stats['crystal_system']['matched'] += 1

            # Check predicted properties against ground truth
            if cgcnn_model is not None:
                # Formation energy sign match
                pred_fe = predicted_fes[sample_idx]
                gt_fe = gt_formation_energies[sample_idx]
                fe_match = check_formation_energy_sign_match(pred_fe, gt_fe)
                if fe_match is not None:
                    stats['formation_energy']['total'] += 1
                    if fe_match:
                        stats['formation_energy']['matched'] += 1

                # Band gap type match
                pred_bg = predicted_bgs[sample_idx]
                gt_bg = gt_band_gaps[sample_idx]
                bg_match = check_band_gap_type_match(pred_bg, gt_bg)
                if bg_match is not None:
                    stats['band_gap']['total'] += 1
                    if bg_match:
                        stats['band_gap']['matched'] += 1

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue

    # Compute percentages
    results = {}
    for feature, counts in stats.items():
        if counts['total'] > 0:
            percentage = 100.0 * counts['matched'] / counts['total']
            results[feature] = {
                'matched': counts['matched'],
                'total': counts['total'],
                'percentage': percentage
            }
        else:
            results[feature] = {
                'matched': 0,
                'total': 0,
                'percentage': None
            }

    return results


def main(args):
    eval_diff_path = Path(args.eval_diff_path)

    if not eval_diff_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_diff_path}")

    # Load CGCNN models if provided
    cgcnn_models = None
    if args.fe_model_path and args.bg_model_path:
        print(f"Loading CGCNN models...")
        print(f"  Formation energy model: {args.fe_model_path}")
        print(f"  Band gap model: {args.bg_model_path}")
        cgcnn_models = load_cgcnn_model(args.fe_model_path, args.bg_model_path)
        print("CGCNN models loaded successfully\n")

    print(f"Computing text prompt matching statistics for: {eval_diff_path}")
    print(f"Symmetry precision: {args.symprec}")
    print(f"Using CGCNN for property prediction: {cgcnn_models is not None}")
    print()

    results = compute_match_statistics(eval_diff_path, cgcnn_model=cgcnn_models, symprec=args.symprec)

    # Print results in table format
    print("=" * 70)
    print("Correctness of Generated Materials Matching Conditions")
    print("=" * 70)
    print(f"{'Global Feature':<25} {'Matched':<10} {'Total':<10} {'Percentage':<10}")
    print("-" * 70)

    for feature in ['formula', 'space_group', 'crystal_system', 'formation_energy', 'band_gap']:
        res = results[feature]
        if res['percentage'] is not None:
            print(f"{feature.replace('_', ' ').title():<25} {res['matched']:<10} {res['total']:<10} {res['percentage']:.2f}%")
        else:
            print(f"{feature.replace('_', ' ').title():<25} {res['matched']:<10} {res['total']:<10} {'N/A':<10}")

    print("=" * 70)

    # Save results
    if args.output:
        output_path = Path(args.output)
        torch.save(results, output_path)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute text prompt matching statistics (Table 4) with CGCNN property prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('eval_diff_path', type=str, help='Path to eval_diff.pt file')
    parser.add_argument('--fe-model-path', type=str, help='Path to trained CGCNN model for formation energy prediction')
    parser.add_argument('--bg-model-path', type=str, help='Path to trained CGCNN model for band gap prediction')
    parser.add_argument('--symprec', type=float, default=0.1, help='Symmetry precision for space group analysis')
    parser.add_argument('--output', type=str, help='Output path to save results')

    args = parser.parse_args()
    main(args)
