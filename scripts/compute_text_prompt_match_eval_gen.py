"""
Compute text prompt matching statistics for eval_gen format (DNG generation).
This script evaluates how well generated materials match the text prompts.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd
from tqdm import tqdm
import sys

# Import CGCNN model
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from train_cgcnn_for_properties import SimpleCGCNN


def load_cgcnn_models(fe_model_path, bg_model_path):
    """Load trained CGCNN models for formation energy and band gap prediction."""
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

    return {
        'fe_model': fe_model,
        'bg_model': bg_model,
        'device': device,
        'fe_val_mae': fe_checkpoint.get('best_val_mae', None),
        'bg_val_mae': bg_checkpoint.get('best_val_mae', None)
    }


def reconstruct_structures_from_eval_gen(data):
    """
    Reconstruct pymatgen structures from eval_gen format.

    Args:
        data: Dictionary with keys 'frac_coords', 'atom_types', 'lengths', 'angles', 'num_atoms'

    Returns:
        List of pymatgen Structure objects
    """
    frac_coords = data['frac_coords']  # [total_atoms, 3]
    atom_types = data['atom_types']    # [total_atoms]
    lengths = data['lengths']          # [num_samples, 3]
    angles = data['angles']            # [num_samples, 3]
    num_atoms = data['num_atoms']      # [num_samples]

    structures = []
    atom_idx = 0

    for i in range(len(num_atoms)):
        n_atoms = int(num_atoms[i].item())

        # Get atoms for this structure
        coords = frac_coords[atom_idx:atom_idx + n_atoms].numpy()
        types = atom_types[atom_idx:atom_idx + n_atoms].numpy()
        atom_idx += n_atoms

        # Create lattice
        a, b, c = lengths[i].numpy()
        alpha, beta, gamma = angles[i].numpy()
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        # Create structure
        structure = Structure(
            lattice,
            types,
            coords,
            coords_are_cartesian=False
        )
        structures.append(structure)

    return structures


def load_test_data(test_csv_path):
    """Load test dataset with ground truth labels."""
    df = pd.read_csv(test_csv_path)
    return df


def extract_properties_from_text(text):
    """Extract expected properties from text description."""
    properties = {}

    # This is a placeholder - actual implementation depends on text format
    # For now, return None to indicate we need ground truth from test set
    return None


def predict_properties_with_cgcnn(structures, cgcnn_models):
    """Predict formation energy and band gap using CGCNN."""
    # CGCNN prediction not implemented yet
    # Would need proper graph construction from structures
    fe_predictions = [None] * len(structures)
    bg_predictions = [None] * len(structures)
    return fe_predictions, bg_predictions


def compute_match_statistics(eval_gen_path, test_csv_path, cgcnn_models=None, symprec=0.1):
    """
    Compute matching statistics between generated structures and text prompts.

    Args:
        eval_gen_path: Path to eval_gen_{label}.pt file
        test_csv_path: Path to test.csv with ground truth
        cgcnn_models: Dictionary with CGCNN models (optional)
        symprec: Symmetry precision for spacegroup analysis

    Returns:
        Dictionary with match statistics
    """
    # Load generated structures
    print(f"Loading generated structures from: {eval_gen_path}")
    data = torch.load(eval_gen_path, map_location='cpu', weights_only=False)
    structures = reconstruct_structures_from_eval_gen(data)
    material_ids = data.get('material_id', None)

    print(f"Loaded {len(structures)} generated structures")

    # Load test data
    print(f"Loading test data from: {test_csv_path}")
    test_df = load_test_data(test_csv_path)

    # Match material_ids to test data
    if material_ids is None:
        print("Warning: No material_ids in eval_gen file, using first N samples from test set")
        test_subset = test_df.head(len(structures))
    else:
        test_subset = test_df[test_df['material_id'].isin(material_ids)]
        # Reorder to match material_ids order
        test_subset = test_subset.set_index('material_id').loc[material_ids].reset_index()

    print(f"Matched {len(test_subset)} test samples")

    # Initialize counters
    results = {
        'total': len(structures),
        'formula_match': 0,
        'spacegroup_match': 0,
        'crystal_system_match': 0,
        'formation_energy_match': 0,
        'band_gap_match': 0,
        'valid_structures': 0,
        'details': []
    }

    # Predict properties if CGCNN models provided
    if cgcnn_models is not None:
        print("Predicting properties with CGCNN...")
        print("Warning: CGCNN prediction not fully implemented yet, using ground truth")
        fe_predictions = None
        bg_predictions = None
    else:
        fe_predictions = None
        bg_predictions = None

    # Compute matches
    print("Computing matches...")
    for i, (structure, row) in enumerate(tqdm(zip(structures, test_subset.iterrows()), total=len(structures))):
        _, row = row
        detail = {
            'index': i,
            'material_id': row.get('material_id', f'sample_{i}')
        }

        try:
            # Formula match
            gen_formula = structure.composition.reduced_formula
            gt_formula = row.get('pretty_formula', row.get('formula', ''))
            formula_match = (gen_formula == gt_formula)
            detail['formula_match'] = formula_match
            detail['gen_formula'] = gen_formula
            detail['gt_formula'] = gt_formula
            if formula_match:
                results['formula_match'] += 1

            # Spacegroup match
            try:
                analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
                gen_spacegroup = analyzer.get_space_group_number()
                gt_spacegroup = int(row.get('spacegroup.number', row.get('spacegroup_number', 0)))
                spacegroup_match = (gen_spacegroup == gt_spacegroup)
                detail['spacegroup_match'] = spacegroup_match
                detail['gen_spacegroup'] = gen_spacegroup
                detail['gt_spacegroup'] = gt_spacegroup
                if spacegroup_match:
                    results['spacegroup_match'] += 1

                # Crystal system match
                gen_crystal_system = analyzer.get_crystal_system()
                gt_crystal_system = row.get('spacegroup.crystal_system', row.get('crystal_system', ''))
                crystal_system_match = (gen_crystal_system == gt_crystal_system)
                detail['crystal_system_match'] = crystal_system_match
                detail['gen_crystal_system'] = gen_crystal_system
                detail['gt_crystal_system'] = gt_crystal_system
                if crystal_system_match:
                    results['crystal_system_match'] += 1
            except Exception as e:
                detail['symmetry_error'] = str(e)

            # Formation energy match (sign match)
            gt_fe = row.get('formation_energy_per_atom', None)
            if gt_fe is not None and fe_predictions is not None and fe_predictions[i] is not None:
                pred_fe = fe_predictions[i]
                fe_match = ((pred_fe >= 0) == (gt_fe >= 0))
                detail['formation_energy_match'] = fe_match
                detail['pred_formation_energy'] = pred_fe
                detail['gt_formation_energy'] = gt_fe
                if fe_match:
                    results['formation_energy_match'] += 1
            else:
                # Use ground truth for now
                detail['formation_energy_match'] = None
                detail['gt_formation_energy'] = gt_fe

            # Band gap match (zero/nonzero match)
            gt_bg = row.get('band_gap', None)
            if gt_bg is not None and bg_predictions is not None and bg_predictions[i] is not None:
                pred_bg = bg_predictions[i]
                bg_match = ((pred_bg == 0) == (gt_bg == 0))
                detail['band_gap_match'] = bg_match
                detail['pred_band_gap'] = pred_bg
                detail['gt_band_gap'] = gt_bg
                if bg_match:
                    results['band_gap_match'] += 1
            else:
                # Use ground truth for now
                detail['band_gap_match'] = None
                detail['gt_band_gap'] = gt_bg

            results['valid_structures'] += 1

        except Exception as e:
            detail['error'] = str(e)

        results['details'].append(detail)

    # Compute percentages
    total = results['total']
    results['formula_match_pct'] = 100.0 * results['formula_match'] / total if total > 0 else 0
    results['spacegroup_match_pct'] = 100.0 * results['spacegroup_match'] / total if total > 0 else 0
    results['crystal_system_match_pct'] = 100.0 * results['crystal_system_match'] / total if total > 0 else 0

    if fe_predictions is not None:
        results['formation_energy_match_pct'] = 100.0 * results['formation_energy_match'] / total if total > 0 else 0
    if bg_predictions is not None:
        results['band_gap_match_pct'] = 100.0 * results['band_gap_match'] / total if total > 0 else 0

    return results


def print_results(results):
    """Print match statistics in a readable format."""
    print("\n" + "="*60)
    print("TEXT PROMPT MATCHING RESULTS (Table 4 Format)")
    print("="*60)
    print(f"Total samples: {results['total']}")
    print(f"Valid structures: {results['valid_structures']}")
    print()
    print(f"Formula match:         {results['formula_match']:4d} / {results['total']:4d} ({results['formula_match_pct']:5.2f}%)")
    print(f"Space group match:     {results['spacegroup_match']:4d} / {results['total']:4d} ({results['spacegroup_match_pct']:5.2f}%)")
    print(f"Crystal system match:  {results['crystal_system_match']:4d} / {results['total']:4d} ({results['crystal_system_match_pct']:5.2f}%)")

    if 'formation_energy_match_pct' in results:
        print(f"Formation energy match: {results['formation_energy_match']:4d} / {results['total']:4d} ({results['formation_energy_match_pct']:5.2f}%)")
    else:
        print(f"Formation energy match: Not computed (CGCNN prediction needed)")

    if 'band_gap_match_pct' in results:
        print(f"Band gap match:        {results['band_gap_match']:4d} / {results['total']:4d} ({results['band_gap_match_pct']:5.2f}%)")
    else:
        print(f"Band gap match:        Not computed (CGCNN prediction needed)")

    print("="*60)


def main(args):
    # Load CGCNN models if provided
    cgcnn_models = None
    if args.fe_model_path and args.bg_model_path:
        print("Loading CGCNN models...")
        print(f"  Formation energy model: {args.fe_model_path}")
        print(f"  Band gap model: {args.bg_model_path}")
        cgcnn_models = load_cgcnn_models(args.fe_model_path, args.bg_model_path)
        if cgcnn_models['fe_val_mae'] is not None:
            print(f"Loaded formation energy model (Val MAE: {cgcnn_models['fe_val_mae']:.4f})")
        else:
            print("Loaded formation energy model")
        if cgcnn_models['bg_val_mae'] is not None:
            print(f"Loaded band gap model (Val MAE: {cgcnn_models['bg_val_mae']:.4f})")
        else:
            print("Loaded band gap model")
        print("CGCNN models loaded successfully\n")

    # Compute match statistics
    results = compute_match_statistics(
        args.eval_gen_path,
        args.test_csv_path,
        cgcnn_models=cgcnn_models,
        symprec=args.symprec
    )

    # Print results
    print_results(results)

    # Save results
    if args.output:
        torch.save(results, args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute text prompt matching statistics for eval_gen format')
    parser.add_argument('eval_gen_path', type=str, help='Path to eval_gen_{label}.pt file')
    parser.add_argument('--test-csv-path', type=str, required=True, help='Path to test.csv with ground truth')
    parser.add_argument('--fe-model-path', type=str, help='Path to formation energy CGCNN model')
    parser.add_argument('--bg-model-path', type=str, help='Path to band gap CGCNN model')
    parser.add_argument('--symprec', type=float, default=0.1, help='Symmetry precision for spacegroup analysis')
    parser.add_argument('--output', type=str, help='Output file path for results')

    args = parser.parse_args()
    main(args)
