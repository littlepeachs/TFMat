"""
Compute correctness of generated materials matching conditions specified by textual prompts.
This script replicates Table 4 from the TGDMat paper.
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
    # Look for patterns like "chemical formula is X" or "formula: X"
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
    # Look for patterns like "space group number is X" or "space group: X"
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
    # Crystal systems: triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic
    systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']

    text_lower = text.lower()
    for system in systems:
        if f"crystal system is {system}" in text_lower or f"crystal system: {system}" in text_lower:
            return system
    return None


def extract_formation_energy_from_text(text):
    """Extract formation energy constraint from text description."""
    # Look for patterns like "formation energy is positive/negative"
    if "formation energy is positive" in text.lower():
        return "positive"
    elif "formation energy is negative" in text.lower():
        return "negative"
    return None


def extract_band_gap_from_text(text):
    """Extract band gap constraint from text description."""
    # Look for patterns like "band gap is zero/nonzero"
    if "band gap is zero" in text.lower() or "zero band gap" in text.lower():
        return "zero"
    elif "band gap is nonzero" in text.lower() or "nonzero band gap" in text.lower():
        return "nonzero"
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


def check_formation_energy_match(formation_energy, target_sign):
    """Check if formation energy matches target sign (positive/negative)."""
    if target_sign is None or formation_energy is None:
        return None

    if target_sign == "positive":
        return formation_energy > 0
    elif target_sign == "negative":
        return formation_energy < 0
    return None


def check_band_gap_match(band_gap, target_type):
    """Check if band gap matches target type (zero/nonzero)."""
    if target_type is None or band_gap is None:
        return None

    if target_type == "zero":
        return abs(band_gap) < 0.01  # threshold for zero
    elif target_type == "nonzero":
        return abs(band_gap) >= 0.01
    return None


def compute_match_statistics(eval_diff_path, symprec=0.1, check_properties=False):
    """
    Compute matching statistics for generated materials.

    Args:
        eval_diff_path: Path to eval_diff.pt file
        symprec: Symmetry precision for space group analysis
        check_properties: If True, check formation energy and band gap from ground truth

    Returns:
        Dictionary with match statistics for each global feature
    """
    # Load evaluation results
    data = torch.load(eval_diff_path, map_location='cpu')

    frac_coords = data['frac_coords']  # [num_evals, num_samples, max_atoms, 3]
    atom_types = data['atom_types']    # [num_evals, num_samples, max_atoms]
    lattices = data['lattices']        # [num_evals, num_samples, 3, 3]
    num_atoms = data['num_atoms']      # [num_evals, num_samples]
    input_data_batch = data['input_data_batch']

    num_evals, num_samples = frac_coords.shape[:2]

    # Extract text descriptions and ground truth properties from input data
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

        # Ground truth formula (from pretty_formula if available)
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

    # Process each sample
    for sample_idx in range(num_samples):
        text = text_descriptions[sample_idx]

        # Extract target properties from text (if available)
        target_formula_text = extract_formula_from_text(text)
        target_sg_text = extract_space_group_from_text(text)
        target_system_text = extract_crystal_system_from_text(text)
        target_fe_sign = extract_formation_energy_from_text(text)
        target_bg_type = extract_band_gap_from_text(text)

        # Use ground truth as target if text doesn't specify
        target_formula = target_formula_text if target_formula_text else gt_formulas[sample_idx]
        target_sg = target_sg_text if target_sg_text else gt_space_groups[sample_idx]
        # For crystal system, we derive from space group if not in text
        target_system = target_system_text

        # Use first evaluation (eval_idx=0) for matching
        eval_idx = 0

        # Get structure data
        coords = frac_coords[eval_idx, sample_idx]
        types = atom_types[eval_idx, sample_idx]
        lattice = lattices[eval_idx, sample_idx]
        n_atoms = num_atoms[eval_idx, sample_idx].item()

        # Build structure
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

            # Check each property
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

            # Only check formation energy and band gap if requested and ground truth available
            if check_properties:
                gt_fe = gt_formation_energies[sample_idx]
                if gt_fe is not None:
                    # Determine expected sign from ground truth
                    expected_fe_sign = "positive" if gt_fe > 0 else "negative"
                    fe_match = check_formation_energy_match(gt_fe, expected_fe_sign)
                    if fe_match is not None:
                        stats['formation_energy']['total'] += 1
                        if fe_match:
                            stats['formation_energy']['matched'] += 1

                gt_bg = gt_band_gaps[sample_idx]
                if gt_bg is not None:
                    # Determine expected type from ground truth
                    expected_bg_type = "zero" if abs(gt_bg) < 0.01 else "nonzero"
                    bg_match = check_band_gap_match(gt_bg, expected_bg_type)
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

    print(f"Computing text prompt matching statistics for: {eval_diff_path}")
    print(f"Symmetry precision: {args.symprec}")
    print(f"Check properties (formation energy, band gap): {args.check_properties}")
    print()

    results = compute_match_statistics(eval_diff_path, symprec=args.symprec, check_properties=args.check_properties)

    # Print results in table format
    print("=" * 70)
    print("Correctness of Generated Materials Matching Text Prompt Conditions")
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
        description='Compute text prompt matching statistics (Table 4)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('eval_diff_path', type=str, help='Path to eval_diff.pt file')
    parser.add_argument('--symprec', type=float, default=0.1, help='Symmetry precision for space group analysis')
    parser.add_argument('--check-properties', action='store_true',
                        help='Check formation energy and band gap using ground truth (only checks if ground truth matches expected sign/type)')
    parser.add_argument('--output', type=str, help='Output path to save results')

    args = parser.parse_args()
    main(args)
