#!/usr/bin/env python3
"""
从前5000个样本中选择元素组成匹配分数最高的2000个样本
然后使用CGCNN预测formation energy和band gap，计算匹配率
"""

import os
import re
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from torch_geometric.data import Data, DataLoader
import sys

# Import CGCNN model
sys.path.insert(0, 'scripts')
from train_cgcnn_for_properties import SimpleCGCNN


def parse_cif_filename(filename):
    """解析 CIF 文件名"""
    match = re.match(r'(\d+)_(mp-\d+)_(.+)\.cif', filename)
    if match:
        return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None


def extract_elements_from_cif(cif_path):
    """从 CIF 文件中提取元素列表"""
    elements = set()
    try:
        with open(cif_path, 'r') as f:
            for line in f:
                if line.startswith('_chemical_formula_sum'):
                    formula_str = line.split("'")[1] if "'" in line else line.split()[1]
                    element_pattern = r'([A-Z][a-z]?)\d*'
                    elements = set(re.findall(element_pattern, formula_str))
                    break
    except Exception as e:
        print(f"Error reading {cif_path}: {e}")
    return elements


def extract_elements_from_formula(formula_str):
    """从化学式字符串中提取元素列表"""
    element_pattern = r'([A-Z][a-z]?)\d*'
    elements = set(re.findall(element_pattern, formula_str))
    return elements


def compute_match_score(gen_elements, gt_elements):
    """计算匹配分数"""
    if len(gt_elements) == 0:
        return 0.0
    matched = gen_elements & gt_elements
    return len(matched) / len(gt_elements)


def load_structure_from_cif(cif_path):
    """从CIF文件加载pymatgen Structure"""
    try:
        structure = Structure.from_file(str(cif_path))
        return structure
    except Exception as e:
        print(f"Error loading structure from {cif_path}: {e}")
        return None


def structure_to_graph(structure, radius=8.0):
    """将pymatgen Structure转换为PyG Data对象"""
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


def load_cgcnn_models(fe_model_path, bg_model_path):
    """加载CGCNN模型"""
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

    return fe_model, bg_model, device


def predict_properties(structures, fe_model, bg_model, device, batch_size=32):
    """使用CGCNN预测formation energy和band gap"""
    fe_predictions = []
    bg_predictions = []

    # Convert structures to graphs
    data_list = []
    valid_indices = []
    for i, structure in enumerate(tqdm(structures, desc="Converting to graphs")):
        if structure is None:
            data_list.append(None)
            continue
        try:
            data = structure_to_graph(structure)
            data_list.append(data)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error converting structure {i}: {e}")
            data_list.append(None)

    # Predict in batches
    valid_data = [data_list[i] for i in valid_indices]
    loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        # Predict formation energies
        fe_preds = []
        for batch in tqdm(loader, desc="Predicting formation energy"):
            batch = batch.to(device)
            out = fe_model(batch)
            fe_preds.extend(out.cpu().numpy())

        # Predict band gaps
        bg_preds = []
        loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        for batch in tqdm(loader, desc="Predicting band gap"):
            batch = batch.to(device)
            out = bg_model(batch)
            bg_preds.extend(out.cpu().numpy())

    # Map back to original order
    fe_predictions = [None] * len(structures)
    bg_predictions = [None] * len(structures)
    for i, idx in enumerate(valid_indices):
        fe_predictions[idx] = fe_preds[i]
        bg_predictions[idx] = bg_preds[i]

    return fe_predictions, bg_predictions


def main():
    # 路径配置
    cif_dir = Path('generated_crystals/text_lattice5_epoch1834_targeted__s100_a10_coords_gfauto/all_cif')
    test_csv_path = Path('data_text/mp_20/test.csv')
    fe_model_path = Path('cgcnn_models/cgcnn_formation_energy_per_atom_best.pth')
    bg_model_path = Path('cgcnn_models/cgcnn_band_gap_best.pth')

    # 读取测试集
    print("Loading test.csv...")
    df_test = pd.read_csv(test_csv_path)
    df_test_subset = df_test.head(5000)

    # 创建 material_id 到 ground truth 的映射
    gt_dict = {}
    for idx, row in df_test_subset.iterrows():
        material_id = row['material_id']
        formula = row['pretty_formula']
        gt_elements = extract_elements_from_formula(formula)
        gt_dict[material_id] = {
            'formula': formula,
            'elements': gt_elements,
            'formation_energy': row['formation_energy_per_atom'],
            'band_gap': row['band_gap'],
            'csv_index': idx
        }

    # 第一步：计算所有样本的元素组成匹配分数
    print("\nStep 1: Computing element composition match scores for 5000 samples...")
    cif_files = sorted(cif_dir.glob('*.cif'))[:5000]

    candidates = []
    for cif_file in tqdm(cif_files, desc="Processing CIF files"):
        index, material_id, _ = parse_cif_filename(cif_file.name)
        if material_id is None or material_id not in gt_dict:
            continue

        gen_elements = extract_elements_from_cif(cif_file)
        gt_info = gt_dict[material_id]
        score = compute_match_score(gen_elements, gt_info['elements'])

        candidates.append({
            'index': index,
            'material_id': material_id,
            'cif_path': cif_file,
            'score': score,
            'gt_info': gt_info
        })

    # 第二步：选择分数最高的2000个样本
    print(f"\nStep 2: Selecting top 2000 samples by match score...")
    candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)
    top_2000 = candidates_sorted[:2000]

    print(f"Selected 2000 samples with scores ranging from {top_2000[-1]['score']:.4f} to {top_2000[0]['score']:.4f}")

    # 统计选中样本的分数分布
    scores_top2000 = [c['score'] for c in top_2000]
    print(f"Average score of top 2000: {np.mean(scores_top2000):.4f}")
    print(f"Score distribution:")
    print(f"  1.0 (perfect): {sum(1 for s in scores_top2000 if s == 1.0)}")
    print(f"  0.67-1.0: {sum(1 for s in scores_top2000 if 0.67 <= s < 1.0)}")
    print(f"  0.33-0.67: {sum(1 for s in scores_top2000 if 0.33 <= s < 0.67)}")
    print(f"  0-0.33: {sum(1 for s in scores_top2000 if 0 < s < 0.33)}")
    print(f"  0 (no match): {sum(1 for s in scores_top2000 if s == 0)}")

    # 第三步：加载CGCNN模型
    print(f"\nStep 3: Loading CGCNN models...")
    fe_model, bg_model, device = load_cgcnn_models(fe_model_path, bg_model_path)
    print(f"Models loaded on device: {device}")

    # 第四步：加载结构并预测性质
    print(f"\nStep 4: Loading structures and predicting properties...")
    structures = []
    for candidate in tqdm(top_2000, desc="Loading structures"):
        structure = load_structure_from_cif(candidate['cif_path'])
        structures.append(structure)

    fe_predictions, bg_predictions = predict_properties(structures, fe_model, bg_model, device)

    # 第五步：计算匹配率和MAE
    print(f"\nStep 5: Computing property match rates and MAE...")
    fe_sign_matches = 0
    fe_total = 0
    bg_type_matches = 0
    bg_total = 0

    fe_errors = []
    bg_errors = []

    results = []
    for i, candidate in enumerate(top_2000):
        gt_info = candidate['gt_info']
        pred_fe = fe_predictions[i]
        pred_bg = bg_predictions[i]
        gt_fe = gt_info['formation_energy']
        gt_bg = gt_info['band_gap']

        result = {
            'material_id': candidate['material_id'],
            'element_score': candidate['score'],
            'gt_fe': gt_fe,
            'pred_fe': pred_fe,
            'gt_bg': gt_bg,
            'pred_bg': pred_bg,
            'fe_sign_match': None,
            'bg_type_match': None
        }

        # Formation energy sign match and MAE
        if pred_fe is not None:
            fe_sign_match = (pred_fe >= 0) == (gt_fe >= 0)
            result['fe_sign_match'] = fe_sign_match
            if fe_sign_match:
                fe_sign_matches += 1
            fe_total += 1
            fe_errors.append(abs(pred_fe - gt_fe))

        # Band gap type match (zero vs nonzero) and MAE
        if pred_bg is not None:
            bg_type_match = (abs(pred_bg) < 0.01) == (abs(gt_bg) < 0.01)
            result['bg_type_match'] = bg_type_match
            if bg_type_match:
                bg_type_matches += 1
            bg_total += 1
            bg_errors.append(abs(pred_bg - gt_bg))

        results.append(result)

    # Calculate MAE
    fe_mae = np.mean(fe_errors) if fe_errors else 0.0
    bg_mae = np.mean(bg_errors) if bg_errors else 0.0

    # 打印最终结果
    print("\n" + "="*80)
    print("FINAL RESULTS (Top 2000 samples by element composition match)")
    print("="*80)
    print(f"\nElement Composition:")
    print(f"  Average match score: {np.mean(scores_top2000):.4f}")
    print(f"\nFormation Energy:")
    print(f"  Sign match: {fe_sign_matches}/{fe_total} ({fe_sign_matches/fe_total*100:.2f}%)")
    print(f"  MAE: {fe_mae:.4f} eV/atom")
    print(f"\nBand Gap:")
    print(f"  Type match (zero/nonzero): {bg_type_matches}/{bg_total} ({bg_type_matches/bg_total*100:.2f}%)")
    print(f"  MAE: {bg_mae:.4f} eV")
    print("="*80)

    # 保存结果
    output_path = 'top2000_element_and_property_match_results.txt'
    with open(output_path, 'w') as f:
        f.write("Top 2000 Samples - Element Composition and Property Match Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Element Composition Average Score: {np.mean(scores_top2000):.4f}\n")
        f.write(f"Formation Energy Sign Match: {fe_sign_matches}/{fe_total} ({fe_sign_matches/fe_total*100:.2f}%)\n")
        f.write(f"Formation Energy MAE: {fe_mae:.4f} eV/atom\n")
        f.write(f"Band Gap Type Match: {bg_type_matches}/{bg_total} ({bg_type_matches/bg_total*100:.2f}%)\n")
        f.write(f"Band Gap MAE: {bg_mae:.4f} eV\n\n")
        f.write("="*80 + "\n")
        f.write("Detailed Results:\n")
        f.write("="*80 + "\n\n")
        for r in results:
            f.write(f"Material ID: {r['material_id']}\n")
            f.write(f"  Element Score: {r['element_score']:.4f}\n")
            pred_fe_str = f"{r['pred_fe']:.4f}" if r['pred_fe'] is not None else 'N/A'
            pred_bg_str = f"{r['pred_bg']:.4f}" if r['pred_bg'] is not None else 'N/A'
            f.write(f"  Formation Energy: GT={r['gt_fe']:.4f}, Pred={pred_fe_str}, Match={r['fe_sign_match']}\n")
            f.write(f"  Band Gap: GT={r['gt_bg']:.4f}, Pred={pred_bg_str}, Match={r['bg_type_match']}\n\n")

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
