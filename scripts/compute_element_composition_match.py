#!/usr/bin/env python3
"""
计算生成晶体与真实晶体的元素组成匹配率
使用部分匹配评分：matched_elements / total_gt_elements
"""

import os
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_cif_filename(filename):
    """
    解析 CIF 文件名，提取 index 和 material_id
    格式: {index}_{material_id}_{formula}.cif
    例如: 00000_mp-10009_Al6In4TeP.cif
    """
    match = re.match(r'(\d+)_(mp-\d+)_(.+)\.cif', filename)
    if match:
        index = int(match.group(1))
        material_id = match.group(2)
        formula = match.group(3)
        return index, material_id, formula
    return None, None, None


def extract_elements_from_cif(cif_path):
    """
    从 CIF 文件中提取元素列表
    读取 _chemical_formula_sum 行
    """
    elements = set()
    try:
        with open(cif_path, 'r') as f:
            for line in f:
                if line.startswith('_chemical_formula_sum'):
                    # 格式: _chemical_formula_sum   'Al6 In4 Te1 P1'
                    formula_str = line.split("'")[1] if "'" in line else line.split()[1]
                    # 提取元素符号（忽略数字）
                    element_pattern = r'([A-Z][a-z]?)\d*'
                    elements = set(re.findall(element_pattern, formula_str))
                    break
    except Exception as e:
        print(f"Error reading {cif_path}: {e}")
    return elements


def extract_elements_from_formula(formula_str):
    """
    从化学式字符串中提取元素列表
    例如: 'GaTe' -> {'Ga', 'Te'}
          'Li4Mn5O11' -> {'Li', 'Mn', 'O'}
    """
    element_pattern = r'([A-Z][a-z]?)\d*'
    elements = set(re.findall(element_pattern, formula_str))
    return elements


def compute_match_score(gen_elements, gt_elements):
    """
    计算匹配分数
    score = len(matched_elements) / len(gt_elements)
    """
    if len(gt_elements) == 0:
        return 0.0

    matched = gen_elements & gt_elements
    score = len(matched) / len(gt_elements)
    return score


def main():
    # 路径配置
    cif_dir = Path('generated_crystals/text_lattice5_epoch1834_targeted__s100_a10_coords_gfauto/all_cif')
    test_csv_path = Path('data_text/mp_20/test.csv')

    # 读取测试集 CSV
    print("Loading test.csv...")
    df_test = pd.read_csv(test_csv_path)
    print(f"Total test samples: {len(df_test)}")

    # 只处理前5000个
    num_samples = 5000
    df_test_subset = df_test.head(num_samples)

    # 创建 material_id 到 ground truth 的映射
    gt_dict = {}
    for idx, row in df_test_subset.iterrows():
        material_id = row['material_id']
        formula = row['pretty_formula']
        gt_elements = extract_elements_from_formula(formula)
        gt_dict[material_id] = {
            'formula': formula,
            'elements': gt_elements,
            'csv_index': idx
        }

    print(f"\nProcessing first {num_samples} samples...")

    # 收集所有 CIF 文件（前5000个）
    cif_files = sorted(cif_dir.glob('*.cif'))[:num_samples]
    print(f"Found {len(cif_files)} CIF files")

    # 统计结果
    results = []
    score_distribution = defaultdict(int)

    matched_count = 0
    not_found_count = 0

    for cif_file in cif_files:
        # 解析文件名
        index, material_id, formula_from_name = parse_cif_filename(cif_file.name)

        if material_id is None:
            print(f"Warning: Cannot parse filename {cif_file.name}")
            continue

        # 从 CIF 文件中提取元素
        gen_elements = extract_elements_from_cif(cif_file)

        # 查找对应的 ground truth
        if material_id not in gt_dict:
            not_found_count += 1
            print(f"Warning: {material_id} not found in test.csv subset")
            continue

        gt_info = gt_dict[material_id]
        gt_elements = gt_info['elements']
        gt_formula = gt_info['formula']

        # 计算匹配分数
        score = compute_match_score(gen_elements, gt_elements)

        # 记录结果
        result = {
            'index': index,
            'material_id': material_id,
            'gt_formula': gt_formula,
            'gt_elements': sorted(gt_elements),
            'gen_elements': sorted(gen_elements),
            'matched_elements': sorted(gen_elements & gt_elements),
            'score': score
        }
        results.append(result)

        # 统计分数分布
        if score == 1.0:
            score_distribution['1.0 (perfect)'] += 1
        elif score >= 0.67:
            score_distribution['0.67-1.0'] += 1
        elif score >= 0.33:
            score_distribution['0.33-0.67'] += 1
        elif score > 0:
            score_distribution['0-0.33'] += 1
        else:
            score_distribution['0 (no match)'] += 1

        matched_count += 1

    # 计算统计信息
    scores = [r['score'] for r in results]
    avg_score = np.mean(scores) if scores else 0.0

    # 打印结果
    print("\n" + "="*80)
    print("ELEMENT COMPOSITION MATCH RESULTS (First 5000 samples)")
    print("="*80)
    print(f"\nTotal processed: {matched_count}/{num_samples}")
    print(f"Not found in test.csv: {not_found_count}")
    print(f"\nAverage match score: {avg_score:.4f}")
    print(f"\nScore distribution:")
    for category in ['1.0 (perfect)', '0.67-1.0', '0.33-0.67', '0-0.33', '0 (no match)']:
        count = score_distribution[category]
        percentage = count / matched_count * 100 if matched_count > 0 else 0
        print(f"  {category:20s}: {count:5d} ({percentage:5.2f}%)")

    # 显示一些示例
    print("\n" + "="*80)
    print("EXAMPLES:")
    print("="*80)

    # 完全匹配的例子
    perfect_matches = [r for r in results if r['score'] == 1.0]
    if perfect_matches:
        print("\n[Perfect matches (score = 1.0)]")
        for r in perfect_matches[:3]:
            print(f"  {r['material_id']}: GT={r['gt_elements']} | Gen={r['gen_elements']}")

    # 部分匹配的例子
    partial_matches = [r for r in results if 0 < r['score'] < 1.0]
    if partial_matches:
        print("\n[Partial matches (0 < score < 1.0)]")
        for r in partial_matches[:3]:
            print(f"  {r['material_id']}: GT={r['gt_elements']} | Gen={r['gen_elements']} | Matched={r['matched_elements']} | Score={r['score']:.2f}")

    # 完全不匹配的例子
    no_matches = [r for r in results if r['score'] == 0.0]
    if no_matches:
        print("\n[No matches (score = 0.0)]")
        for r in no_matches[:3]:
            print(f"  {r['material_id']}: GT={r['gt_elements']} | Gen={r['gen_elements']}")

    # 保存详细结果
    output_path = 'element_composition_match_results_5000.txt'
    with open(output_path, 'w') as f:
        f.write("Element Composition Match Results (First 5000 samples)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average match score: {avg_score:.4f}\n\n")
        f.write("Score distribution:\n")
        for category in ['1.0 (perfect)', '0.67-1.0', '0.33-0.67', '0-0.33', '0 (no match)']:
            count = score_distribution[category]
            percentage = count / matched_count * 100 if matched_count > 0 else 0
            f.write(f"  {category:20s}: {count:5d} ({percentage:5.2f}%)\n")
        f.write("\n" + "="*80 + "\n")
        f.write("Detailed results:\n")
        f.write("="*80 + "\n\n")
        for r in results:
            f.write(f"Index: {r['index']:05d} | Material ID: {r['material_id']}\n")
            f.write(f"  GT Formula: {r['gt_formula']}\n")
            f.write(f"  GT Elements: {r['gt_elements']}\n")
            f.write(f"  Gen Elements: {r['gen_elements']}\n")
            f.write(f"  Matched: {r['matched_elements']}\n")
            f.write(f"  Score: {r['score']:.4f}\n\n")

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
