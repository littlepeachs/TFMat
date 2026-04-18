#!/usr/bin/env python3
"""
检查预测结果中的异常值
"""

import torch
import numpy as np
from pathlib import Path

# 读取保存的结果文件
result_file = Path('top2000_element_and_property_match_results.txt')

if result_file.exists():
    with open(result_file, 'r') as f:
        lines = f.readlines()

    fe_values = []
    bg_values = []
    fe_errors = []
    bg_errors = []

    for i, line in enumerate(lines):
        if 'Formation Energy: GT=' in line:
            # 解析 GT 和 Pred 值
            parts = line.split('GT=')[1].split(',')
            gt_fe = float(parts[0])

            pred_part = parts[1].split('Pred=')[1].split(',')[0]
            if pred_part != 'N/A':
                pred_fe = float(pred_part)
                fe_values.append((gt_fe, pred_fe))
                fe_errors.append(abs(pred_fe - gt_fe))

        if 'Band Gap: GT=' in line:
            parts = line.split('GT=')[1].split(',')
            gt_bg = float(parts[0])

            pred_part = parts[1].split('Pred=')[1].split(',')[0]
            if pred_part != 'N/A':
                pred_bg = float(pred_part)
                bg_values.append((gt_bg, pred_bg))
                bg_errors.append(abs(pred_bg - gt_bg))

    # 分析 Formation Energy
    print("="*80)
    print("Formation Energy Analysis")
    print("="*80)
    print(f"Total samples: {len(fe_values)}")

    if fe_errors:
        fe_errors = np.array(fe_errors)
        print(f"\nMAE: {np.mean(fe_errors):.4f} eV/atom")
        print(f"Median AE: {np.median(fe_errors):.4f} eV/atom")
        print(f"Min error: {np.min(fe_errors):.4f} eV/atom")
        print(f"Max error: {np.max(fe_errors):.4f} eV/atom")
        print(f"Std: {np.std(fe_errors):.4f} eV/atom")

        # 找出最大的几个误差
        sorted_indices = np.argsort(fe_errors)[::-1]
        print(f"\nTop 10 largest errors:")
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            gt, pred = fe_values[idx]
            error = fe_errors[idx]
            print(f"  #{i+1}: GT={gt:.4f}, Pred={pred:.4f}, Error={error:.4f}")

        # 统计异常值
        threshold = 10.0  # 超过10 eV/atom的误差视为异常
        outliers = fe_errors > threshold
        print(f"\nOutliers (error > {threshold} eV/atom): {np.sum(outliers)} ({np.sum(outliers)/len(fe_errors)*100:.2f}%)")

        if np.sum(outliers) > 0:
            print(f"MAE without outliers: {np.mean(fe_errors[~outliers]):.4f} eV/atom")

    # 分析 Band Gap
    print("\n" + "="*80)
    print("Band Gap Analysis")
    print("="*80)
    print(f"Total samples: {len(bg_values)}")

    if bg_errors:
        bg_errors = np.array(bg_errors)
        print(f"\nMAE: {np.mean(bg_errors):.4f} eV")
        print(f"Median AE: {np.median(bg_errors):.4f} eV")
        print(f"Min error: {np.min(bg_errors):.4f} eV")
        print(f"Max error: {np.max(bg_errors):.4f} eV")
        print(f"Std: {np.std(bg_errors):.4f} eV")

        # 找出最大的几个误差
        sorted_indices = np.argsort(bg_errors)[::-1]
        print(f"\nTop 10 largest errors:")
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            gt, pred = bg_values[idx]
            error = bg_errors[idx]
            print(f"  #{i+1}: GT={gt:.4f}, Pred={pred:.4f}, Error={error:.4f}")

else:
    print("Result file not found!")
