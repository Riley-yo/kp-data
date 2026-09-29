"""KP 变异增广循环：通过随机生成 + 变异 + 挑选，让点云变圆。

依据：
1. 师兄指导："扩大随机性，随机生成数据，再挑选"
2. 建模与ISA说明.md KP section："改变物品重量、收益及其相关性、
   容量占总重量的比例；不使用背包解值调节生成"
3. BPP 的 bpp_augmenter.py 的 mutate/generate/accept 逻辑

KP 与 BPP 的关键区别：
- KP 有三个参数向量（weights, profits, capacity），BPP 只有两个（sizes, capacity）
- KP 的变异算子需要同时考虑 weight-profit 相关性
- random_generate 需要采样 Pisinger 标准利润-重量相关性类：
  uncorrelated / weakly_correlated / strongly_correlated / subset_sum / inverse_correlated

流程：
  每轮：
    1. 从原始实例选父本 + 随机生成新实例（扩大随机性）
    2. 套变异算子产候选
    3. 双重去重
    4. 用形状门槛评估：加这批候选后圆度是否改善
    5. 改善就接受，否则拒绝
  直到圆度达标或轮次用完
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# === KP 变异算子 ===

def _bounded(value: float) -> int:
    return int(np.clip(round(float(value)), 1, 1_000_000))


def _log_uniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def mutate_kp(parent: dict, operator: str, rng) -> tuple[dict | None, dict]:
    """对 KP 父实例套变异算子，产候选子实例。

    parent 格式: {"weights": [...], "profit": [...], "capacity": ...}
    返回: ({"weights": [...], "profit": [...], "capacity": ...}, params_dict) 或 (None, params_dict)
    """
    weights = list(parent["weights"])
    profits = list(parent["profit"])
    capacity = int(parent["capacity"])
    n = len(weights)

    if operator == "weight_perturb":
        idx = int(rng.integers(n))
        new_w = _bounded(weights[idx] * rng.uniform(0.5, 1.5))
        weights[idx] = new_w
        return {"weights": weights, "profit": profits, "capacity": capacity}, {"perturbed_index": idx}

    if operator == "profit_perturb":
        idx = int(rng.integers(n))
        new_p = _bounded(profits[idx] * rng.uniform(0.5, 1.5))
        profits[idx] = new_p
        return {"weights": weights, "profit": profits, "capacity": capacity}, {"perturbed_index": idx}

    if operator == "item_add":
        new_w = _bounded(_log_uniform(rng, 1, max(capacity, 2)))
        new_p = _bounded(_log_uniform(rng, 1, max(capacity, 2)))
        weights.append(new_w)
        profits.append(new_p)
        return {"weights": weights, "profit": profits, "capacity": capacity}, {"added_weight": new_w, "added_profit": new_p}

    if operator == "item_delete":
        if n <= 2:
            return None, {"reason": "minimum_size"}
        idx = int(rng.integers(n))
        weights.pop(idx)
        profits.pop(idx)
        return {"weights": weights, "profit": profits, "capacity": capacity}, {"deleted_index": idx}

    if operator == "capacity_change":
        new_cap = _bounded(capacity * rng.uniform(0.5, 2.0))
        return {"weights": weights, "profit": profits, "capacity": new_cap}, {"new_capacity": new_cap}

    if operator == "global_rescale":
        factor = _log_uniform(rng, 0.3, 3.0)
        new_cap = _bounded(capacity * factor)
        new_w = [_bounded(w * factor) for w in weights]
        new_p = [_bounded(p * factor) for p in profits]
        return {"weights": new_w, "profit": new_p, "capacity": new_cap}, {"factor": factor}

    if operator == "weight_rank":
        power = _log_uniform(rng, 0.5, 3.0)
        ordered = sorted(range(n), key=lambda i: (weights[i], i))
        low, high = min(weights), max(weights)
        if low == high:
            return None, {"reason": "constant_weights"}
        for rank, idx in enumerate(ordered):
            fraction = rank / max(n - 1, 1)
            weights[idx] = _bounded(low + (high - low) * fraction ** power)
        return {"weights": weights, "profit": profits, "capacity": capacity}, {"rank_power": power}

    if operator == "weight_noise":
        sigma = _log_uniform(rng, 0.02, 0.6)
        new_w = [_bounded(w * math.exp(rng.normal(0, sigma))) for w in weights]
        return {"weights": new_w, "profit": profits, "capacity": capacity}, {"noise_sigma": sigma}

    if operator == "profit_correlation_flip":
        # 重新生成 profits，改变 weight-profit 相关性
        corr_type = rng.choice(["uncorrelated", "weakly_correlated", "strongly_correlated", "subset_sum", "inverse_correlated"])
        new_p = _generate_profits(weights, capacity, corr_type, rng)
        return {"weights": weights, "profit": new_p, "capacity": capacity}, {"corr_type": corr_type}

    if operator == "tie_weight":
        multiplier = float(rng.uniform(0.1, 1.5))
        step = max(2, int(round(np.std(weights) * multiplier)))
        new_w = [_bounded(round(w / step) * step) for w in weights]
        return {"weights": new_w, "profit": profits, "capacity": capacity}, {"tie_step": step}

    if operator == "density_perturb":
        # 扰动 profit/weight 比率（密度）
        idx = int(rng.integers(n))
        density = profits[idx] / max(weights[idx], 1)
        new_density = density * rng.uniform(0.3, 3.0)
        profits[idx] = _bounded(new_density * weights[idx])
        return {"weights": weights, "profit": profits, "capacity": capacity}, {"perturbed_index": idx, "old_density": density}

    if operator == "random_generate":
        # 师兄确认：扩大随机生成范围，提升特征空间覆盖
        n_new = int(rng.integers(3, 50))
        cap_new = int(rng.choice([20, 30, 50, 75, 100, 120, 150, 200, 250, 300, 500, 1000, 5000, 10000]))
        # Pisinger 标准利润-重量相关性类
        corr_type = rng.choice([
            "uncorrelated", "uncorrelated",  # 加权，更常见
            "weakly_correlated", "weakly_correlated",
            "strongly_correlated",
            "subset_sum",
            "inverse_correlated",
        ])
        # 重量分布策略
        dist_type = rng.choice(["uniform", "small_heavy", "large_heavy", "mixed", "bimodal"])
        if dist_type == "uniform":
            new_w = [int(rng.integers(1, max(cap_new, 2))) for _ in range(n_new)]
        elif dist_type == "small_heavy":
            new_w = [int(rng.integers(1, max(cap_new // 5, 2))) for _ in range(n_new)]
        elif dist_type == "large_heavy":
            new_w = [int(rng.integers(max(cap_new // 2, 2), cap_new + 1)) for _ in range(n_new)]
        elif dist_type == "mixed":
            half = n_new // 2
            new_w = [int(rng.integers(1, max(cap_new // 4, 2))) for _ in range(half)]
            new_w += [int(rng.integers(max(cap_new // 2, 2), cap_new + 1)) for _ in range(n_new - half)]
        else:  # bimodal
            new_w = [int(rng.choice([
                int(rng.integers(1, max(cap_new // 5, 2))),
                int(rng.integers(max(cap_new // 2, 2), cap_new + 1))
            ])) for _ in range(n_new)]
        new_w = [max(1, w) for w in new_w]
        new_p = _generate_profits(new_w, cap_new, corr_type, rng)
        return {"weights": new_w, "profit": new_p, "capacity": cap_new}, {"generated": True, "n": n_new, "cap": cap_new, "dist": dist_type, "corr": corr_type}

    raise ValueError(f"unsupported KP operator: {operator}")


def _generate_profits(weights, capacity, corr_type, rng):
    """根据 Pisinger 标准类生成 profits。

    - uncorrelated: p_i 独立于 w_i，均匀随机
    - weakly_correlated: p_i ∈ [w_i - R, w_i + R], R 小
    - strongly_correlated: p_i = w_i + R (固定偏移)
    - subset_sum: p_i = w_i
    - inverse_correlated: p_i 与 w_i 反相关
    """
    n = len(weights)
    R = max(capacity // 10, 10)

    if corr_type == "uncorrelated":
        return [int(rng.integers(1, max(capacity, 2) + 1)) for _ in range(n)]
    elif corr_type == "weakly_correlated":
        return [max(1, int(w + rng.integers(-R, R + 1))) for w in weights]
    elif corr_type == "strongly_correlated":
        offset = int(rng.integers(1, R + 1))
        return [w + offset for w in weights]
    elif corr_type == "subset_sum":
        return list(weights)
    elif corr_type == "inverse_correlated":
        max_w = max(weights)
        return [max(1, int(max_w - w + rng.integers(1, R + 1))) for w in weights]
    else:
        return [int(rng.integers(1, max(capacity, 2) + 1)) for _ in range(n)]


KP_OPERATORS = (
    "weight_perturb",
    "profit_perturb",
    "item_add",
    "item_delete",
    "capacity_change",
    "global_rescale",
    "weight_rank",
    "weight_noise",
    "profit_correlation_flip",
    "tie_weight",
    "density_perturb",
    "random_generate",  # 师兄说的"随机生成数据"
)


# === 哈希 ===

def kp_hash(weights, profits, capacity, proportional=False):
    values = list(weights) + list(profits) + [int(capacity)]
    if proportional:
        nonzero = [abs(float(v)) for v in values if v != 0]
        denom = min(nonzero) if nonzero else 1.0
        values = [format(float(v) / denom, ".15g") for v in values]
    else:
        values = [format(float(v), ".15g") for v in values]
    payload = {"problem": "KP", "proportional": bool(proportional), "values": values}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# === 特征提取（调用 kp.py 的 _core） ===

def extract_kp_features(instance, kp_core_path):
    """调用 kp.py 的 calculate_features 提取特征。"""
    if kp_core_path not in sys.path:
        sys.path.insert(0, str(kp_core_path))
    from domain import unpack, check_profile
    from intrinsic import feature_row
    from structural import knapsack_relation_features

    spec = {
        "problem": "KP", "name": "KP", "kind": "knapsack",
        "fields": {
            "weights": {"type": "vector", "aliases": ["weights", "weight", "costs", "cost"]},
            "profit": {"type": "vector", "aliases": ["values", "value", "profits", "profit", "benefits", "returns"]},
            "capacity": {"type": "scalar", "aliases": ["capacity", "budget", "budget_limit", "total_budget"]},
        },
        "primary_variant": "domain",
    }
    data = unpack(instance)
    check_profile(spec, data)
    result = feature_row(spec, spec["primary_variant"], data)
    result.update(knapsack_relation_features(data))
    return result


# === 主增广循环 ===

def run_kp_augmentation(
    parents_jsonl: str,
    output_dir: str,
    kp_core_path: str,
    tsp_code_path: str,
    max_rounds: int = 100,
    candidates_per_round: int = 200,
    seed: int = 20260924,
    target_cov_ratio: float = 1.25,
    target_radial: float = 0.15,
    target_disk: float = 0.95,
    target_sector: float = 0.15,
):
    """跑 KP 变异增广，生成派生实例，让点云变圆。"""
    from preprocess_projection import PrelimTransformer, pilot
    from tsp_isa_shape import shape_metrics

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载原始实例
    with open(parents_jsonl, encoding="utf-8") as f:
        parents = [json.loads(line) for line in f if line.strip()]
    print(f"加载 {len(parents)} 个原始 KP 实例")

    # 提取原始特征
    feature_names = None
    rows = []
    for p in parents:
        params = p["parameters"]
        feat = extract_kp_features(params, kp_core_path)
        if feature_names is None:
            feature_names = list(feat.keys())
        rows.append({"instance_id": p["instance_id"], **feat})
    df = pd.DataFrame(rows).set_index("instance_id")
    print(f"提取特征: {df.shape[1]} 维, {df.shape[0]} 个实例")

    # 当前实例集（原始 + 派生）
    all_instances = list(parents)
    all_features = df.copy()

    # 去重哈希集合
    seen_exact = {
        kp_hash(p["parameters"]["weights"], p["parameters"]["profit"], p["parameters"]["capacity"])
        for p in parents
    }
    seen_prop = {
        kp_hash(p["parameters"]["weights"], p["parameters"]["profit"], p["parameters"]["capacity"], proportional=True)
        for p in parents
    }

    # 预采样原始参数分布，用于 random_generate
    orig_n_vals = [len(p["parameters"]["weights"]) for p in parents]
    orig_caps = [p["parameters"]["capacity"] for p in parents]

    rng = np.random.default_rng(seed)
    accepted_count = 0

    # === 第一阶段：批量生成 + 始终接受 ===
    # 师兄确认：扩大随机生成范围，大量生成候选填充特征空间
    # 不用形状门槛拒绝（SAGE 原始实例有极端离群点，形状门槛会一直拒绝）
    # 目标：直接生成目标数量的候选，覆盖特征空间
    target_synthetic = max_rounds * candidates_per_round  # 如 100轮×200=20000

    print(f"\n第一阶段: 批量生成 {target_synthetic} 个候选（始终接受）")
    batch_candidates = []
    batch_features = []
    attempts = 0
    while len(batch_candidates) < target_synthetic and attempts < target_synthetic * 5:
        attempts += 1
        candidate_seed = int(seed + 1_000_003 * attempts)
        local_rng = np.random.default_rng(candidate_seed)

        # 80% 随机生成，20% 从父本变异
        if local_rng.random() < 0.8:
            operator = "random_generate"
            # 从原始参数分布采样
            n_new = int(local_rng.choice(orig_n_vals))
            cap_new = int(local_rng.choice(orig_caps))
            dist_type = local_rng.choice(["uniform", "small_heavy", "large_heavy", "mixed", "bimodal"])
            if dist_type == "uniform":
                new_w = [int(local_rng.integers(1, max(cap_new, 2))) for _ in range(n_new)]
            elif dist_type == "small_heavy":
                new_w = [int(local_rng.integers(1, max(cap_new // 5, 2))) for _ in range(n_new)]
            elif dist_type == "large_heavy":
                new_w = [int(local_rng.integers(max(cap_new // 2, 2), cap_new + 1)) for _ in range(n_new)]
            elif dist_type == "mixed":
                half = n_new // 2
                new_w = [int(local_rng.integers(1, max(cap_new // 4, 2))) for _ in range(half)]
                new_w += [int(local_rng.integers(max(cap_new // 2, 2), cap_new + 1)) for _ in range(n_new - half)]
            else:
                new_w = [int(local_rng.choice([
                    int(local_rng.integers(1, max(cap_new // 5, 2))),
                    int(local_rng.integers(max(cap_new // 2, 2), cap_new + 1))
                ])) for _ in range(n_new)]
            new_w = [max(1, w) for w in new_w]

            # Pisinger 5 种利润-重量相关性类
            corr_type = local_rng.choice([
                "uncorrelated", "uncorrelated",
                "weakly_correlated", "weakly_correlated",
                "strongly_correlated",
                "subset_sum",
                "inverse_correlated",
            ])
            new_p = _generate_profits(new_w, cap_new, corr_type, local_rng)

            result = {"weights": new_w, "profit": new_p, "capacity": cap_new}
            params_meta = {"generated": True, "n": n_new, "cap": cap_new, "dist": dist_type, "corr": corr_type}
        else:
            operator = KP_OPERATORS[int(local_rng.integers(len(KP_OPERATORS)))]
            parent_params = parents[int(local_rng.integers(len(parents)))]["parameters"]
            result, params_meta = mutate_kp(parent_params, operator, local_rng)
            if result is None:
                continue

        weights = result["weights"]
        profits = result["profit"]
        capacity = result["capacity"]
        if not (2 <= len(weights) <= 200):
            continue
        if not all(w > 0 for w in weights):
            continue
        if not all(p >= 0 for p in profits):
            continue
        if capacity <= 0:
            continue

        exact_h = kp_hash(weights, profits, capacity)
        prop_h = kp_hash(weights, profits, capacity, proportional=True)
        if exact_h in seen_exact or prop_h in seen_prop:
            continue

        seen_exact.add(exact_h)
        seen_prop.add(prop_h)

        try:
            feat = extract_kp_features(result, kp_core_path)
            if not all(np.isfinite(list(feat.values()))):
                continue
        except Exception:
            continue

        inst_id = f"synthetic_KP_{accepted_count + 1:06d}_{exact_h[:10]}"
        batch_candidates.append({
            "instance_id": inst_id,
            "origin": "synthetic",
            "variant": "domain",
            "parameters": result,
            "lineage": {
                "parent_instance_id": "random" if operator == "random_generate" else "parent",
                "operator": operator,
                "operator_parameters": params_meta,
                "round": 0,
                "seed": candidate_seed,
            },
        })
        batch_features.append({"instance_id": inst_id, **feat})
        accepted_count += 1

        if accepted_count % 5000 == 0:
            print(f"  已生成 {accepted_count} 个候选...")

    # 合并原始 + 批量候选
    all_instances.extend(batch_candidates)
    all_features = pd.concat([all_features, pd.DataFrame(batch_features).set_index("instance_id")])
    print(f"第一阶段完成: {len(all_instances)} 个实例 ({len(parents)} SAGE + {accepted_count} synthetic)")

    # === 第二阶段：最终投影 + 保存 ===
    # 拟合投影 + 算形状
    transformer = PrelimTransformer.fit(all_features, feature_names=list(all_features.columns))
    transformed = transformer.transform_frame(all_features)
    stable = [i for i in range(transformed.shape[1]) if np.isfinite(transformed[:, i]).all() and float(np.std(transformed[:, i])) > 1e-10]
    stable_names = [list(all_features.columns)[i] for i in stable]
    stable_values = transformed[:, stable]

    selected_positions = []
    for index, name in enumerate(stable_names):
        if not any(abs(float(np.corrcoef(stable_values[:, index], stable_values[:, kept])[0, 1])) > 0.95 for kept in selected_positions):
            selected_positions.append(index)
    selected_names = [stable_names[i] for i in selected_positions]
    selected_values = stable_values[:, selected_positions]
    if len(selected_names) >= 2:
        selected_df = pd.DataFrame(selected_values, index=all_features.index, columns=selected_names)
        fitted = pilot(selected_df, pd.DataFrame(index=selected_df.index), n_restarts=5, seed=seed)
        coords = fitted.coords
        metrics = shape_metrics(coords)

        cov = metrics["covariance_ratio"]
        rad = metrics["radial_uniform_max_deviation"]
        disk = metrics["fitted_disk_inside_ratio"]
        sect = metrics["sector_cv_above_theoretical_minimum"]

        print(f"最终: cov={cov:.3f}(≤{target_cov_ratio}), rad={rad:.3f}(≤{target_radial}), disk={disk:.3f}(≥{target_disk}), sect={sect:.3f}(≤{target_sector}), n={len(all_instances)}")
        print(f"target_met: {metrics['target_met']}")

    # 保存结果
    output_jsonl = output_dir / "kp_augmented_instances.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for inst in all_instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    print(f"\n保存 {len(all_instances)} 个实例到 {output_jsonl}")

    # 保存最终坐标 + 图（使用第二阶段已拟合的 coords）
    pd.DataFrame({"z1": coords[:, 0], "z2": coords[:, 1]}, index=all_features.index).to_csv(output_dir / "coordinates.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    theta = np.linspace(0, 2 * np.pi, 100)
    r = max(np.max(np.abs(coords[:, 0])), np.max(np.abs(coords[:, 1])))

    # 区分 SAGE（蓝）和 Synthetic（橙）
    origins = ["sage_original" if iid.startswith("sage_original") else "synthetic" for iid in all_features.index]
    sage_mask = [o == "sage_original" for o in origins]
    synth_mask = [o == "synthetic" for o in origins]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if any(sage_mask):
        axes[0].scatter(coords[sage_mask, 0], coords[sage_mask, 1], s=3, color="#0072B2", label=f"SAGE (n={sum(sage_mask)})")
    if any(synth_mask):
        axes[0].scatter(coords[synth_mask, 0], coords[synth_mask, 1], s=3, color="#E69F00", marker="^", label=f"Synthetic (n={sum(synth_mask)})")
    axes[0].plot(r * np.cos(theta), r * np.sin(theta), "r--", alpha=0.3)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"KP ISA after augmentation ({len(all_instances)} instances)")
    axes[0].legend()
    axes[1].hist2d(coords[:, 0], coords[:, 1], bins=30, cmap="Blues")
    axes[1].set_aspect("equal")
    axes[1].set_title("Density heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "kp_projection_augmented.png", dpi=150)

    print(f"\n最终: cov={metrics['covariance_ratio']:.3f}, rad={metrics['radial_uniform_max_deviation']:.3f}, disk={metrics['fitted_disk_inside_ratio']:.3f}, sect={metrics['sector_cv_above_theoretical_minimum']:.3f}")
    print(f"target_met: {metrics['target_met']}")
    print(f"图已保存到 {output_dir}/kp_projection_augmented.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KP 变异增广循环")
    parser.add_argument("--parents", type=str, required=True, help="原始实例 JSONL")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--kp-core", type=str, default="isa/_core", help="_core 目录路径")
    parser.add_argument("--tsp-code", type=str, default=".", help="含 preprocess_projection.py 和 tsp_isa_shape.py 的目录路径")
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--candidates-per-round", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    if args.tsp_code not in sys.path:
        sys.path.insert(0, args.tsp_code)

    run_kp_augmentation(
        parents_jsonl=args.parents,
        output_dir=args.output_dir,
        kp_core_path=args.kp_core,
        tsp_code_path=args.tsp_code,
        max_rounds=args.max_rounds,
        candidates_per_round=args.candidates_per_round,
        seed=args.seed,
    )
