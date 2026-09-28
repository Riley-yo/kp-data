"""KP 4种formulation的COPT求解脚本。

对每个KP实例，用COPT跑4种建模（物品选择/容量状态弧流/完整可行模式选择/精确罚项QUBO），
记录求解时间、是否最优、目标值，生成 performance.csv 和 good.csv。

依据：建模与ISA说明.md KP section + 4种formulation数学公式。

用法（服务器上）：
  python solve_kp_formulations.py --instances KP_原始实例.jsonl --output-dir kp_perf --time-limit 60
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any


# === 实例加载 ===

def load_instances(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# === Formulation 1: 物品选择（标准 0/1 背包） ===
# x_i = 物品 i 是否选中（二进制）
# max sum(p_i * x_i)
# s.t. sum(w_i * x_i) <= C

def solve_compact(weights, profits, capacity, coptpy, time_limit):
    n = len(weights)
    env = coptpy.Envr()
    model = env.createModel("kp_compact")
    model.param.TimeLimit = float(time_limit)
    model.param.Threads = 4

    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype=coptpy.COPT.BINARY, name=f"x_{i}")

    # 目标: max sum(p_i * x_i)
    model.setObjective(
        coptpy.LinExpr([(x[i], float(profits[i])) for i in range(n)]),
        sense=coptpy.COPT.MAXIMIZE,
    )

    # 容量约束
    model.addConstr(
        coptpy.LinExpr([(x[i], float(weights[i])) for i in range(n)]),
        coptpy.COPT.LESS_EQUAL, float(capacity), name="capacity",
    )

    model.solve()
    status = model.status
    is_optimal = (status == coptpy.COPT.OPTIMAL)
    obj = float(model.ObjVal) if is_optimal else -1
    del model
    del env
    return is_optimal, obj, status


# === Formulation 2: 容量状态弧流（DP 网络流） ===
# 网络流：节点 (i, d)，i=物品层(0..n)，d=容量状态(0..C)
# 弧：选物品 i 的弧 ((i-1,d)->(i,d+w_i)) 和不选的弧 ((i-1,d)->(i,d))
# 源点 o=(0,0)，汇点 t=(n, d) for all d
# 二元弧流 f_a，流守恒

def solve_arc_flow(weights, profits, capacity, coptpy, time_limit):
    n = len(weights)
    # 节点编号: (i, d) -> i*(capacity+1) + d
    def node_id(i, d):
        return i * (capacity + 1) + d

    source = node_id(0, 0)
    # 汇点：虚拟节点
    sink = (n + 1) * (capacity + 1)
    num_nodes = (n + 1) * (capacity + 1) + 1

    # 收集弧: (from_node, to_node, profit, label)
    # label = -1 表示不选物品, label = i 表示选物品 i（1-indexed）
    arcs = []
    for i in range(1, n + 1):
        w = weights[i - 1]
        p = profits[i - 1]
        for d in range(capacity + 1):
            # 不选物品 i
            arcs.append((node_id(i - 1, d), node_id(i, d), 0.0, 0))
            # 选物品 i
            if d + w <= capacity:
                arcs.append((node_id(i - 1, d), node_id(i, d + w), float(p), i))

    # 汇点弧：(n, d) -> sink, profit=0
    for d in range(capacity + 1):
        arcs.append((node_id(n, d), sink, 0.0, 0))

    env = coptpy.Envr()
    model = env.createModel("kp_arc_flow")
    model.param.TimeLimit = float(time_limit)
    model.param.Threads = 4

    # 二元弧流变量
    f = {}
    for idx, (u, v, profit, label) in enumerate(arcs):
        f[idx] = model.addVar(
            vtype=coptpy.COPT.BINARY,
            obj=profit,
            name=f"f_{idx}",
        )

    # 出入弧索引
    out_arcs = {i: [] for i in range(num_nodes)}
    in_arcs = {i: [] for i in range(num_nodes)}
    for idx, (u, v, profit, label) in enumerate(arcs):
        out_arcs[u].append(idx)
        in_arcs[v].append(idx)

    # 流守恒约束
    for v_node in range(num_nodes):
        out_count = len(out_arcs[v_node])
        in_count = len(in_arcs[v_node])
        if v_node == source:
            # 源点: 出流 = 1
            if out_count:
                model.addConstr(
                    coptpy.LinExpr([(f[i], 1.0) for i in out_arcs[v_node]]),
                    coptpy.COPT.EQUAL, 1.0, name=f"flow_source",
                )
        elif v_node == sink:
            # 汇点: 入流 = 1
            if in_count:
                model.addConstr(
                    coptpy.LinExpr([(f[i], 1.0) for i in in_arcs[v_node]]),
                    coptpy.COPT.EQUAL, 1.0, name=f"flow_sink",
                )
        else:
            # 中间节点: 出流 = 入流
            out_expr = coptpy.LinExpr([(f[i], 1.0) for i in out_arcs[v_node]]) if out_arcs[v_node] else coptpy.LinExpr()
            in_expr = coptpy.LinExpr([(f[i], 1.0) for i in in_arcs[v_node]]) if in_arcs[v_node] else coptpy.LinExpr()
            model.addConstr(out_expr - in_expr, coptpy.COPT.EQUAL, 0.0, name=f"flow_{v_node}")

    model.setObjective(
        coptpy.LinExpr([(f[idx], arcs[idx][2]) for idx in range(len(arcs))]),
        sense=coptpy.COPT.MAXIMIZE,
    )

    model.solve()
    status = model.status
    is_optimal = (status == coptpy.COPT.OPTIMAL)
    obj = float(model.ObjVal) if is_optimal else -1
    del model
    del env
    return is_optimal, obj, status


# === Formulation 3: 完整可行模式选择（集合划分） ===
# 枚举所有可行模式（物品子集，总重 <= C）
# max sum(C_omega * lambda_omega), C_omega = sum(p_i * a_i)
# s.t. sum(lambda_omega) = 1 (恰好选一个模式)
# lambda_omega 二元

def solve_pattern_selection(weights, profits, capacity, coptpy, time_limit):
    n = len(weights)
    # 枚举所有可行模式（子集和 <= capacity）
    patterns = []
    max_patterns = 50000

    def dfs(start, current_pattern, current_weight, current_profit):
        if len(patterns) >= max_patterns:
            return
        if current_pattern:
            patterns.append((list(current_pattern), current_profit))
        for i in range(start, n):
            if current_weight + weights[i] <= capacity:
                current_pattern.append(i)
                dfs(i + 1, current_pattern, current_weight + weights[i], current_profit + profits[i])
                current_pattern.pop()

    dfs(0, [], 0, 0)

    if not patterns:
        return False, -1, -1

    env = coptpy.Envr()
    model = env.createModel("kp_pattern_selection")
    model.param.TimeLimit = float(time_limit)
    model.param.Threads = 4

    # lambda_omega: 是否选模式 omega
    lam = {}
    for idx in range(len(patterns)):
        lam[idx] = model.addVar(
            vtype=coptpy.COPT.BINARY,
            obj=float(patterns[idx][1]),
            name=f"lambda_{idx}",
        )

    # 恰好选一个模式
    model.addConstr(
        coptpy.LinExpr([(lam[idx], 1.0) for idx in range(len(patterns))]),
        coptpy.COPT.EQUAL, 1.0, name="select_one",
    )

    model.setObjective(
        coptpy.LinExpr([(lam[idx], float(patterns[idx][1])) for idx in range(len(patterns))]),
        sense=coptpy.COPT.MAXIMIZE,
    )

    model.solve()
    status = model.status
    is_optimal = (status == coptpy.COPT.OPTIMAL)
    obj = float(model.ObjVal) if is_optimal else -1
    del model
    del env
    return is_optimal, obj, status


# === Formulation 4: 精确罚项 QUBO ===
# 原二元量 x_i，整数松弛量 s = E_C(b_s) 编码容量残差
# M = 1 + sum(|p_i|)
# min F + M * (R_cap)^2, F = -sum(p_i * x_i), R_cap = sum(w_i * x_i) + s - C
# s 用二进制编码：s = sum(2^h * b_h) + (C - 2^m + 1) * b_m

def solve_qubo(weights, profits, capacity, coptpy, time_limit):
    n = len(weights)
    import math as _math

    # 编码容量残差 s 的位数
    # m(C) = floor(log2(C+1))
    m_cap = _math.floor(_math.log2(capacity + 1)) if capacity > 0 else 0
    num_s_bits = m_cap + 1  # 0..m_cap

    # 罚权
    M = 1.0 + sum(abs(p) for p in profits)

    env = coptpy.Envr()
    model = env.createModel("kp_qubo")
    model.param.TimeLimit = float(time_limit)
    model.param.Threads = 4

    # 原二元量 x_i
    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype=coptpy.COPT.BINARY, name=f"x_{i}")

    # 容量残差的二进制编码位 b_h (h=0..m_cap)
    b = {}
    for h in range(num_s_bits):
        b[h] = model.addVar(vtype=coptpy.COPT.BINARY, name=f"b_{h}")

    # s = sum(2^h * b_h) + (C - 2^m + 1) * b_m
    # 即标准二进制 + 最高位的余量位
    s_coeffs = [(b[h], float(2 ** h)) for h in range(m_cap)]
    if m_cap >= 0:
        s_coeffs.append((b[m_cap], float(capacity - 2 ** m_cap + 1)))

    # R_cap = sum(w_i * x_i) + s - C
    # 构建二次项 M * (R_cap)^2 = M * (sum(w_i*x_i) + s - C)^2
    # 展开后为二次项，COPT 支持 addQConstr

    # 先构建线性部分 L = sum(w_i * x_i) + s - C
    # 然后添加 M * L^2 作为目标（最小化）
    # COPT 的 QP 目标：model.setObjective(quad_expr, sense)

    # 线性部分
    lin_terms = [(x[i], float(weights[i])) for i in range(n)] + s_coeffs
    # L = lin_terms - C

    # 二次项展开：L^2 = (sum_i a_i * v_i - C)^2
    # = sum_i sum_j a_i*a_j * v_i*v_j - 2*C*sum_i a_i*v_i + C^2
    # 其中 v_i 包括 x_i 和 b_h

    # 构建所有变量的列表
    all_vars = list(x.values()) + list(b.values())
    all_coeffs = [float(weights[i]) for i in range(n)] + [c for _, c in s_coeffs]

    # 二次项矩阵 Q[i][j] = a_i * a_j
    # 目标: min -sum(p_i * x_i) + M * (sum(a_i * v_i) - C)^2
    #       = min -sum(p_i * x_i) + M * (sum_i sum_j a_i*a_j*v_i*v_j - 2*C*sum_i a_i*v_i + C^2)

    # COPT 二次目标构建
    # 线性部分：-p_i * x_i + M * (-2*C * a_i * v_i)
    lin_obj = coptpy.LinExpr()
    for i in range(n):
        lin_obj.addTerms(-float(profits[i]), x[i])
    # M * (-2*C) * sum(a_i * v_i)
    for idx in range(len(all_vars)):
        lin_obj.addTerms(M * (-2.0 * float(capacity)) * all_coeffs[idx], all_vars[idx])
    # M * C^2 (常数，不影响优化但加入目标)
    # COPT 目标不支持常数项，可忽略

    # 二次部分：M * sum_i sum_j a_i * a_j * v_i * v_j
    quad_expr = coptpy.QuadExpr()
    quad_expr.addLinear(lin_obj)
    for i in range(len(all_vars)):
        for j in range(i, len(all_vars)):
            coeff = M * all_coeffs[i] * all_coeffs[j]
            if i == j:
                quad_expr.addTerms(coeff, all_vars[i], all_vars[i])
            else:
                quad_expr.addTerms(coeff, all_vars[i], all_vars[j])
                quad_expr.addTerms(coeff, all_vars[j], all_vars[i])

    model.setObjective(quad_expr, sense=coptpy.COPT.MINIMIZE)

    model.solve()
    status = model.status
    is_optimal = (status == coptpy.COPT.OPTIMAL)
    # 从 x 恢复目标值
    if is_optimal:
        obj = sum(profits[i] for i in range(n) if x[i].X > 0.5)
    else:
        obj = -1
    del model
    del env
    return is_optimal, float(obj), status


# === 主函数 ===

def run_all_formulations(instances, output_dir, time_limit):
    try:
        import coptpy
    except ImportError:
        raise RuntimeError("COPT Python API (coptpy) is unavailable. Run: pip install coptpy")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formulations = ["compact", "arc_flow", "pattern_selection", "qubo"]

    perf_rows = []
    good_rows = []
    obj_rows = []

    total = len(instances)
    for idx, inst in enumerate(instances):
        params = inst.get("parameters", inst.get("instance", {}))
        weights = params["weights"]
        profits = params.get("profit", params.get("profits", []))
        capacity = params["capacity"]
        inst_id = inst.get("instance_id", f"inst_{idx}")
        n = len(weights)

        perf_row = {"instance_id": inst_id}
        good_row = {"instance_id": inst_id}
        obj_row = {"instance_id": inst_id}

        for form_name in formulations:
            try:
                start_time = time.time()
                if form_name == "compact":
                    is_opt, obj, status = solve_compact(weights, profits, capacity, coptpy, time_limit)
                elif form_name == "arc_flow":
                    # 弧流对大容量节点数爆炸，跳过 capacity > 5000
                    if capacity > 5000 or n > 50:
                        perf_row[form_name] = ""
                        good_row[form_name] = ""
                        obj_row[form_name] = ""
                        continue
                    is_opt, obj, status = solve_arc_flow(weights, profits, capacity, coptpy, time_limit)
                elif form_name == "pattern_selection":
                    # 模式枚举对大实例爆炸，跳过 n > 25
                    if n > 25:
                        perf_row[form_name] = ""
                        good_row[form_name] = ""
                        obj_row[form_name] = ""
                        continue
                    is_opt, obj, status = solve_pattern_selection(weights, profits, capacity, coptpy, time_limit)
                elif form_name == "qubo":
                    # QUBO 对大容量位数爆炸，跳过 capacity > 1000 或 n > 30
                    if capacity > 1000 or n > 30:
                        perf_row[form_name] = ""
                        good_row[form_name] = ""
                        obj_row[form_name] = ""
                        continue
                    is_opt, obj, status = solve_qubo(weights, profits, capacity, coptpy, time_limit)
                elapsed = time.time() - start_time

                perf_row[form_name] = f"{elapsed:.6f}"
                good_row[form_name] = "1" if is_opt else "0"
                obj_row[form_name] = str(obj) if obj > 0 else ""
            except Exception as e:
                perf_row[form_name] = ""
                good_row[form_name] = ""
                obj_row[form_name] = ""
                print(f"  {inst_id} / {form_name}: ERROR {e}")

        perf_rows.append(perf_row)
        good_rows.append(good_row)
        obj_rows.append(obj_row)

        if (idx + 1) % 50 == 0 or idx == 0 or idx == total - 1:
            print(f"  [{idx+1}/{total}] {inst_id} (n={n}, cap={capacity}): " +
                  " ".join(f"{f}={perf_row.get(f, '')}" for f in formulations))

    # 写CSV
    perf_path = output_dir / "performance.csv"
    with perf_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id"] + formulations)
        writer.writeheader()
        writer.writerows(perf_rows)

    good_path = output_dir / "good.csv"
    with good_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id"] + formulations)
        writer.writeheader()
        writer.writerows(good_rows)

    obj_path = output_dir / "objectives.csv"
    with obj_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id"] + formulations)
        writer.writeheader()
        writer.writerows(obj_rows)

    print(f"\n保存 {len(perf_rows)} 个实例的求解结果到 {output_dir}")
    print(f"  performance.csv: 求解时间(秒)")
    print(f"  good.csv: 最优性标签(0/1)")
    print(f"  objectives.csv: 目标值(最大收益)")

    for form_name in formulations:
        opt_count = sum(1 for row in good_rows if row.get(form_name) == "1")
        has_data = sum(1 for row in good_rows if row.get(form_name) != "")
        print(f"  {form_name}: {opt_count}/{has_data} 最优 ({100*opt_count/max(has_data,1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KP 4种formulation COPT求解")
    parser.add_argument("--instances", type=str, required=True, help="KP原始实例JSONL")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--time-limit", type=float, default=60.0, help="每个formulation求解时限(秒)")
    parser.add_argument("--max-instances", type=int, default=None, help="最多处理多少个实例(调试用)")
    args = parser.parse_args()

    instances = load_instances(args.instances)
    if args.max_instances:
        instances = instances[:args.max_instances]
    print(f"加载 {len(instances)} 个KP实例, 时限={args.time_limit}秒")
    run_all_formulations(instances, args.output_dir, args.time_limit)
