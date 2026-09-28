"""KP（背包问题）数据集校验、哈希、问题构造与参考解求解模块。

对标 bpp_adapter.py，但实例格式从"物品列表+箱容量"改为"重量+收益+容量"。
参考解用 COPT 标准 0/1 背包模型求解。

KP 的 instance 格式（与 E:\\D\\kp.py 的 SPEC.fields 对齐）：
    {
        "n_items": 5,
        "weights": [4651, 1335, 1428, 4755, 2930],
        "profit":  [581, 832, 245, 858, 382],
        "capacity": 10569
    }
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


NEUTRAL_PROMPT_VERSION = "kp-neutral-item-list-v1"
FORMULATION_LEAKAGE_TERMS = (
    "compact",
    "arc-flow",
    "arc flow",
    "dp-flow",
    "dp flow",
    "set cover",
    "set-cover",
    "set partition",
    "qubo",
    "dynamic programming",
    "branch and bound",
    "fptas",
    "greedy",
    "0-1 knapsack",
    "0/1 knapsack",
    "knapsack formulation",
)


@dataclass(frozen=True)
class KPReferenceSolution:
    objective: float
    selected_items: tuple[int, ...]  # 选中的物品索引（升序）
    backend: str


class KPAdapter:
    """KP 数据适配器。负责统一校验、实例归一化、提示生成和基准求解。"""
    problem = "KP"
    problem_variant = "domain"

    def __init__(self):
        self.prompt_version = NEUTRAL_PROMPT_VERSION

    def load_instances(self, path: str | Path) -> list[dict[str, Any]]:
        """从 JSONL 文件读取 KP 实例，逐条校验唯一性与合法性。

        支持两种 JSONL schema：
        1. extractor 格式（instance_id/origin/variant/parameters/lineage）
        2. adapter 格式（problem/canonical_hash/proportional_hash/source_kind/
           generation_depth/instance/lineage）
        """
        path = Path(path)
        records = []
        exact_hashes = set()
        proportional_hashes = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                record = self._normalize_record(raw, line_number)
                if record["canonical_hash"] in exact_hashes:
                    raise ValueError(f"duplicate canonical_hash at line {line_number}")
                if record["proportional_hash"] in proportional_hashes:
                    raise ValueError(f"duplicate proportional_hash at line {line_number}")
                exact_hashes.add(record["canonical_hash"])
                proportional_hashes.add(record["proportional_hash"])
                records.append(record)
        if not records:
            raise ValueError("KP instance file is empty")
        return records

    def _normalize_record(self, raw: dict[str, Any], line_number: int) -> dict[str, Any]:
        """统一校验记录，兼容 extractor 和 adapter 两种 schema。"""
        # 判断 schema 类型
        if "instance" in raw and "canonical_hash" in raw:
            # adapter 格式
            return self._normalize_adapter_record(raw, line_number)
        else:
            # extractor 格式 — 转换为 adapter 格式
            return self._normalize_extractor_record(raw, line_number)

    def _normalize_extractor_record(self, raw: dict[str, Any], line_number: int) -> dict[str, Any]:
        """从 extractor 格式（parameters + lineage）转换为 adapter 格式。"""
        params = raw.get("parameters", {})
        instance = self._normalize_instance_from_params(params, line_number)

        exact_hash = kp_instance_hash(instance)
        proportional_hash = kp_instance_hash(instance, proportional=True)

        lineage = raw.get("lineage", {})
        # 从 lineage 提取哈希，如果没有则重新计算
        if "canonical_hash" in lineage:
            exact_hash_stored = str(lineage["canonical_hash"])
            if exact_hash_stored != exact_hash:
                raise ValueError(f"line {line_number} has an invalid canonical_hash")
        if "proportional_hash" in lineage:
            prop_hash_stored = str(lineage["proportional_hash"])
            if prop_hash_stored != proportional_hash:
                raise ValueError(f"line {line_number} has an invalid proportional_hash")

        instance_id = str(raw.get("instance_id", f"sage_original_KP_{exact_hash[:10]}"))
        origin = str(raw.get("origin", "sage_original"))
        # 映射 origin -> source_kind
        if origin in ("sage_original", "sage"):
            source_kind = "sage_original"
        elif origin == "synthetic":
            source_kind = "sage_derived"
        else:
            source_kind = origin

        root_parent_hash = exact_hash

        return {
            "instance_id": instance_id,
            "canonical_hash": exact_hash,
            "proportional_hash": proportional_hash,
            "source_kind": source_kind,
            "generation_depth": 0 if source_kind == "sage_original" else 1,
            "root_parent_hash": root_parent_hash,
            "split_group_id": self.split_group_id(root_parent_hash),
            "instance": instance,
            "lineage": {
                "base_id": lineage.get("base_id", ""),
                "source": lineage.get("source", ""),
                "root_parent_hash": root_parent_hash,
            },
        }

    def _normalize_adapter_record(self, raw: dict[str, Any], line_number: int) -> dict[str, Any]:
        """校验 adapter 格式记录。"""
        if raw.get("problem", "KP") != "KP":
            raise ValueError(f"line {line_number} is not a KP record")
        for name in ("canonical_hash", "proportional_hash", "source_kind", "instance"):
            if name not in raw:
                raise ValueError(f"line {line_number} is missing {name}")

        generation_depth = int(raw.get("generation_depth", 0))
        if generation_depth not in (0, 1):
            raise ValueError(f"line {line_number} has generation_depth={generation_depth}")

        source_kind = str(raw["source_kind"])
        if generation_depth == 0 and source_kind != "sage_original":
            raise ValueError(f"line {line_number} has inconsistent original source_kind")
        if generation_depth == 1 and source_kind != "sage_derived":
            raise ValueError(f"line {line_number} has invalid derived source_kind")

        instance = self._normalize_instance(raw["instance"], line_number)
        exact_hash = kp_instance_hash(instance)
        proportional_hash = kp_instance_hash(instance, proportional=True)

        if str(raw["canonical_hash"]) != exact_hash:
            raise ValueError(f"line {line_number} has an invalid canonical_hash")
        if str(raw["proportional_hash"]) != proportional_hash:
            raise ValueError(f"line {line_number} has an invalid proportional_hash")

        root_parent_hash = (
            str(raw["canonical_hash"])
            if generation_depth == 0
            else str(raw.get("parent_canonical_hash", ""))
        )
        if not root_parent_hash:
            raise ValueError(f"line {line_number} has no root parent hash")

        instance_id = str(raw.get("instance_id", f"sage_original_KP_{exact_hash[:10]}"))
        lineage = {k: v for k, v in raw.items() if k in (
            "base_id", "parent_base_id", "parent_canonical_hash",
            "operator", "operator_parameters", "round", "seed",
        )}
        lineage["root_parent_hash"] = root_parent_hash

        return {
            "instance_id": instance_id,
            "canonical_hash": str(raw["canonical_hash"]),
            "proportional_hash": str(raw["proportional_hash"]),
            "source_kind": source_kind,
            "generation_depth": generation_depth,
            "root_parent_hash": root_parent_hash,
            "split_group_id": self.split_group_id(root_parent_hash),
            "instance": instance,
            "lineage": lineage,
        }

    def _normalize_instance(self, raw: dict[str, Any], line_number: int) -> dict[str, Any]:
        """校验 adapter 格式的 instance（含 n_items/weights/profit/capacity）。"""
        n_items = int(raw.get("n_items", 0))
        if not 2 <= n_items <= 200:
            raise ValueError(f"line {line_number} has unsupported KP size n_items={n_items}")
        weights = raw.get("weights", [])
        profit = raw.get("profit", raw.get("profits", []))
        if not isinstance(weights, list) or len(weights) != n_items:
            raise ValueError(f"line {line_number} has weights count mismatch")
        if not isinstance(profit, list) or len(profit) != n_items:
            raise ValueError(f"line {line_number} has profit count mismatch")
        capacity = int(raw.get("capacity", 0))
        if capacity < 1:
            raise ValueError(f"line {line_number} has invalid capacity={capacity}")
        normalized_weights = []
        normalized_profit = []
        for w, p in zip(weights, profit):
            numeric_w = int(w)
            numeric_p = int(p)
            if numeric_w <= 0:
                raise ValueError(f"line {line_number} has invalid weight {numeric_w}")
            if numeric_p < 0:
                raise ValueError(f"line {line_number} has invalid profit {numeric_p}")
            normalized_weights.append(numeric_w)
            normalized_profit.append(numeric_p)
        return {
            "n_items": n_items,
            "weights": normalized_weights,
            "profit": normalized_profit,
            "capacity": capacity,
        }

    def _normalize_instance_from_params(self, params: dict[str, Any], line_number: int) -> dict[str, Any]:
        """从 extractor 的 parameters 格式（weights/profit/capacity）构建 instance。"""
        weights = params.get("weights", [])
        profit = params.get("profit", params.get("profits", []))
        capacity = int(params.get("capacity", 0))
        n_items = len(weights)

        if not isinstance(weights, list) or n_items < 2:
            raise ValueError(f"line {line_number} has invalid weights")
        if not isinstance(profit, list) or len(profit) != n_items:
            raise ValueError(f"line {line_number} has profit count mismatch")
        if capacity < 1:
            raise ValueError(f"line {line_number} has invalid capacity={capacity}")

        normalized_weights = []
        normalized_profit = []
        for w, p in zip(weights, profit):
            numeric_w = int(w)
            numeric_p = int(p)
            if numeric_w <= 0:
                raise ValueError(f"line {line_number} has invalid weight {numeric_w}")
            if numeric_p < 0:
                raise ValueError(f"line {line_number} has invalid profit {numeric_p}")
            normalized_weights.append(numeric_w)
            normalized_profit.append(numeric_p)

        return {
            "n_items": n_items,
            "weights": normalized_weights,
            "profit": normalized_profit,
            "capacity": capacity,
        }

    @staticmethod
    def split_group_id(root_parent_hash: str) -> str:
        digest = hashlib.sha256(root_parent_hash.encode("utf-8")).hexdigest()[:20]
        return f"kp-root:{digest}"

    def render_question(self, instance: dict[str, Any]) -> str:
        """生成中性题面，避免泄露公式名词。"""
        n = int(instance["n_items"])
        weights = instance["weights"]
        profits = instance["profit"]
        capacity = int(instance["capacity"])
        item_lines = "\n".join(
            f"- Item {i}: Value = {profits[i]}, Weight = {weights[i]}"
            for i in range(n)
        )
        question = "\n".join([
            "# Question",
            "",
            f"Consider a selection problem with {n} items and a capacity of {capacity}.",
            "Each item has a value and a weight.",
            "The total weight of selected items must not exceed the capacity.",
            "Maximize the total value of selected items.",
            "Each item can either be fully selected or not selected at all.",
            "",
            f"Number of items: {n}",
            f"Capacity: {capacity}",
            "Item details:",
            item_lines,
            "",
            "Return a mathematical optimization model and executable Python solver code.",
        ])
        lowered = question.lower()
        leaked = [term for term in FORMULATION_LEAKAGE_TERMS if term in lowered]
        if leaked:
            raise AssertionError(f"neutral KP question leaks formulation terms: {leaked}")
        return question

    @staticmethod
    def encode_instance(instance: dict[str, Any]) -> dict[str, Any]:
        return {
            "n_items": int(instance["n_items"]),
            "weights": list(instance["weights"]),
            "profit": list(instance["profit"]),
            "capacity": int(instance["capacity"]),
        }

    @staticmethod
    def count_source_kinds(records: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            source_kind = record["source_kind"]
            counts[source_kind] = counts.get(source_kind, 0) + 1
        return dict(sorted(counts.items()))

    def reference_solution(
        self,
        instance: dict[str, Any],
        backend: str = "copt",
        time_limit: float = 120.0,
    ) -> KPReferenceSolution:
        """用 COPT 建标准 0/1 背包模型求解 KP 参考解。

        标准 0/1 背包模型：
          x_i = 物品 i 是否选中（二进制）
          目标: max sum(p_i * x_i)
          约束: sum(w_i * x_i) <= C
        """
        if backend != "copt":
            raise ValueError(f"unsupported reference backend: {backend}")

        try:
            import coptpy
        except ImportError:
            raise RuntimeError("COPT Python API (coptpy) is unavailable. Run: pip install coptpy")

        n = int(instance["n_items"])
        weights = instance["weights"]
        profits = instance["profit"]
        capacity = int(instance["capacity"])

        env = coptpy.Envr()
        model = env.createModel("kp_reference")
        model.param.TimeLimit = float(time_limit)
        model.param.Threads = 4

        # x_i: 物品 i 是否选中（二进制）
        x = {}
        for i in range(n):
            x[i] = model.addVar(
                vtype=coptpy.COPT.BINARY,
                obj=float(profits[i]),  # 目标: max sum(p_i * x_i)
                name=f"x_{i}",
            )

        # 容量约束: sum(w_i * x_i) <= C
        expr = coptpy.LinExpr([(x[i], float(weights[i])) for i in range(n)])
        model.addConstr(expr, coptpy.COPT.LESS_EQUAL, float(capacity), name="capacity")

        model.setObjective(
            coptpy.LinExpr([(x[i], float(profits[i])) for i in range(n)]),
            sense=coptpy.COPT.MAXIMIZE,
        )

        model.solve()
        status = model.status

        if status != coptpy.COPT.OPTIMAL:
            del model
            del env
            raise RuntimeError(f"COPT KP reference solve did not reach optimality: status={status}")

        objective = float(model.ObjVal)

        # 提取选中物品
        selected = tuple(i for i in range(n) if x[i].X > 0.5)

        del model
        del env

        return KPReferenceSolution(objective, selected, "copt")


def kp_instance_hash(instance: dict[str, Any], proportional: bool = False) -> str:
    """对 KP 实例计算稳定的哈希值。

    哈希 payload 包含 weights + profit + [capacity]（三向量），
    确保仅 profits 不同的实例有不同的哈希。
    """
    values = list(instance["weights"]) + list(instance["profit"]) + [int(instance["capacity"])]
    if proportional:
        nonzero = [abs(float(v)) for v in values if v != 0]
        denominator = min(nonzero) if nonzero else 1.0
        values = [format(float(v) / denominator, ".15g") for v in values]
    else:
        values = [format(float(v), ".15g") for v in values]
    payload = {
        "problem": "KP",
        "proportional": bool(proportional),
        "n_items": int(instance["n_items"]),
        "weights_profits_capacity": values,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
