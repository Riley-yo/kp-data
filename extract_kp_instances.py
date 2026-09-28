"""从 SAGE-SFT-100K_labeled.csv 提取 KP（背包问题）实例，写成 JSONL 格式。

用法（在 WSL2 或服务器上）：
    python extract_kp_instances.py <SAGE_CSV路径> <输出JSONL路径>

输出格式（每行一个 JSON）：
    {
        "instance_id": "sage_original_KP_0001_<hash>",
        "origin": "sage_original",
        "variant": "domain",
        "parameters": {
            "weights": [4651, 1335, 1428, 4755, 2930],
            "profit": [581, 832, 245, 858, 382],
            "capacity": 10569
        },
        "lineage": {
            "base_id": "162_0",
            "source": "OptMATH",
            "canonical_hash": "sha256...",
            "proportional_hash": "sha256..."
        }
    }

字段命名对齐 E:\\D\\kp.py 的 SPEC.fields：
    weights (aliases: weights/weight/costs/cost)
    profit  (aliases: values/value/profits/profit/benefits/returns)
    capacity (aliases: capacity/budget/budget_limit/total_budget)
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


def _strip_thousands(value: str) -> str:
    """去掉千分位逗号：'10,569' -> '10569'。"""
    return value.replace(",", "")


def _preprocess_question(question: str) -> str:
    """预处理题目文本。

    1. 去掉 ** markdown 粗体标记
    2. 去掉 LaTeX 转义反斜杠（如 \\$ -> $）
    3. 截断 # Note: 块（无参数信息，只有 Gurobi 编码要求）
    4. 保留原始换行结构（物品列表靠换行分隔）
    """
    text = question.replace("**", "").replace("\\$", "$")
    note_idx = text.find("# Note:")
    if note_idx != -1:
        text = text[:note_idx]
    return text


def _extract_capacity(text: str) -> int | None:
    """从题目文本提取背包容量。

    支持：
      - maximum weight capacity of N (units|kilograms|kg)
      - weight capacity of N
      - capacity of N
      - must not exceed N (units|kilograms) — IndOR 变体
      - budget of N (million)? (yuan|units)?
    """
    patterns = [
        r"(?:maximum\s+)?weight\s+capacity\s+of\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        r"capacity\s+of\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        r"weight\s+capacity\s+of\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        r"capacity\s+limit\s+of\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        r"maximum\s+capacity\s+of\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        # "must not exceed N units" — IndOR 变体
        r"must\s+not\s+exceed\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        # "does not exceed N units"
        r"does\s+not\s+exceed\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        # "budget ... is N units" / "total budget ... is N"
        r"budget[^\n]*?is\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        # "no more than N units"
        r"no\s+more\s+than\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        # "total weight capacity ... is N units"
        r"total\s+weight\s+capacity[^\n]*?is\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        # "total workload capacity ... is N units"
        r"total\s+(?:workload|weight)\s+capacity[^\n]*?(?:is|limited to)\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
        # "weight limit ... is N" / "weight limit for ... is N"
        r"weight\s+limit[^\n]*?is\s*(\d[\d,]*)\s*(?:units?|kilograms?|kg)?",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(_strip_thousands(m.group(1)))
    return None


def _extract_budget(text: str) -> int | None:
    """从 IndOR 题目文本提取预算（budget）。

    支持 'budget of N million yuan' / 'budget of N yuan' 格式。
    注意：不做 million ×1000000 单位换算，因为表格列头通常也以 million yuan
    为单位标注，weight 和 capacity 需保持同一数值尺度（KP 特征多为比值，尺度无关）。
    """
    m = re.search(
        r"budget\s+of\s*(\d[\d,]*)\s*(?:million\s+)?(?:yuan|units?)?",
        text, re.IGNORECASE,
    )
    if m:
        return int(_strip_thousands(m.group(1)))
    return None


def _parse_optmath_items(text: str) -> tuple[list[int], list[int]] | None:
    """解析 OptMATH 格式的物品列表。

    支持 7 种 pattern（按优先级），返回 (weights, profits)。
    profit = value（收益），weight = 重量。
    物品关键词可为 Item / Package / Shipment / Equipment / Supply / Cargo。
    """
    item_kw = r"(?:Item|Package|Shipment|Equipment|Supply|Cargo|Product|Component|Crop|Resource|Project|Candidate|Material|Food)"
    # item_separator matches space, underscore, or nothing between kw and number
    item_sep = r"[\s_]*"

    # Pattern 1: <kw> N: Value = X, Weight = Y  (含反向 Weight 在前)
    # 匹配 "Item 0: Value = 758, Weight = 124 kg" 或 "Package 0: Value = 253 units, Weight = 29 kg"
    # 含可选的数字编号前缀 "1. Package 0: ..."
    items_vw = re.findall(
        r"(?:\d+\.\s*)?" + item_kw + r"\s*(\d+)\s*:?\s*Value\s*=\s*(\d[\d,]*).*?Weight\s*=\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_vw:
        items_vw.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_vw]
        weights = [int(_strip_thousands(m[2])) for m in items_vw]
        return weights, profits

    items_wv = re.findall(
        r"(?:\d+\.\s*)?" + item_kw + r"\s*(\d+)\s*:?\s*Weight\s*=\s*(\d[\d,]*).*?Value\s*=\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_wv:
        items_wv.sort(key=lambda x: int(x[0]))
        weights = [int(_strip_thousands(m[1])) for m in items_wv]
        profits = [int(_strip_thousands(m[2])) for m in items_wv]
        return weights, profits

    # Pattern 2: <kw> N has a value/utility of X and weighs/weight of Y
    # "Shipment 0 has a value of X and weighs Y"
    # "Component 0 has a value of X and a weight of Y"
    # "Resource 0 has an educational value of 963 units and costs 2436 units."
    items_has = re.findall(
        item_kw + r"\s*(\d+|[A-Z])\s+has an?\s+(?:educational\s+)?(?:value|utility)\s+of\s*(\d[\d,]*).*?(?:and\s+)?(?:weighs|a weight of|weight of|costs)\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_has:
        def sort_key(m):
            try:
                return int(m[0])
            except ValueError:
                return ord(m[0].upper()) if m[0].isalpha() else 0
        items_has.sort(key=sort_key)
        profits = [int(_strip_thousands(m[1])) for m in items_has]
        weights = [int(_strip_thousands(m[2])) for m in items_has]
        return weights, profits

    # Pattern 3: item_N (value[:= ]X, weight[:= ]Y) 紧凑行内格式（含反引号包裹）
    # 支持 "value: X"、"value=X"、"value X" 三种分隔
    items_inline = re.findall(
        r"`?item_(\d+)`?\s*\(\s*value\s*[:= ]\s*(\d[\d,]*).*?weight\s*[:= ]\s*(\d[\d,]*)\s*\)",
        text, re.IGNORECASE,
    )
    if items_inline:
        items_inline.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_inline]
        weights = [int(_strip_thousands(m[2])) for m in items_inline]
        return weights, profits

    # Pattern 3a: `item_N`: Value = X, Weight = Y（反引号格式）
    items_backtick = re.findall(
        r"`?item_(\d+)`?\s*:?\s*Value\s*=\s*(\d[\d,]*).*?Weight\s*=\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_backtick:
        items_backtick.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_backtick]
        weights = [int(_strip_thousands(m[2])) for m in items_backtick]
        return weights, profits

    # Pattern 3b: item_N ... has a value of X ... weighs Y （散文式行内）
    # "The first shipment, item_0, has a value of 147 units and weighs 2,427 units."
    items_inline_prose = re.findall(
        r"item_(\d+).*?has a value of\s*(\d[\d,]*).*?weighs\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_inline_prose:
        items_inline_prose.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_inline_prose]
        weights = [int(_strip_thousands(m[2])) for m in items_inline_prose]
        return weights, profits

    # Pattern 3c: item_N is valued at X and weighs Y （行内散文）
    # "item_0 is valued at 394 units and weighs 788 units"
    items_inline_valued = re.findall(
        r"item_(\d+)\s+is valued at\s*(\d[\d,]*).*?weighs\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_inline_valued:
        items_inline_valued.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_inline_valued]
        weights = [int(_strip_thousands(m[2])) for m in items_inline_valued]
        return weights, profits

    # Pattern 4: <kw> N: Valued at X, weighing/weighs Y
    # "Item 0: Valued at 596 units and weighs 3583 units."
    # "item_0: Valued at 532 units and weighing 957 units."
    items_valued = re.findall(
        item_kw + item_sep + r"(\d+|[A-Z])\s*:?\s*Valued at\s*(\d[\d,]*).*?(?:weighing|weighs)\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_valued:
        def sort_key(m):
            try:
                return int(m[0])
            except ValueError:
                return ord(m[0].upper()) if m[0].isalpha() else 0
        items_valued.sort(key=sort_key)
        profits = [int(_strip_thousands(m[1])) for m in items_valued]
        weights = [int(_strip_thousands(m[2])) for m in items_valued]
        return weights, profits

    # Pattern 4a: <kw> N: Valued at X, with a cost of Y
    # "Item 0: Valued at 927 units, with a cost of 3517 units."
    items_valued_cost = re.findall(
        item_kw + r"\s*(\d+)\s*:?\s*Valued at\s*(\d[\d,]*).*?with a cost of\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_valued_cost:
        items_valued_cost.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_valued_cost]
        weights = [int(_strip_thousands(m[2])) for m in items_valued_cost]
        return weights, profits

    # Pattern 4b: <kw> N: Value/utility of X, weight of Y
    # "Item 0: Value of 483, Weight of 4500"  /  "Item 0: Utility of 568, weight of 4594"
    items_valueof = re.findall(
        item_kw + r"\s*(\d+)\s*:?\s*(?:Value|Utility|Benefit|Performance)\s+of\s*(\d[\d,]*).*?(?:weight|cost)\s+of\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_valueof:
        items_valueof.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_valueof]
        weights = [int(_strip_thousands(m[2])) for m in items_valueof]
        return weights, profits

    # Pattern 4c: <kw> N has a value of X and a cost of Y
    # "cost" 作为 weight 的别名
    items_cost = re.findall(
        item_kw + r"\s*(\d+)\s+has a value of\s*(\d[\d,]*).*?and a cost of\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_cost:
        items_cost.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_cost]
        weights = [int(_strip_thousands(m[2])) for m in items_cost]
        return weights, profits

    # Pattern 4d: <kw> N has a value of X and a weight of Y
    # "Item 0 has a value of 468 and a weight of 123" — 行内/散文式
    items_weightof = re.findall(
        item_kw + r"\s*(\d+)\s+has a value of\s*(\d[\d,]*).*?and a weight of\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_weightof:
        items_weightof.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_weightof]
        weights = [int(_strip_thousands(m[2])) for m in items_weightof]
        return weights, profits

    # Pattern 4e: <kw> N with a value of X and weight of Y
    # "Item 0 with a value of 179 and weight of 3,130" — 行内/散文式
    items_with = re.findall(
        item_kw + r"\s*(\d+)\s+with a value of\s*(\d[\d,]*).*?and weight of\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_with:
        items_with.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_with]
        weights = [int(_strip_thousands(m[2])) for m in items_with]
        return weights, profits

    # Pattern 5: <kw> N: Has/Provides a value/utility of X and weighs/costs Y
    # "Package 0: This package has a value of X and weighs Y"
    # "Item 0: Provides a value of 371 units and weighs 605 units."
    items_has2 = re.findall(
        item_kw + r"\s*(\d+)\s*:?\s*(?:This\s+\w+\s+)?(?:Has|Provides|Contributes)\s+an?\s+(?:value|utility|educational value|performance)\s+of\s*(\d[\d,]*).*?(?:and\s+)?(?:weighs|costs|adds a workload of|with a weight of)\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_has2:
        items_has2.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_has2]
        weights = [int(_strip_thousands(m[2])) for m in items_has2]
        return weights, profits

    # Pattern 5a: <kw> N: Valued at X, with a weight of Y
    # "Item 0: Valued at 639 units with a weight of 3937 units."
    items_valued_weight = re.findall(
        item_kw + r"\s*(\d+)\s*:?\s*Valued at\s*(\d[\d,]*).*?with a weight of\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_valued_weight:
        items_valued_weight.sort(key=lambda x: int(x[0]))
        profits = [int(_strip_thousands(m[1])) for m in items_valued_weight]
        weights = [int(_strip_thousands(m[2])) for m in items_valued_weight]
        return weights, profits

    # Pattern 5b: 嵌套子项目格式 — <kw> N: \n - Value: X \n - Weight: Y
    # "Shipment 1:  \n  - Value: $526  \n  - Weight: 1,561 kg"
    items_nested = re.findall(
        item_kw + r"\s*(\d+|[A-Z])\s*:?\s*\n\s*[-*]\s*Value\s*:\s*\$?\s*(\d[\d,]*).*?\n\s*[-*]\s*Weight\s*:\s*\$?\s*(\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if items_nested:
        def sort_key(m):
            try:
                return int(m[0])
            except ValueError:
                return ord(m[0].upper()) if m[0].isalpha() else 0
        items_nested.sort(key=sort_key)
        profits = [int(_strip_thousands(m[1])) for m in items_nested]
        weights = [int(_strip_thousands(m[2])) for m in items_nested]
        return weights, profits

    # Pattern 6: 分块格式 — value 列表和 weight 列表分别给出
    # "Value (Revenue): ... Item 0: $336 ..."  +  "Weight: ... Item 0: 46 kg ..."
    # 先定位 Value/Revenue 段和 Weight 段
    value_section = re.search(
        r"(?:Value|Revenue|Profit)[^\n:]*:\s*\n(.*?)(?=\n\s*(?:[-*]\s*)?(?:Weight|Objective|Constraint|Decision|Your|$))",
        text, re.IGNORECASE | re.DOTALL,
    )
    weight_section = re.search(
        r"Weight[^\n:]*:\s*\n(.*?)(?=\n\s*(?:[-*]\s*)?(?:Objective|Constraint|Decision|Your|$))",
        text, re.IGNORECASE | re.DOTALL,
    )
    if value_section and weight_section:
        profit_block = re.findall(
            item_kw + r"\s*(\d+)\s*:?\s*\$?\s*(\d[\d,]*)",
            value_section.group(1), re.IGNORECASE,
        )
        weight_block = re.findall(
            item_kw + r"\s*(\d+)\s*:?\s*(\d[\d,]*)",
            weight_section.group(1), re.IGNORECASE,
        )
        if profit_block and weight_block:
            profit_map = {int(m[0]): int(_strip_thousands(m[1])) for m in profit_block}
            weight_map = {int(m[0]): int(_strip_thousands(m[1])) for m in weight_block}
            common = sorted(set(profit_map) & set(weight_map))
            if len(common) >= 2:
                profits = [profit_map[i] for i in common]
                weights = [weight_map[i] for i in common]
                return weights, profits

    # Pattern 7: "units of value" / "units of weight" 分块格式
    # "Item 0: 576 units of value" ... + "Item 0: 123 units of weight" ...
    value_lines = re.findall(
        item_kw + r"\s*(\d+)\s*:\s*(\d[\d,]*)\s*units of value",
        text, re.IGNORECASE,
    )
    weight_lines = re.findall(
        item_kw + r"\s*(\d+)\s*:\s*(\d[\d,]*)\s*units of weight",
        text, re.IGNORECASE,
    )
    if value_lines and weight_lines:
        profit_map = {int(m[0]): int(_strip_thousands(m[1])) for m in value_lines}
        weight_map = {int(m[0]): int(_strip_thousands(m[1])) for m in weight_lines}
        common = sorted(set(profit_map) & set(weight_map))
        if len(common) >= 2:
            profits = [profit_map[i] for i in common]
            weights = [weight_map[i] for i in common]
            return weights, profits

    # Pattern 8: OptMATH markdown 表格（| Item | Value | Weight |）
    # 表头列名含 value/profit → profit 列，含 weight/cost → weight 列
    md_table = _parse_optmath_table(text)
    if md_table is not None:
        return md_table

    return None


def _parse_optmath_table(text: str) -> tuple[list[int], list[int]] | None:
    """解析 OptMATH markdown 表格（表头为 Item/Value/Weight 等）。

    与 IndOR 表格不同：OptMATH 表头第一列通常是 Item 编号（1-based），
    后两列是 Value 和 Weight。无名称列。
    """
    lines = text.split("\n")
    separator_idx = None
    header_line = None
    for i, line in enumerate(lines):
        if re.match(r"\s*\|[\s\-:|]+\|\s*$", line):
            separator_idx = i
            if i > 0:
                header_line = lines[i - 1]
            break

    if separator_idx is None or header_line is None:
        return None

    header_cells = [c.strip() for c in header_line.split("|")[1:-1]]
    if len(header_cells) < 3:
        return None

    id_col = None
    weight_col = None
    profit_col = None
    for idx, cell in enumerate(header_cells):
        cell_lower = cell.lower()
        if id_col is None and any(kw in cell_lower for kw in ("item", "id", "#", "no")):
            id_col = idx
        if weight_col is None and any(kw in cell_lower for kw in ("weight", "cost")):
            weight_col = idx
        if profit_col is None and any(kw in cell_lower for kw in ("value", "profit", "benefit", "revenue")):
            profit_col = idx

    if weight_col is None or profit_col is None or weight_col == profit_col:
        return None

    weights = []
    profits = []
    for line in lines[separator_idx + 1:]:
        line = line.strip()
        if not line or not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) <= max(weight_col, profit_col):
            continue
        w_match = re.search(r"(\d[\d,]*)", cells[weight_col])
        p_match = re.search(r"(\d[\d,]*)", cells[profit_col])
        if w_match and p_match:
            weights.append(int(_strip_thousands(w_match.group(1))))
            profits.append(int(_strip_thousands(p_match.group(1))))

    if len(weights) >= 2 and len(weights) == len(profits):
        return weights, profits
    return None


def _parse_indor_table(text: str) -> tuple[list[int], list[int]] | None:
    """解析 IndOR markdown 表格格式。

    检测 |---| 分隔行，根据表头列名判断哪列是 weight（cost）、哪列是 profit（benefit）。
    """
    lines = text.split("\n")
    separator_idx = None
    header_line = None
    for i, line in enumerate(lines):
        if re.match(r"\s*\|[\s\-:|]+\|\s*$", line):
            separator_idx = i
            if i > 0:
                header_line = lines[i - 1]
            break

    if separator_idx is None or header_line is None:
        return None

    header_cells = [c.strip() for c in header_line.split("|")[1:-1]]
    if len(header_cells) < 3:
        return None

    weight_col = None
    profit_col = None
    for idx, cell in enumerate(header_cells):
        cell_lower = cell.lower()
        if weight_col is None and any(
            kw in cell_lower for kw in ("cost", "expense", "price", "weight")
        ):
            weight_col = idx
        if profit_col is None and any(
            kw in cell_lower
            for kw in ("capacity", "increase", "saving", "benefit", "value", "profit", "return")
        ):
            profit_col = idx

    if weight_col is None or profit_col is None or weight_col == profit_col:
        return None

    weights = []
    profits = []
    for line in lines[separator_idx + 1:]:
        line = line.strip()
        if not line or not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) <= max(weight_col, profit_col):
            continue
        w_match = re.search(r"(\d[\d,]*)", cells[weight_col])
        p_match = re.search(r"(\d[\d,]*)", cells[profit_col])
        if w_match and p_match:
            weights.append(int(_strip_thousands(w_match.group(1))))
            profits.append(int(_strip_thousands(p_match.group(1))))

    if len(weights) >= 2 and len(weights) == len(profits):
        return weights, profits
    return None


def parse_kp_question(question: str) -> tuple[list[int], list[int], int] | None:
    """从 KP 题目文本解析物品重量、收益和背包容量。

    支持 SAGE-SFT-100K 里的 OptMATH（5 种格式）和 IndOR（markdown 表格）变体。
    返回 (weights, profits, capacity) 或 None。
    """
    text = _preprocess_question(question)

    result = _parse_optmath_items(text)
    if result is None:
        result = _parse_indor_table(text)
    if result is None:
        return None

    weights, profits = result

    capacity = _extract_capacity(text)
    if capacity is None:
        capacity = _extract_budget(text)
    if capacity is None:
        return None

    if len(weights) != len(profits):
        return None
    if len(weights) < 2 or len(weights) > 200:
        return None
    if not all(w > 0 for w in weights):
        return None
    if not all(p >= 0 for p in profits):
        return None
    if capacity <= 0:
        return None

    return weights, profits, capacity


def kp_instance_hash(
    weights: list[int], profits: list[int], capacity: int, proportional: bool = False
) -> str:
    """计算 KP 实例的哈希（精确或比例）。

    哈希 payload 包含 weights + profits + [capacity]（三向量），
    确保仅 profits 不同的实例有不同的哈希。
    """
    values = list(weights) + list(profits) + [capacity]
    if proportional:
        nonzero = [abs(float(v)) for v in values if v != 0]
        denominator = min(nonzero) if nonzero else 1.0
        values = [format(float(v) / denominator, ".15g") for v in values]
    else:
        values = [format(float(v), ".15g") for v in values]
    payload = {
        "problem": "KP",
        "proportional": bool(proportional),
        "n_items": len(weights),
        "weights_profits_capacity": values,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_kp(csv_path: str | Path, output_path: str | Path) -> dict:
    """从 SAGE CSV 提取 KP 实例，去重，写成 JSONL。"""
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    total_kp = 0
    parsed_ok = 0
    parsed_fail = 0
    instances = []
    seen_exact = set()
    seen_proportional = set()
    source_breakdown = {"OptMATH": 0, "IndOR": 0, "other": 0}
    fail_by_source = {"OptMATH": 0, "IndOR": 0, "other": 0}

    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) <= 7 or row[7] != "KP":
                continue
            total_kp += 1
            source = row[5] if len(row) > 5 else "OptMATH"
            if source not in source_breakdown:
                source = "other"

            question = row[2]
            parsed = parse_kp_question(question)
            if parsed is None:
                parsed_fail += 1
                fail_by_source[source] += 1
                continue

            weights, profits, capacity = parsed
            exact_hash = kp_instance_hash(weights, profits, capacity)
            prop_hash = kp_instance_hash(weights, profits, capacity, proportional=True)

            if exact_hash in seen_exact or prop_hash in seen_proportional:
                continue

            seen_exact.add(exact_hash)
            seen_proportional.add(prop_hash)
            parsed_ok += 1
            source_breakdown[source] += 1

            instances.append({
                "instance_id": f"sage_original_KP_{parsed_ok:04d}_{exact_hash[:10]}",
                "origin": "sage_original",
                "variant": "domain",
                "parameters": {
                    "weights": weights,
                    "profit": profits,
                    "capacity": capacity,
                },
                "lineage": {
                    "base_id": row[0],
                    "source": source,
                    "canonical_hash": exact_hash,
                    "proportional_hash": prop_hash,
                },
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    from collections import Counter
    n_dist = Counter(len(inst["parameters"]["weights"]) for inst in instances)

    summary = {
        "total_kp_rows": total_kp,
        "parsed_ok": parsed_ok,
        "parsed_fail": parsed_fail,
        "unique_instances": len(instances),
        "source_breakdown": {k: v for k, v in source_breakdown.items() if v > 0},
        "fail_by_source": {k: v for k, v in fail_by_source.items() if v > 0},
        "n_items_distribution": dict(sorted(n_dist.items())),
        "output_path": str(output_path),
    }
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python extract_kp_instances.py <SAGE_CSV路径> <输出JSONL路径>")
        sys.exit(1)
    result = extract_kp(sys.argv[1], sys.argv[2])
    print(json.dumps(result, ensure_ascii=False, indent=2))
