"""Patched registry.py for KP support.

This is a patch file that adds KP to the allowed problems and registry path.
On the server, copy this over the tsp_augmentation/registry.py or apply the diffs.
"""
import sys
from pathlib import Path

# Patch 1: registry.py — add "KP" to allowed problems
REGISTRY_PATCH = '''
--- registry.py original ---
    if raw["problem"] not in ("TSP", "BPP"):
        raise ValueError(f"unsupported registry problem: {raw['problem']}")
--- registry.py patched ---
    if raw["problem"] not in ("TSP", "BPP", "KP"):
        raise ValueError(f"unsupported registry problem: {raw['problem']}")
'''

# Patch 2: registry.py — add KP to default registry path
DEFAULT_PATH_PATCH = '''
--- default_tsp_registry_path original ---
    names = {
        "ATSP": "atsp_v1_copt.json",
        "STSP": "stsp_v1_copt.json",
        "BPP": "bpp_v1_copt.json",
    }
--- default_tsp_registry_path patched ---
    names = {
        "ATSP": "atsp_v1_copt.json",
        "STSP": "stsp_v1_copt.json",
        "BPP": "bpp_v1_copt.json",
        "KP": "kp_v1_copt.json",
    }
'''

# Patch 3: run_expert_generation.py — add KP to choices and adapter
RUN_EXPERT_PATCH = '''
--- run_expert_generation.py original ---
    parser.add_argument("--problem-variant", choices=("ATSP", "STSP", "BPP"), required=True)
...
    if args.problem_variant == "ATSP":
        adapter = ATSPAdapter()
    elif args.problem_variant == "STSP":
        adapter = STSPAdapter()
    else:
        from tsp_augmentation.bpp_adapter import BPPAdapter
        adapter = BPPAdapter()
--- run_expert_generation.py patched ---
    parser.add_argument("--problem-variant", choices=("ATSP", "STSP", "BPP", "KP"), required=True)
...
    if args.problem_variant == "ATSP":
        adapter = ATSPAdapter()
    elif args.problem_variant == "STSP":
        adapter = STSPAdapter()
    elif args.problem_variant == "KP":
        from kp_adapter import KPAdapter
        adapter = KPAdapter()
    else:
        from tsp_augmentation.bpp_adapter import BPPAdapter
        adapter = BPPAdapter()
'''

# Patch 4: prompting.py — add lambda_ to canonical family prefixes
PROMPTING_PATCH = '''
--- prompting.py original ---
Use the canonical variable family names from required_symbols in model.addVar(..., name=...) values. Every generated variable name must begin with its canonical family prefix, such as x_, u_, g_, f_, r_, d_, q_, y_, or z_.
--- prompting.py patched ---
Use the canonical variable family names from required_symbols in model.addVar(..., name=...) values. Every generated variable name must begin with its canonical family prefix, such as x_, u_, g_, f_, r_, d_, q_, y_, z_, or lambda_.
'''

if __name__ == "__main__":
    print("KP framework patches for TSP augmentation codebase.")
    print("Apply these diffs to tsp_augmentation/registry.py, run_expert_generation.py, prompting.py")
    print(REGISTRY_PATCH)
    print(DEFAULT_PATH_PATCH)
    print(RUN_EXPERT_PATCH)
    print(PROMPTING_PATCH)
