import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import boxcox_normmax
from projection_plot import read_indexed


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def stable_boxcox(positive, lam, anchor):
    logs = np.log(positive)
    if abs(lam) < 1e-12:
        return logs - anchor
    # Positive affine rescaling of Box-Cox; Z scores are unchanged.
    return np.expm1(lam * (logs - anchor)) / lam


def prepare_features(frame):
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Instance IDs and feature names must be unique")
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Feature inputs must be nonempty and finite; missing values are not imputed")
    transformed, parameters, constant = {}, {}, []
    for name in frame:
        values = frame[name].to_numpy(dtype=float)
        if np.ptp(values) == 0:
            constant.append(name)
            continue
        shift = 1.0 - float(values.min())
        positive = values + shift
        lam = float(boxcox_normmax(positive, method="mle"))
        logs = np.log(positive)
        anchor = float(logs.max() if lam >= 0 else logs.min())
        converted = stable_boxcox(positive, lam, anchor)
        mean, std = float(converted.mean()), float(converted.std(ddof=1))
        if not np.isfinite(converted).all() or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Box-Cox is numerically invalid for {name}; no silent replacement")
        transformed[name] = (converted - mean) / std
        parameters[name] = {"shift": shift, "lambda": lam, "log_anchor": anchor, "mean": mean, "std": std}
    result = pd.DataFrame(transformed, index=frame.index)
    return result, {"features": parameters, "constant_features": constant,
                    "normalization": "column minimum shifted to 1; Box-Cox MLE; sample Z score (ddof=1)",
                    "performance_inputs_used": False, "shape_used_for_selection": False}


def prepare_performance(frame):
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Performance instance IDs and formulation names must be unique")
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Measured performance must be nonempty and finite")
    global_minimum = float(frame.to_numpy(dtype=float).min())
    offset = float(np.finfo(float).eps)
    transformed, parameters, constant = {}, {}, []
    for name in frame:
        values = frame[name].to_numpy(dtype=float)
        if np.ptp(values) == 0:
            constant.append(name)
            continue
        positive = (values - global_minimum) + offset
        lam = float(boxcox_normmax(positive, method="mle"))
        logs = np.log(positive)
        anchor = float(logs.max() if lam >= 0 else logs.min())
        converted = stable_boxcox(positive, lam, anchor)
        mean, std = float(converted.mean()), float(converted.std(ddof=1))
        if not np.isfinite(converted).all() or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Performance Box-Cox is numerically invalid for {name}")
        transformed[name] = (converted - mean) / std
        parameters[name] = {"subtract_minimum": global_minimum, "positive_offset": offset,
                            "lambda": lam, "log_anchor": anchor, "mean": mean, "std": std}
    if not transformed:
        raise ValueError("All measured performance responses are constant; no PILOT performance trend")
    return pd.DataFrame(transformed, index=frame.index), {
        "features": parameters, "constant_features": constant,
        "normalization": "global minimum shifted to machine epsilon; Box-Cox MLE; sample Z score (ddof=1)",
        "role": "separate measured Y response, never intrinsic X",
        "reference": "https://github.com/andremun/InstanceSpace/blob/e7fa8002d8979576ce53115f745ce6f9f3331df0/PRELIM.m"}


def transform_features(frame, fitted):
    result = {}
    for name, parameter in fitted["features"].items():
        values = frame[name].to_numpy(dtype=float)
        if "subtract_minimum" in parameter:
            values = (values - parameter["subtract_minimum"]) + parameter["positive_offset"]
        else:
            values = values + parameter["shift"]
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Instance outside frozen Box-Cox domain for {name}")
        result[name] = (stable_boxcox(values, parameter["lambda"], parameter["log_anchor"]) - parameter["mean"]) / parameter["std"]
    output = pd.DataFrame(result, index=frame.index)
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError("Frozen transformation produced nonfinite features")
    return output


def fit_projection(frame, transformed, args):
    from paper_sifted import select_features
    from paper_pilot import fit_paper_pilot
    from space_coverage import shape_diagnostics, coverage_targets
    from paper_cloister import cloister_boundary_with_evidence

    responses = read_indexed(args.performance)
    labels = read_indexed(args.good)
    for table in (responses, labels):
        table.index = table.index.astype(str)
        if not table.index.is_unique or not table.columns.is_unique or set(table.index) != set(frame.index):
            raise ValueError("Performance tables must contain exactly the feature instance IDs, without omissions or duplicates")
    responses, labels = responses.reindex(frame.index), labels.reindex(frame.index)
    if set(responses.columns) != set(labels.columns):
        raise ValueError("Continuous and binary responses must cover the same formulations")
    if not np.isfinite(responses.to_numpy(float)).all() or np.any(responses.to_numpy(float) <= 0):
        raise ValueError("Performance CSV requires strictly positive measured capped solve seconds")
    selected, sifted = select_features(transformed, labels, seed=args.seed)
    write_json(args.output_dir / "sifted.json", sifted)
    y, y_prelim = prepare_performance(responses)
    y.columns = ["performance::" + name for name in y.columns]
    fitted = fit_paper_pilot(transformed[selected], y, seed=args.seed)
    a = pd.DataFrame(fitted.projection, index=selected, columns=["z1", "z2"])
    z = pd.DataFrame(fitted.coordinates, index=frame.index, columns=["z1", "z2"])
    np.testing.assert_allclose(transformed[selected].to_numpy() @ a.to_numpy(), z.to_numpy(), atol=1e-10)
    a.to_csv(args.output_dir / "projection_matrix.csv", float_format="%.17g")
    z.to_csv(args.output_dir / "coordinates.csv", float_format="%.17g")
    transformed[selected].to_csv(args.output_dir / "selected_features.csv", float_format="%.17g")
    pd.DataFrame(fitted.reconstruction, index=["z1", "z2"], columns=selected + list(y.columns)).to_csv(
        args.output_dir / "reconstruction_matrix.csv", float_format="%.17g")
    # These transformations apply only to Y, never to intrinsic generation inputs.
    write_json(args.output_dir / "performance_prelim.json", y_prelim)
    responses.to_csv(args.output_dir / "performance.csv", float_format="%.17g")
    labels.to_csv(args.output_dir / "good.csv")
    method = {"selected_features": selected, "trials": fitted.trials, "chosen_trial": fitted.chosen_trial,
              "sifted_implementation_version": sifted["implementation_version"],
              "seed": args.seed, "coordinates": "transformed intrinsic X[selected] @ A; no postprocessing",
              "performance_scaling": y_prelim["normalization"] + "; measured capped wall seconds, separately from X",
              "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (args.performance, args.good)},
              "circle_used_in_feature_selection": False, "circle_used_in_projection": False,
              "performance_partition_generated": False}
    write_json(args.output_dir / "pilot.json", method)
    diagnostics = shape_diagnostics(z.to_numpy())
    write_json(args.output_dir / "shape_diagnostics.json", diagnostics)
    try:
        boundary, boundary_evidence = cloister_boundary_with_evidence(transformed[selected], fitted.projection)
        write_json(args.output_dir / "boundary_status.json", boundary_evidence)
        pd.DataFrame(boundary, columns=["z1", "z2"]).to_csv(args.output_dir / "cloister_boundary.csv", index=False)
        targets, distances = coverage_targets(z.to_numpy(), boundary)
        pd.DataFrame({"z1": targets[:, 0], "z2": targets[:, 1], "nearest_distance": distances}).to_csv(
            args.output_dir / "coverage_targets.csv", index=False)
    except ValueError as error:
        write_json(args.output_dir / "coverage_status.json", {"status": "boundary_unavailable", "reason": str(error)})
    from projection_plot import draw_projection
    draw_projection(args.output_dir)
    return {"projection_fitted": True, "status": "projection_complete_partitions_deferred",
            "selected_feature_count": len(selected), "shape_diagnostics": diagnostics}


def run_cli(spec, calculate_features):
    from paper_sifted import IMPLEMENTATION_VERSION
    parser = argparse.ArgumentParser(description=spec["name"] + "：实例特征与 ISA 数据准备")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", type=Path, help="原始实例特征 CSV，首列为实例 ID")
    source.add_argument("--instances", type=Path, help="含实例 ID、origin、parameters 的 JSONL")
    source.add_argument("--coordinates", type=Path, help="已有二维坐标 CSV；仅重绘，不重新拟合")
    parser.add_argument("--origins", type=Path, help="特征 CSV 对应的 instance_id、origin 来源表")
    parser.add_argument("--performance", type=Path, help="独立实测性能 CSV：截断求解秒数，首列实例 ID")
    parser.add_argument("--good", type=Path, help="同一协议下的最优性及时限 0/1 标签 CSV")
    parser.add_argument("--projection-method", choices=["paper"], default="paper", help="二维装箱论文的 PRELIM、SIFTED、PILOT 流程")
    parser.add_argument("--fitted-projection", type=Path, help="已拟合的同问题 ISA 目录；以冻结映射纳入全部特征实例，不要求新实例已有 Y")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output-dir", type=Path, required=True, help="新的输出目录")
    args = parser.parse_args()
    if bool(args.performance) != bool(args.good):
        parser.error("--performance 和 --good 必须同时提供")
    if args.fitted_projection and (args.performance or args.good or args.coordinates):
        parser.error("冻结映射只接收实例或特征输入，不同时拟合或直接重绘")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty; existing records are not overwritten")
    if args.coordinates:
        if not args.origins or args.performance or args.good:
            parser.error("重绘需要 --coordinates 和 --origins，不接收性能数据")
        from projection_plot import draw_projection
        draw_projection(args.output_dir, args.coordinates, args.origins,
                        title=spec['problem'] + ' / supplied ISA coordinates')
        return
    if args.features and not args.origins:
        parser.error("--features 必须提供 --origins，禁止推测来源")
    if args.instances and args.origins:
        parser.error("JSONL 已含来源，不同时接收 --origins")
    audit, originals = [], []
    if args.instances:
        source_path = args.instances
        records = [json.loads(line) for line in source_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        ids = [row.get("instance_id", row.get("question_id")) for row in records]
        if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("Every instance must have a nonempty, unique string ID")
        rows = []
        for instance_id, record in zip(ids, records):
            originals.append(record)
            try:
                origin = record["origin"]
                if origin not in {"sage", "sage_original", "synthetic"}:
                    raise ValueError("origin must be sage, sage_original or synthetic")
                if origin == "synthetic" and not all(k in record for k in ("generator", "seed", "validation")):
                    raise ValueError("Synthetic instance needs generator, seed and validation provenance")
                variant = record.get("variant", spec["primary_variant"])
                if variant != spec["primary_variant"]:
                    raise ValueError("Instance variant differs from the selected primary representation")
                values = calculate_features(record["parameters"], record.get("question"))
                if not values or not np.isfinite(list(values.values())).all():
                    raise ValueError("Empty or nonfinite intrinsic features")
                rows.append({"instance_id": instance_id, **values})
                audit.append({"instance_id": instance_id, "origin": origin, "status": "features_available", "reason": ""})
            except (ValueError, KeyError, TypeError, AssertionError, IndexError, ZeroDivisionError) as error:
                audit.append({"instance_id": instance_id, "origin": record.get("origin"),
                              "status": "needs_parameter_review", "reason": str(error)})
        frame = pd.DataFrame(rows).set_index("instance_id") if rows else pd.DataFrame()
    else:
        source_path = args.features or (Path(spec["default_features"]) if spec["default_features"] else None)
        if source_path is None:
            parser.error("本问题尚无完整特征表，请提供 --instances 或 --features；不会补造数据")
        frame = read_indexed(source_path)
        frame.index = frame.index.astype(str)
        unknown = set(frame.columns) - set(spec["candidate_features"])
        if unknown:
            raise ValueError(f"Undeclared feature columns: {sorted(unknown)}")
        origins_path = args.origins or (Path(spec["default_origins"]) if args.features is None and spec.get("default_origins") else None)
        if origins_path:
            origins = pd.read_csv(origins_path, dtype={"instance_id": str}).set_index("instance_id")
            if not origins.index.is_unique or set(origins.index) != set(frame.index) or not origins["origin"].isin(["sage", "sage_original", "synthetic"]).all():
                raise ValueError("Origin table must label every feature row exactly once as sage, sage_original or synthetic")
            audit = [{"instance_id": str(i), "origin": origins.loc[i, "origin"], "status": "features_available", "reason": ""} for i in frame.index]
        else:
            raise ValueError("Origin table is required; source labels are never inferred")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if originals:
        (args.output_dir / "retained_instances.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in originals) + "\n", encoding="utf-8")
    pd.DataFrame(audit).to_csv(args.output_dir / "instance_audit.csv", index=False, encoding="utf-8-sig")
    status = {"problem": spec["problem"], "variant": spec["primary_variant"], "source": str(source_path.resolve()),
              "sifted_implementation_version": IMPLEMENTATION_VERSION,
              "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "input_records_retained": len(audit), "feature_records": len(frame),
              "sage_input_records": sum(row["origin"] in {"sage", "sage_original"} for row in audit),
              "synthetic_input_records": sum(row["origin"] == "synthetic" for row in audit),
              "performance_features_used": False, "new_instances_generated": 0,
              "projection_fitted": False, "performance_partition_generated": False,
              "status": "waiting_for_formulation_measurements_and_sifted"}
    if any(row['status'] != 'features_available' for row in audit):
        status.update(status='needs_parameter_review', reason='Some input rows failed; no partial projection is published')
        write_json(args.output_dir / 'summary.json', status)
        raise SystemExit(2)
    if args.fitted_projection:
        from frozen_paper_projection import apply_projection
        try:
            status.update(apply_projection(frame, args.fitted_projection, args.output_dir, spec))
        except (ValueError, AssertionError, KeyError) as error:
            status.update(status='projection_application_needs_review', reason=str(error))
        write_json(args.output_dir / 'summary.json', status)
        print(json.dumps(status, ensure_ascii=False))
        if not status.get('projection_applied'):
            raise SystemExit(2)
        return
    if len(frame):
        frame.to_csv(args.output_dir / "features.csv", encoding="utf-8-sig", float_format="%.17g")
        try:
            transformed, fitted = prepare_features(frame)
        except ValueError as error:
            status.update(status="needs_preprocessing_review", reason=str(error))
        else:
            transformed.to_csv(args.output_dir / "transformed_features.csv", encoding="utf-8-sig", float_format="%.17g")
            write_json(args.output_dir / "prelim.json", fitted)
            status["nonconstant_features"] = len(transformed.columns)
            if args.performance:
                try:
                    status.update(fit_projection(frame, transformed, args))
                except (ValueError, RuntimeError) as error:
                    status.update(status="projection_needs_review", reason=str(error))
    else:
        status["status"] = "needs_parameter_review"
    write_json(args.output_dir / "summary.json", status)
    print(json.dumps(status, ensure_ascii=False))
    if args.performance and not status["projection_fitted"]:
        raise SystemExit(2)
