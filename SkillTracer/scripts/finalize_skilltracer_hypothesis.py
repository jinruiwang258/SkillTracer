#!/usr/bin/env python3
"""Recompute all SkillTracer hypothesis outcomes from saved query evidence.

This readback step enforces the paper's Equation (2) mirror rule and rebuilds
the four requested summary files without additional model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

try:
    from run_skilltracer_hypothesis import (
        ALPHA, N, R, save_tables, semantic_mirror_from_labels, theoretical_test,
    )
    from run_skilltracer_specified import write_json
except ModuleNotFoundError:
    from scripts.run_skilltracer_hypothesis import (
        ALPHA, N, R, save_tables, semantic_mirror_from_labels, theoretical_test,
    )
    from scripts.run_skilltracer_specified import write_json


def finalize(output: Path) -> dict:
    config = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
    if config["functional_dimensions_N"] != N or config["verification_queries_R"] != R:
        raise ValueError("experiment configuration must use N=5 and R=5")
    subsets = config["subsets"]
    results = {}
    changed = 0
    total_queries = 0
    integrity_failures = []
    coverage_gaps = []
    for group in subsets:
        for name in group["skill_ids"]:
            folder = output / "skills" / name
            spec_path = folder / "publisher" / "spec.json"
            if not spec_path.is_file():
                coverage_gaps.append(f"{name}: publisher spec not generated")
                results[name] = {
                    "dataset": "skills_sample", "subset_id": group["subset_id"],
                    "skill_id": name, "skill_name": name, "N": N, "R": R,
                    "detection_outcomes": None, "K": None, "detectability": None,
                    "theoretical_p_value": None, "theoretical_log_p_value": None,
                    "negative_log10_theoretical_p": None,
                    "statistically_significant": None, "status": "not_started",
                }
                continue
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if len(spec["dimensions"]) != N:
                integrity_failures.append(f"{name}: wrong dimension count")
                continue
            outcomes = []
            queries = []
            triggers = []
            for index in range(1, R + 1):
                path = folder / "queries" / f"query_{index:02d}" / "record.json"
                if not path.is_file():
                    coverage_gaps.append(f"{name}: missing query {index}")
                    outcomes.append(None)
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                total_queries += 1
                for field in ("prompt_path", "final_output_path", "artifact_path",
                              "stdout_log_path", "stderr_log_path"):
                    stored_path = Path(record[field])
                    if stored_path.is_absolute():
                        try:
                            record[field] = str(stored_path.relative_to(output))
                        except ValueError:
                            integrity_failures.append(f"{name}: {field} points outside the experiment")
                if record["status"] != "complete":
                    coverage_gaps.append(f"{name}: incomplete query {index}: {record.get('error')}")
                    write_json(path, record)
                    outcomes.append(None)
                    continue
                queries.append(record["query"])
                triggers.append(record["trigger"])
                verification = record["verification"]
                artifact_path = Path(record["artifact_path"])
                if not artifact_path.is_absolute():
                    artifact_path = output / artifact_path
                present = artifact_path.is_file()
                if present != verification["artifact_present"]:
                    integrity_failures.append(f"{name}: artifact presence changed for query {index}")
                if present:
                    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    if digest != verification.get("sha256"):
                        integrity_failures.append(f"{name}: artifact hash mismatch for query {index}")
                final_path = Path(record["final_output_path"])
                if not final_path.is_absolute():
                    final_path = output / final_path
                final_text = final_path.read_text(encoding="utf-8") if final_path.is_file() else ""
                observable = spec["artifact"] in final_text
                semantic = verification.get("semantic_judge") or {}
                mirror = semantic_mirror_from_labels(semantic.get("modules"), spec)
                full = bool(present and observable and verification["module_count_ok"] and mirror)
                if int(full) != record["detection_outcome"]:
                    changed += 1
                verification["artifact_present"] = present
                verification["observable_in_final_response"] = observable
                verification["semantic_mirror"] = mirror
                verification["full_verification"] = full
                record["detection_outcome"] = int(full)
                record["verification_rule_version"] = "paper_equation_2_any_positive_permutation"
                write_json(path, record)
                outcomes.append(int(full))
            if queries and len(set(queries)) != len(queries):
                integrity_failures.append(f"{name}: repeated verification query text")
            if triggers and (len(set(triggers)) != 1 or triggers[0] != spec["trigger_request"]):
                integrity_failures.append(f"{name}: publisher key differs across queries")
            complete = len(outcomes) == R and all(x is not None for x in outcomes)
            if complete:
                k = sum(outcomes)
                p, log_p, neg_log10 = theoretical_test(k)
            else:
                k = p = log_p = neg_log10 = None
            result = {
                "dataset": "skills_sample", "subset_id": group["subset_id"],
                "skill_id": name, "skill_name": name, "N": N, "R": R,
                "detection_outcomes": outcomes, "K": k,
                "detectability": k / R if k is not None else None,
                "theoretical_p_value": p, "theoretical_log_p_value": log_p,
                "negative_log10_theoretical_p": neg_log10,
                "statistically_significant": p < ALPHA if p is not None else None,
                "status": "complete" if complete else "incomplete",
            }
            write_json(folder / "skill_result.json", result)
            results[name] = result
    save_tables(output, subsets, results)
    audit = {
        "skills_expected": sum(len(group["skill_ids"]) for group in subsets),
        "skills_complete": sum(x["status"] == "complete" for x in results.values()),
        "queries_expected": R * sum(len(group["skill_ids"]) for group in subsets),
        "queries_recorded": total_queries,
        "outcomes_corrected_by_equation_2": changed,
        "integrity_checks_passed": not integrity_failures,
        "experiment_complete": not integrity_failures and not coverage_gaps,
        "integrity_failures": integrity_failures,
        "coverage_gaps": coverage_gaps,
    }
    write_json(output / "finalization_audit.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit = finalize(args.output.resolve())
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["experiment_complete"] and audit["skills_complete"] == audit["skills_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
