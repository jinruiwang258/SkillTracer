#!/usr/bin/env python3
"""Skill-level theoretical binomial test for specified-setting SkillTracer.

The experiment runs five distinct, keyed Codex verification queries per skill.
Theoretical p-values describe the idealized uniform-permutation null only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import sys
import threading
import time

import scipy
from scipy.stats import binom

try:
    from run_skilltracer_specified import (
        ARTIFACT, BASE_URL, MODEL, SCRIPT, generate_spec, judge_semantic_mirror,
        make_package, run_codex, verify_artifact, write_json,
    )
except ModuleNotFoundError:
    from scripts.run_skilltracer_specified import (
        ARTIFACT, BASE_URL, MODEL, SCRIPT, generate_spec, judge_semantic_mirror,
        make_package, run_codex, verify_artifact, write_json,
    )

N = 5
R = 5
ALPHA = 0.05
SUBSET_SIZE = 10
P0 = math.factorial(N) / math.factorial(2 * N)
QUERY_TEMPLATES = (
    "{task}",
    "In the current project, complete this task and identify the resulting files: {task}",
    "Using the supplied project inputs, carry out this request and report the results: {task}",
    "Please perform the following skill-related task, then give a concise completion report: {task}",
    "Work through the available inputs and produce the requested deliverables for this task: {task}",
)
STOP_EVENT = threading.Event()


def payment_required(message: str) -> bool:
    lowered = message.lower()
    return "402 payment required" in lowered or "insufficient balance" in lowered \
        or "http error 402" in lowered


def available_skills(root: Path) -> list[str]:
    source = root / "data" / "skills_sample"
    instructions = root / "data" / "instruction" / "skills_sample"
    names = sorted(p.name for p in source.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())
    missing = [name for name in names if not (instructions / name / "instruction.md").is_file()]
    if missing:
        raise ValueError(f"missing instruction.md for: {', '.join(missing)}")
    return names


def partition(names: list[str]) -> list[dict]:
    return [
        {"subset_id": f"subset_{start // SUBSET_SIZE + 1:03d}",
         "skill_ids": names[start:start + SUBSET_SIZE]}
        for start in range(0, len(names), SUBSET_SIZE)
    ]


def query_variants(task: str, skill: str, seed: int) -> list[dict]:
    variants = [
        {"variant_id": index + 1, "query": template.format(task=task.strip())}
        for index, template in enumerate(QUERY_TEMPLATES)
    ]
    digest = hashlib.sha256(f"{seed}:{skill}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(variants)
    assert len({v["query"] for v in variants}) == R
    return variants


def theoretical_test(k: int) -> tuple[float, float, float]:
    if not 0 <= k <= R:
        raise ValueError("K must be between zero and R")
    log_p = 0.0 if k == 0 else float(binom.logsf(k - 1, R, P0))
    return math.exp(log_p), log_p, -log_p / math.log(10)


def semantic_mirror_from_labels(labels: object, spec: dict) -> bool:
    """Equation (2): any positive permutation and its negative reverse.

    Allowing all N! positive permutations is necessary for p0=N!/(2N)!.
    """
    if not isinstance(labels, list) or len(labels) != 2 * N:
        return False
    if not all(isinstance(item, dict) for item in labels):
        return False
    polarities = [item.get("polarity") for item in labels]
    if polarities != ["normal"] * N + ["abnormal"] * N:
        return False
    positive = [item.get("dimension_id") for item in labels[:N]]
    negative = [item.get("dimension_id") for item in labels[N:]]
    if not all(isinstance(dimension, str) for dimension in positive + negative):
        return False
    expected = {item["id"] for item in spec["dimensions"]}
    return len(expected) == N and len(set(positive)) == N \
        and set(positive) == expected and negative == list(reversed(positive))


def full_verification(project: Path, spec: dict, final_text: str, key: str,
                      model: str) -> tuple[dict, bool]:
    """Apply V(y*) = I[artifact observable] * semantic Mirror(M-hat).

    A transient semantic-judge API error makes the query incomplete, not a
    negative detection. The caller may retry that query on resume.
    """
    report = verify_artifact(project, spec, final_text)
    report["module_count_ok"] = False
    report["full_verification"] = False
    if not report["artifact_present"]:
        return report, False

    artifact = (project / ARTIFACT).read_text(encoding="utf-8", errors="replace")
    # The generated functional summary uses two numbered sections. Checking
    # their module counts prevents a judge from filling in missing modules.
    normal, separator, abnormal = artifact.partition("## Invalid or abnormal conditions")
    numbered = re.compile(r"^\s*\d+\.\s+\S.*$", re.M)
    normal_count = len(numbered.findall(normal))
    abnormal_count = len(numbered.findall(abnormal)) if separator else 0
    report["numbered_modules"] = {"normal_section": normal_count,
                                  "abnormal_section": abnormal_count}
    report["module_count_ok"] = normal_count == N and abnormal_count == N

    semantic = None
    for attempt in range(3):
        semantic = judge_semantic_mirror(key, model, project, spec)
        if not str(semantic.get("reason", "")).startswith("judge error:"):
            break
        time.sleep(2 ** attempt)
    assert semantic is not None
    report["semantic_judge"] = semantic
    report["semantic_mirror"] = semantic_mirror_from_labels(semantic.get("modules"), spec)
    if str(semantic.get("reason", "")).startswith("judge error:"):
        report["validation_error"] = semantic["reason"]
        return report, True
    report["full_verification"] = bool(
        report["artifact_present"]
        and report["observable_in_final_response"]
        and report["module_count_ok"]
        and report["semantic_mirror"]
    )
    return report, False


def run_skill(name: str, subset_id: str, args: argparse.Namespace, key: str) -> dict:
    root = args.root
    skill_dir = args.output / "skills" / name
    instruction_dir = root / "data" / "instruction" / "skills_sample" / name
    task = (instruction_dir / "instruction.md").read_text(encoding="utf-8").strip()
    spec_path = skill_dir / "publisher" / "spec.json"
    package = skill_dir / "protected_skill"
    try:
        if STOP_EVENT.is_set():
            raise RuntimeError("experiment stopped after HTTP 402 Payment Required")
        if spec_path.is_file():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if len(spec["dimensions"]) != N:
                raise ValueError("saved spec does not have five dimensions")
        else:
            source_text = (root / "data" / "skills_sample" / name / "SKILL.md").read_text(encoding="utf-8")
            for attempt in range(3):
                try:
                    spec = generate_spec(name, source_text, task, key, args.model,
                                         dimension_count=N)
                    break
                except (ValueError, json.JSONDecodeError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            write_json(spec_path, spec)
        trigger = spec["trigger_request"]
        (skill_dir / "publisher" / "trigger.txt").write_text(trigger + "\n", encoding="utf-8")
        if not package.is_dir():
            make_package(root / "data" / "skills_sample" / name, package, spec)
        variants = query_variants(task, name, args.seed)
        outcomes: list[int | None] = []
        for index, variant in enumerate(variants, 1):
            if STOP_EVENT.is_set():
                break
            query_dir = skill_dir / "queries" / f"query_{index:02d}"
            record_path = query_dir / "record.json"
            if record_path.is_file():
                saved = json.loads(record_path.read_text(encoding="utf-8"))
                if saved.get("query") == variant["query"] and saved.get("trigger") == trigger \
                        and saved.get("status") == "complete":
                    outcomes.append(saved["detection_outcome"])
                    continue
            if query_dir.exists():
                archive = skill_dir / "failed_attempts"
                archive.mkdir(parents=True, exist_ok=True)
                destination = archive / f"query_{index:02d}_{time.time_ns()}"
                shutil.move(str(query_dir), str(destination))
            query_dir.mkdir(parents=True, exist_ok=True)
            run = {}
            verification = {}
            try:
                run = run_codex(args, name, package, instruction_dir,
                                variant["query"], f"verification_{index}",
                                trigger, key, query_dir)
                project = Path(run["project"])
                if run["exit_code"] != 0:
                    stderr = (query_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")
                    if payment_required(stderr):
                        STOP_EVENT.set()
                    raise RuntimeError(f"Codex exit code {run['exit_code']}")
                verification, judge_error = full_verification(
                    project, spec, run["final_text"], key, args.model)
                if judge_error:
                    if payment_required(str(verification.get("validation_error", ""))):
                        STOP_EVENT.set()
                    raise RuntimeError("semantic judge API error")
                outcome = int(verification["full_verification"])
                status, error = "complete", None
            except Exception as exc:
                if payment_required(str(exc)):
                    STOP_EVENT.set()
                outcome = None
                status, error = "error", f"{type(exc).__name__}: {exc}"
            record = {
                "dataset": "skills_sample", "subset_id": subset_id,
                "skill_id": name, "skill_name": name,
                "query_index": index, "variant_id": variant["variant_id"],
                "query": variant["query"], "trigger": trigger,
                "prompt_path": str(query_dir.relative_to(args.output) / "prompt.txt"),
                "prompt": (query_dir / "prompt.txt").read_text(encoding="utf-8")
                          if (query_dir / "prompt.txt").is_file() else None,
                "output": run.get("final_text"),
                "final_output_path": str(query_dir.relative_to(args.output) / "final.txt"),
                "artifact_path": str(query_dir.relative_to(args.output) / "project" / ARTIFACT),
                "stdout_log_path": str(query_dir.relative_to(args.output) / "stdout.log"),
                "stderr_log_path": str(query_dir.relative_to(args.output) / "stderr.log"),
                "exit_code": run.get("exit_code"),
                "elapsed_seconds": run.get("elapsed_seconds"),
                "verification": verification,
                "detection_outcome": outcome,
                "status": status, "error": error,
            }
            write_json(record_path, record)
            outcomes.append(outcome)
            print(f"{name} query {index}/{R}: {status} X={outcome}", flush=True)
        if len(outcomes) == R and all(x is not None for x in outcomes):
            k = sum(outcomes)
            p, log_p, neg_log10 = theoretical_test(k)
            status = "complete"
        else:
            k = p = log_p = neg_log10 = None
            status = "incomplete"
        result = {
            "dataset": "skills_sample", "subset_id": subset_id,
            "skill_id": name, "skill_name": name,
            "N": N, "R": R, "detection_outcomes": outcomes,
            "K": k, "detectability": k / R if k is not None else None,
            "theoretical_p_value": p, "theoretical_log_p_value": log_p,
            "negative_log10_theoretical_p": neg_log10,
            "statistically_significant": p < ALPHA if p is not None else None,
            "status": status,
        }
    except Exception as exc:
        if payment_required(str(exc)):
            STOP_EVENT.set()
        result = {
            "dataset": "skills_sample", "subset_id": subset_id,
            "skill_id": name, "skill_name": name,
            "N": N, "R": R, "detection_outcomes": None,
            "K": None, "detectability": None,
            "theoretical_p_value": None, "theoretical_log_p_value": None,
            "negative_log10_theoretical_p": None,
            "statistically_significant": None,
            "status": "error", "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(skill_dir / "skill_result.json", result)
    return result


def save_tables(output: Path, subsets: list[dict], results: dict[str, dict]) -> None:
    skill_fields = [
        "dataset", "subset_id", "skill_id", "skill_name", "N", "R",
        "detection_outcomes", "K", "detectability", "theoretical_p_value",
        "theoretical_log_p_value", "negative_log10_theoretical_p",
        "statistically_significant", "status", "error",
    ]
    with (output / "skill_level_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=skill_fields)
        writer.writeheader()
        for subset in subsets:
            for name in subset["skill_ids"]:
                row = results.get(name, {})
                writer.writerow({field: json.dumps(row[field]) if field == "detection_outcomes"
                                 and row.get(field) is not None else row.get(field)
                                 for field in skill_fields})
    subset_fields = [
        "dataset", "subset_id", "number_of_skills", "complete_skills",
        "mean_detectability", "median_theoretical_p_value",
        "mean_negative_log10_theoretical_p", "significant_skills_rate",
        "perfect_detection_rate", "k_0", "k_1", "k_2", "k_3", "k_4", "k_5",
    ]
    with (output / "subset_level_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=subset_fields)
        writer.writeheader()
        for subset in subsets:
            items = [results[name] for name in subset["skill_ids"] if name in results]
            complete = [item for item in items if item["status"] == "complete"]
            ready = len(complete) == len(subset["skill_ids"])
            row = {
                "dataset": "skills_sample", "subset_id": subset["subset_id"],
                "number_of_skills": len(subset["skill_ids"]),
                "complete_skills": len(complete),
                "mean_detectability": statistics.mean(x["detectability"] for x in complete) if ready else None,
                "median_theoretical_p_value": statistics.median(x["theoretical_p_value"] for x in complete) if ready else None,
                "mean_negative_log10_theoretical_p": statistics.mean(x["negative_log10_theoretical_p"] for x in complete) if ready else None,
                "significant_skills_rate": statistics.mean(x["statistically_significant"] for x in complete) if ready else None,
                "perfect_detection_rate": statistics.mean(x["K"] == R for x in complete) if ready else None,
            }
            row.update({f"k_{k}": sum(x["K"] == k for x in complete) for k in range(R + 1)})
            writer.writerow(row)
    with (output / "query_level_results.jsonl").open("w", encoding="utf-8") as stream:
        for subset in subsets:
            for name in subset["skill_ids"]:
                for index in range(1, R + 1):
                    path = output / "skills" / name / "queries" / f"query_{index:02d}" / "record.json"
                    if path.is_file():
                        stream.write(json.dumps(json.loads(path.read_text(encoding="utf-8")),
                                                ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=str,
                        help="Optional output path recorded in experiment metadata")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--limit-skills", type=int, help="Focused smoke run; omit for all skills")
    parser.add_argument("--skills", nargs="+", help="Focused named-skill run; omit for all skills")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    args.remote_output_root = args.remote_output_root or str(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        parser.error("empty API key file")
    names = available_skills(args.root)
    if args.skills:
        unknown = sorted(set(args.skills) - set(names))
        if unknown:
            parser.error(f"unknown skills: {', '.join(unknown)}")
        selected = set(args.skills)
        names = [name for name in names if name in selected]
    if args.limit_skills is not None:
        names = names[:args.limit_skills]
    subsets = partition(names)
    config = {
        "dataset": "skills_sample", "source_project": str(args.root),
        "execution_project": str(args.root),
        "remote_output_root": args.remote_output_root,
        "query_file_paths_relative_to_output_root": True,
        "source_dataset": str(args.root / "data" / "skills_sample"),
        "dataset_order": "lexicographically sorted skill directory names with SKILL.md",
        "subsets": subsets, "subset_size": SUBSET_SIZE,
        "setting": "specified", "platform": "Codex CLI", "model": args.model,
        "base_url": BASE_URL, "credential_stored": False,
        "functional_dimensions_N": N, "verification_queries_R": R,
        "alpha": ALPHA, "random_seed": args.seed,
        "query_variation_templates": QUERY_TEMPLATES,
        "theoretical_null": {
            "hypothesis": "2N distinct semantic modules are uniformly randomly permuted",
            "single_query_probability_exact": "5!/10! = 1/30240",
            "single_query_probability": P0,
            "repetition_assumption": "five independent Bernoulli draws under the idealized null",
            "test": "one-sided exact binomial survival function",
            "calculation": "scipy.stats.binom.logsf(K-1, 5, 1/30240); K=0 -> 0",
            "interpretation": "theoretical p-value, not an empirical real-world false-positive rate",
        },
        "verification_rule": "artifact file exists and is named in final response; five normal and five abnormal numbered modules; LLM semantic labels satisfy any permutation of the five positive dimensions followed by the corresponding reverse-ordered negative dimensions",
        "scipy_version": scipy.__version__,
        "codex_bin": args.codex_bin, "timeout_seconds": args.timeout,
        "workers": args.workers,
        "stop_on_payment_required": True,
        "failed_query_attempts_retained": True,
    }
    write_json(args.output / "experiment_config.json", config)
    results = {}
    subset_by_skill = {name: group["subset_id"] for group in subsets for name in group["skill_ids"]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_skill, name, subset_by_skill[name], args, key): name for name in names}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[result["skill_id"]] = result
            save_tables(args.output, subsets, results)
            print(f"{result['skill_id']}: {result['status']} K={result['K']}", flush=True)
    return 0 if all(x["status"] == "complete" for x in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
