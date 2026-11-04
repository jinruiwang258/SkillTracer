#!/usr/bin/env python3
"""Rejudge completed SkillTracer runs and rebuild independently checked cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil

try:
    from run_skilltracer_specified import (
        PHASES, judge_semantic_mirror, judge_utility, verify_artifact,
        write_case_study, write_json,
    )
    from verify_skilltracer_case import verify_case
except ModuleNotFoundError:
    from scripts.run_skilltracer_specified import (
        PHASES, judge_semantic_mirror, judge_utility, verify_artifact,
        write_case_study, write_json,
    )
    from scripts.verify_skilltracer_case import verify_case


def process_skill(folder: Path, output: Path, root: Path, key: str) -> dict:
    record = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        return record
    spec = json.loads((folder / "publisher_spec.json").read_text(encoding="utf-8"))
    instruction = (root / "data" / "instruction" / "skills_sample"
                   / folder.name / "instruction.md").read_text(encoding="utf-8")
    for phase in PHASES:
        phase_dir = folder / phase
        prior = json.loads((phase_dir / "verification.json").read_text(encoding="utf-8"))
        project = phase_dir / "project"
        final = (phase_dir / "final.txt").read_text(encoding="utf-8")
        run = dict(prior, project=str(project), final_text=final)
        prior["utility"] = judge_utility(key, spec["model"], instruction, run)
        verification = verify_artifact(project, spec, final)
        if verification["artifact_present"]:
            semantic = judge_semantic_mirror(key, spec["model"], project, spec)
            verification["semantic_judge"] = semantic
            verification["semantic_mirror"] = semantic["valid"]
            verification["verified"] = bool(
                verification["observable_in_final_response"]
                and (verification["content_match"] or semantic["valid"])
                and (verification["structural_mirror"] or semantic["valid"])
            )
        prior["verification"] = verification
        write_json(phase_dir / "verification.json", prior)
        record["phases"][phase] = prior
    phases = record["phases"]
    record["success"] = bool(
        all(phases[p]["exit_code"] == 0 and phases[p]["utility"]["completed"] for p in PHASES)
        and not phases["clean"]["verification"]["artifact_present"]
        and not phases["protected_no_trigger"]["verification"]["artifact_present"]
        and phases["protected_with_trigger"]["verification"]["verified"]
    )
    write_json(folder / "run_summary.json", record)
    case = output / "successful_cases" / folder.name
    if case.exists():
        shutil.rmtree(case)
    if not record["success"]:
        return record
    (case / "publisher").mkdir(parents=True)
    shutil.copytree(folder / "protected_skill", case / "protected_skill")
    shutil.copy2(folder / "trigger.txt", case / "publisher" / "trigger.txt")
    shutil.copy2(folder / "publisher_spec.json", case / "publisher" / "spec.json")
    shutil.copy2(folder / "run_summary.json", case / "verification.json")
    for phase in PHASES:
        shutil.copytree(folder / phase / "project", case / "projects" / phase)
        shutil.copy2(folder / phase / "final.txt", case / f"{phase}_final.txt")
    write_case_study(case, spec, record, instruction)
    hashes = [f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.relative_to(case)}"
              for item in sorted(case.rglob("*")) if item.is_file()]
    (case / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    check = verify_case(case)
    write_json(case / "independent_verification.json", check)
    if not check["passed"]:
        record["success"] = False
        record["verification_error"] = "independent case check failed"
        write_json(folder / "run_summary.json", record)
        shutil.rmtree(case)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project root containing data/instruction/skills_sample")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.results.resolve()
    root = args.root.resolve()
    key = args.api_key_file.read_text(encoding="utf-8").strip()
    folders = sorted(p for p in (output / "all_runs").iterdir() if p.is_dir())
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_skill, folder, output, root, key): folder.name for folder in folders}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['skill']}: success={result.get('success', False)}", flush=True)
    write_json(output / "summary.json", {
        "total": len(folders), "completed": len(results),
        "successful": sum(bool(r.get("success")) for r in results),
        "results": sorted(results, key=lambda r: r["skill"]),
        "utility_judge": "GPT-5.6-Sol with output file excerpts",
        "semantic_mirror_judge": "GPT-5.6-Sol module classification",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
