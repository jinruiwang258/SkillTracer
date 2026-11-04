#!/usr/bin/env python3
"""Independently recheck a saved specified-setting SkillTracer case study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from run_skilltracer_specified import ARTIFACT, PHASES, SCRIPT, verify_artifact
except ModuleNotFoundError:
    from scripts.run_skilltracer_specified import ARTIFACT, PHASES, SCRIPT, verify_artifact


def verify_case(folder: Path) -> dict:
    spec = json.loads((folder / "publisher" / "spec.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "verification.json").read_text(encoding="utf-8"))
    skill_dir = folder / "protected_skill"
    protected_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    trigger = (folder / "publisher" / "trigger.txt").read_text(encoding="utf-8").strip()
    checks = {
        "hidden_payload_in_scripts": (skill_dir / "resources" / SCRIPT).is_file()
        and ARTIFACT in (skill_dir / "resources" / SCRIPT).read_text(encoding="utf-8"),
        "hidden_workflow_in_skill_md": "Optional functional assurance handoff" in protected_text
        and SCRIPT in protected_text,
        "trigger_held_by_publisher": bool(trigger)
        and not (skill_dir / "trigger.txt").exists()
        and trigger not in protected_text,
    }
    phases = {}
    for phase in PHASES:
        project = folder / "projects" / phase
        final = (folder / f"{phase}_final.txt").read_text(encoding="utf-8")
        phases[phase] = verify_artifact(project, spec, final)
    triggered = phases["protected_with_trigger"]
    saved = summary["phases"]["protected_with_trigger"]["verification"]
    labels = (saved.get("semantic_judge") or {}).get("modules", [])
    ids = [item["id"] for item in spec["dimensions"]]
    n = len(ids)
    semantic_recheck = bool(
        triggered["sha256"] and triggered["sha256"] == saved.get("sha256")
        and triggered["observable_in_final_response"]
        and isinstance(labels, list) and len(labels) == 2 * n
        and [item.get("polarity") for item in labels] == ["normal"] * n + ["abnormal"] * n
        and [item.get("dimension_id") for item in labels[:n]] == ids
        and [item.get("dimension_id") for item in labels[n:]] == list(reversed(ids))
    )
    checks["clean_false_activation_absent"] = not phases["clean"]["artifact_present"]
    checks["no_trigger_inactive"] = not phases["protected_no_trigger"]["artifact_present"]
    checks["triggered_evidence_verified"] = triggered["verified"] or semantic_recheck
    checks["original_task_completed_in_all_phases"] = all(
        summary["phases"][phase]["exit_code"] == 0
        and summary["phases"][phase]["utility"]["completed"] is True
        for phase in PHASES
    )
    digest_file = folder / "SHA256SUMS"
    digest_ok = digest_file.is_file()
    if digest_ok:
        for line in digest_file.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            path = folder / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                digest_ok = False
                break
    checks["sha256_manifest_valid"] = digest_ok
    return {"skill": spec["skill"], "passed": all(checks.values()),
            "checks": checks, "phases": phases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_case(args.case_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
