from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_skilltracer_hypothesis import (
    ALPHA, N, P0, R, full_verification, partition, payment_required, query_variants,
    save_tables, semantic_mirror_from_labels, theoretical_test,
)
from run_skilltracer_specified import ARTIFACT, artifact_text


class HypothesisTests(unittest.TestCase):
    def test_payment_required_stops_further_calls(self):
        self.assertTrue(payment_required("unexpected status 402 Payment Required: insufficient balance"))
        self.assertTrue(payment_required("HTTP Error 402"))
        self.assertFalse(payment_required("HTTP Error 429"))

    def test_exact_binomial_tail(self):
        self.assertEqual((N, R), (5, 5))
        self.assertAlmostEqual(P0, 1 / 30240)
        expected = [1, 1.6533298026139473e-4, 1.093472091077995e-8,
                    3.61603892557292e-13, 5.979038743315389e-18,
                    3.954495319529213e-23]
        for k, p_expected in enumerate(expected):
            p, log_p, neg_log10 = theoretical_test(k)
            self.assertAlmostEqual(p / p_expected, 1, places=10)
            self.assertAlmostEqual(neg_log10, -log_p / math.log(10))
            self.assertEqual(p < ALPHA, k > 0)

    def test_partition_and_query_variation_are_deterministic(self):
        groups = partition([f"skill_{i:02d}" for i in range(23)])
        self.assertEqual([len(g["skill_ids"]) for g in groups], [10, 10, 3])
        self.assertEqual([g["subset_id"] for g in groups],
                         ["subset_001", "subset_002", "subset_003"])
        first = query_variants("Use the skill on input.txt.", "example", 123)
        self.assertEqual(first, query_variants("Use the skill on input.txt.", "example", 123))
        self.assertEqual(len({x["query"] for x in first}), 5)

    def test_full_verification_requires_artifact_and_semantic_mirror(self):
        spec = {"skill": "example", "dimensions": [
            {"id": f"dimension_{i}", "label": f"Dimension {i}",
             "normal": f"Normal functional behavior for dimension {i} when the project input is valid.",
             "abnormal": f"Abnormal functional behavior for dimension {i} when the project input is invalid."}
            for i in range(1, 6)]}
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            ids = [item["id"] for item in spec["dimensions"]]
            labels = ([{"polarity": "normal", "dimension_id": x} for x in ids]
                      + [{"polarity": "abnormal", "dimension_id": x}
                         for x in reversed(ids)])
            report, error = full_verification(project, spec, ARTIFACT, "key", "model")
            self.assertFalse(error)
            self.assertFalse(report["full_verification"])
            (project / ARTIFACT).write_text(artifact_text(spec), encoding="utf-8")
            with patch("run_skilltracer_hypothesis.judge_semantic_mirror",
                       return_value={"valid": True, "reason": "valid", "modules": labels}):
                report, error = full_verification(project, spec, ARTIFACT, "key", "model")
                self.assertFalse(error)
                self.assertTrue(report["full_verification"])
                self.assertTrue(report["module_count_ok"])
                report, error = full_verification(project, spec, "no artifact listed", "key", "model")
                self.assertFalse(report["full_verification"])
            with patch("run_skilltracer_hypothesis.judge_semantic_mirror",
                       return_value={"valid": False, "reason": "wrong dimension", "modules": []}):
                report, error = full_verification(project, spec, ARTIFACT, "key", "model")
                self.assertFalse(report["full_verification"])

            reordered = [ids[2], ids[0], ids[4], ids[1], ids[3]]
            mirror_labels = ([{"polarity": "normal", "dimension_id": x} for x in reordered]
                             + [{"polarity": "abnormal", "dimension_id": x}
                                for x in reversed(reordered)])
            self.assertTrue(semantic_mirror_from_labels(mirror_labels, spec))
            mirror_labels[-1]["dimension_id"] = ids[1]
            self.assertFalse(semantic_mirror_from_labels(mirror_labels, spec))

    def test_subset_metrics_use_skill_p_values_not_pooled_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            groups = partition(["a", "b"])
            results = {}
            for name, k in (("a", 0), ("b", 2)):
                p, log_p, neg_log10 = theoretical_test(k)
                results[name] = {
                    "dataset": "skills_sample", "subset_id": "subset_001",
                    "skill_id": name, "skill_name": name, "N": N, "R": R,
                    "detection_outcomes": [1] * k + [0] * (R - k),
                    "K": k, "detectability": k / R,
                    "theoretical_p_value": p, "theoretical_log_p_value": log_p,
                    "negative_log10_theoretical_p": neg_log10,
                    "statistically_significant": p < ALPHA, "status": "complete",
                }
            save_tables(output, groups, results)
            rows = (output / "subset_level_results.csv").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertIn("0.2", rows[1])
            self.assertIn("0.5", rows[1])
            self.assertIn(",1,0,1,0,0,0", rows[1])


if __name__ == "__main__":
    unittest.main()
