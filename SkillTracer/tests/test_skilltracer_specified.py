from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_skilltracer_specified import ARTIFACT, artifact_text, make_package, verify_artifact


class SkillTracerSpecifiedTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "skill": "example", "dimensions": [
                {"id": f"dimension_{i}", "label": f"Dimension {i}",
                 "normal": f"Expected functional behavior for dimension {i} when normal inputs are provided and the requested task completes correctly.",
                 "abnormal": f"Exception behavior for dimension {i} when invalid inputs are supplied and the requested task cannot complete correctly."}
                for i in range(1, 4)
            ],
        }

    def test_generated_script_produces_mirrored_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "SKILL.md").write_text("# Example\n\nDo the task.\n")
            package = root / "package"
            make_package(source, package, self.spec)
            project = root / "project"
            project.mkdir()
            subprocess.run(["bash", str(package / "resources" / "generate_functional_summary.sh")],
                           cwd=project, check=True)
            result = verify_artifact(project, self.spec, ARTIFACT)
            self.assertTrue(result["verified"])
            self.assertEqual([m["id"] for m in result["modules"][-3:]],
                             ["dimension_3", "dimension_2", "dimension_1"])

    def test_reordered_or_unreported_artifact_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            text = artifact_text(self.spec)
            (project / ARTIFACT).write_text(text)
            self.assertFalse(verify_artifact(project, self.spec, "done")["verified"])
            (project / ARTIFACT).write_text(text.replace("(`dimension_3`)", "(`dimension_1`)", 1))
            self.assertFalse(verify_artifact(project, self.spec, ARTIFACT)["verified"])


if __name__ == "__main__":
    unittest.main()
