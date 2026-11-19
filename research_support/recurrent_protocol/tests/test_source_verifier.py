"""Small synthetic source fixtures; no RL imports, model loads or learning."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "verify_source.py"
SPEC = importlib.util.spec_from_file_location("publication_source_verifier", MODULE_PATH)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class SourceVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Executing this file would fail; AST parsing must not execute it.
        self.original = b'raise RuntimeError("must not execute scientific modules")\n'
        (self.root / "module.py").write_bytes(self.original)
        self.entry = {
            "original_path": "module.py", "published_path": "module.py",
            "original_sha256": digest(self.original), "published_sha256": digest(self.original),
            "byte_identical": True, "transformation": None,
        }
        self.manifest = {
            "schema_version": 1, "study_id": "synthetic_source_fixture",
            "original_source_seal_sha256": "a" * 64,
            "original_first_party_count": 1, "entries": [self.entry],
        }
        self.catalog = {
            "schema_version": 1, "study_id": "synthetic_source_fixture",
            "original_source_seal_sha256": "a" * 64,
            "original_files": {"module.py": digest(self.original)},
        }
        self.publish_manifest()

    def publish_manifest(self):
        data = json.dumps(self.manifest).encode()
        (self.root / verifier.MANIFEST_NAME).write_bytes(data)
        self.catalog["publication_manifest_sha256"] = digest(data)

    def test_valid_archive_is_read_without_execution(self):
        receipt = verifier.verify(self.root, self.catalog)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["scientific_modules_executed"], 0)
        self.assertEqual(receipt["python_files_syntax_checked"], 1)
        self.assertEqual(len(list(self.root.iterdir())), 2)

    def test_corrupt_source_is_rejected(self):
        (self.root / "module.py").write_text("changed = True\n")
        with self.assertRaisesRegex(ValueError, "Published bytes differ"):
            verifier.verify(self.root, self.catalog)

    def test_modified_manifest_is_rejected(self):
        (self.root / verifier.MANIFEST_NAME).write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            verifier.verify(self.root, self.catalog)

    def test_traversal_is_rejected(self):
        self.entry["published_path"] = "../module.py"
        self.publish_manifest()
        with self.assertRaisesRegex(ValueError, "traversal"):
            verifier.verify(self.root, self.catalog)

    def test_original_identity_mismatch_is_rejected(self):
        self.entry["original_sha256"] = "b" * 64
        self.publish_manifest()
        with self.assertRaisesRegex(ValueError, "Original identity differs"):
            verifier.verify(self.root, self.catalog)

    def test_declared_derivative_is_allowed_and_reported(self):
        changed = b'raise RuntimeError("still not executed")\n'
        (self.root / "module.py").write_bytes(changed)
        self.entry.update(published_sha256=digest(changed), byte_identical=False,
                          transformation="synthetic documented path substitution")
        self.publish_manifest()
        receipt = verifier.verify(self.root, self.catalog)
        self.assertEqual(receipt["transformed_files"], ["module.py"])

    def test_false_identity_declaration_is_rejected(self):
        self.entry["byte_identical"] = False
        self.publish_manifest()
        with self.assertRaisesRegex(ValueError, "Byte-identity declaration differs"):
            verifier.verify(self.root, self.catalog)

    def test_unknown_catalog_version_is_rejected(self):
        self.catalog["schema_version"] = 99
        with self.assertRaisesRegex(ValueError, "catalog schema"):
            verifier.verify(self.root, self.catalog)


if __name__ == "__main__":
    unittest.main()
