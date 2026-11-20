"""Publication validation rejects mismatched downloads and unsafe archives."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from research_support.verify_publication import load_manifest, safe_name, verify_assets


class PublicationManifestTests(unittest.TestCase):
    def test_relative_paths_reject_traversal_and_host_roots(self):
        for path in ('../x','/x','C:/x','x\\y','x//y','./x'):
            with self.subTest(path=path), self.assertRaises(ValueError): safe_name(path)
        self.assertEqual(safe_name('bundle/data.json'),'bundle/data.json')

    def test_archive_hash_top_level_and_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); archive=root/'asset.zip'
            with zipfile.ZipFile(archive,'w') as z: z.writestr('bundle/data.json','{}')
            entry=dict(filename='asset.zip',bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),top_level_folder='bundle')
            self.assertEqual(verify_assets({'assets':[entry]},root)[0]['files'],1)
            with self.assertRaises(ValueError): verify_assets({'assets':[{**entry,'sha256':'0'*64}]},root)
            with self.assertRaises(ValueError): verify_assets({'assets':[{**entry,'top_level_folder':'wrong'}]},root)

    def test_duplicate_manifest_asset_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'manifest.json'
            entry=dict(filename='asset.zip',bytes=1,sha256='a'*64,top_level_folder='bundle')
            p.write_text(json.dumps(dict(schema_version='frozen-study-publication-v1',assets=[entry,entry])))
            with self.assertRaises(ValueError):load_manifest(p)

    def test_zip_traversal_is_rejected_even_when_asset_digest_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); p=root/'asset.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('bundle/../escape.txt','bad')
            entry=dict(filename=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),top_level_folder='bundle')
            with self.assertRaises(ValueError):verify_assets({'assets':[entry]},root)

    def test_directory_traversal_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); p=root/'asset.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('bundle/../escape/','')
            entry=dict(filename=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),top_level_folder='bundle')
            with self.assertRaises(ValueError):verify_assets({'assets':[entry]},root)

    def test_zip_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); p=root/'asset.zip'
            item=zipfile.ZipInfo('bundle/link');item.create_system=3;item.external_attr=0o120777<<16
            with zipfile.ZipFile(p,'w') as z:z.writestr(item,'../../elsewhere')
            entry=dict(filename=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),top_level_folder='bundle')
            with self.assertRaises(ValueError):verify_assets({'assets':[entry]},root)

    def test_manifest_cannot_require_nested_archive_root(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'manifest.json'
            entry=dict(filename='asset.zip',bytes=1,sha256='a'*64,top_level_folder='extra/bundle')
            p.write_text(json.dumps(dict(schema_version='frozen-study-publication-v1',assets=[entry])))
            with self.assertRaises(ValueError):load_manifest(p)


if __name__=='__main__':unittest.main()
