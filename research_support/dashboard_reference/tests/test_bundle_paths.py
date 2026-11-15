"""Offline relocation, missing-release, identity, boundary and tamper checks."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from research_support.dashboard_reference.readers.bundle_paths import BundleIndex,portable_relative,within


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.bundle=self.root/'bundle';self.bundle.mkdir()
        self.releases=self.root/'releases'
        (self.releases/'synthetic_release/payload').mkdir(parents=True)
        raw=b'{"fixture":true}'
        (self.bundle/'metadata.json').write_bytes(raw)
        (self.releases/'synthetic_release/payload/world.json').write_bytes(raw)
        def entry(artifact,storage,path):
            return dict(artifact_id=artifact,storage=storage,relative_path=path,
                **({'release_id':'synthetic_release'} if storage=='release' else {}),
                bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),category='synthetic_test',
                source_ref=dict(experiment_id='synthetic_test',artifact_path=artifact+'.json',sha256=hashlib.sha256(raw).hexdigest()),source_byte_identical=True)
        self.manifest=dict(schema_version='gradient-gators-reference-bundle-v2',files=[entry('metadata','repository','metadata.json'),entry('world','release','payload/world.json')])
        self.write_manifest()
    def tearDown(self):self.temp.cleanup()
    def write_manifest(self):
        (self.bundle/'bundle_manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')
    def index(self,release=True):return BundleIndex(self.bundle,releases_root=self.releases if release else None)

    def test_repository_only_without_release_and_clear_missing_world(self):
        index=self.index(False)
        self.assertTrue(index.read_json('metadata')['fixture'])
        self.assertEqual(index.verify_all(include_releases=False)['skipped_release_files'],1)
        with self.assertRaisesRegex(FileNotFoundError,'explicitly configure releases_root'):index.read_json('world')
        with self.assertRaises(FileNotFoundError):index.verify_all()

    def test_portable_relocation_with_same_indexed_bytes(self):
        moved=self.root/'relocated';moved.mkdir()
        shutil.copytree(self.bundle,moved/'repo')
        shutil.copytree(self.releases,moved/'assets')
        index=BundleIndex(moved/'repo',releases_root=moved/'assets')
        self.assertEqual(index.verify_all()['files'],2)
        self.assertEqual(index.read_bytes('metadata'),index.read_bytes('world'))
        self.assertEqual(index.artifact_for_source('synthetic_test','world.json'),'world')

    def test_missing_file_tampered_size_and_same_size_hash_rejected(self):
        index=self.index();path=self.releases/'synthetic_release/payload/world.json'
        original=path.read_bytes();path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'size'):index.read_json('world')
        path.write_bytes(original.replace(b'true',b'null'))
        with self.assertRaisesRegex(ValueError,'hash'):index.read_json('world')
        path.unlink()
        with self.assertRaises(FileNotFoundError):index.read_json('world')

    def test_traversal_network_hidden_drive_and_ambiguous_paths_rejected(self):
        for value in ('../outside.json','/outside.json','Z:/outside.json','.git/config','x/../y','x//y','x/./y','x\\y','https://invalid/x'):
            with self.subTest(path=value),self.assertRaises(ValueError):portable_relative(value)
        for root in ('https://invalid/repo',r'\\server\share'):
            with self.assertRaises(ValueError):BundleIndex(root)

    def test_resolved_symlink_escape_is_rejected(self):
        target=self.bundle/'link.json'
        original=Path.resolve
        def resolved(path,*args,**kwargs):
            return self.root/'outside.json' if path==target else original(path,*args,**kwargs)
        with patch.object(Path,'resolve',resolved):
            with self.assertRaisesRegex(ValueError,'symlink'):within(self.bundle,'link.json')

    def test_duplicate_artifact_storage_and_source_identities_rejected(self):
        for change in ('id','storage','source'):
            candidate=copy.deepcopy(self.manifest['files'][0])
            candidate['artifact_id']='extra';candidate['relative_path']='different.json'
            candidate['source_ref']['artifact_path']='different.json'
            if change=='id':candidate['artifact_id']='metadata'
            if change=='storage':candidate['relative_path']='METADATA.json'
            if change=='source':candidate['source_ref']['artifact_path']='METADATA.json'
            self.manifest['files'].append(candidate);self.write_manifest()
            with self.assertRaises(ValueError):self.index()
            self.manifest['files'].pop()

    def test_alias_requires_matching_bytes_and_unambiguous_source(self):
        alias=dict(artifact_id='world',source_ref=dict(experiment_id='second_producer',artifact_path='same_world.json',sha256=self.manifest['files'][1]['sha256']),reason='byte_identical_world_deduplication')
        self.manifest['source_aliases']=[alias];self.write_manifest()
        index=self.index()
        self.assertEqual(index.artifact_for_source('second_producer','same_world.json'),'world')
        with self.assertRaises(ValueError):index.artifact_for_source('second_producer','same_world.json','0'*64)
        alias['source_ref']['sha256']='0'*64;self.write_manifest()
        with self.assertRaises(ValueError):self.index()

    def test_bad_manifest_counts_hashes_and_duplicate_json_keys_rejected(self):
        for key,value in (('bytes',True),('sha256','invalid'),('source_byte_identical',None),('release_id','../escape')):
            previous=self.manifest['files'][1].get(key)
            self.manifest['files'][1][key]=value;self.write_manifest()
            with self.assertRaises(ValueError):self.index()
            self.manifest['files'][1][key]=previous
        (self.bundle/'bundle_manifest.json').write_text('{"files":[],"files":[]}',encoding='utf-8')
        with self.assertRaises(ValueError):self.index()


if __name__=='__main__':unittest.main(verbosity=2)
