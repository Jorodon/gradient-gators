"""Portable output references and supported legacy input formats."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_support.dashboard_reference.readers import legacy_csv,normalized_model
from research_support.dashboard_reference.readers.results_adapter import load_export,load_bundle_export,adapt
from research_support.dashboard_reference.readers.bundle_paths import BundleIndex


class PortableOutputTests(unittest.TestCase):
    def test_results_sources_are_relative_or_artifact_ids(self):
        source=CODE/'examples/research_results.json'
        study=load_export(source,allowed_root=source.parent)
        self.assertEqual(study['sources'],['research_results.json'])
        self.assertEqual(study['results']['export_path'],'research_results.json')
        self.assertEqual({row['source'] for row in study['records']},{'research_results.json'})
        bundle_study=load_bundle_export(BundleIndex(source.parent),'synthetic_results_export')
        self.assertEqual(bundle_study['sources'],['artifact:synthetic_results_export'])
        with self.assertRaises(ValueError):adapt(json.loads(source.read_text()),source,'a'*64)

    def test_legacy_csv_sources_are_relative_and_final_rows_stay_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'episodes.csv'
            fields=['split','cell','seed','checkpoint_timestep','success','truncated','end_reason','episode_length','parent_id','invalid_moves','damage_taken']
            with path.open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                for i in range(300):writer.writerow(dict(split='TRAIN',cell='synthetic',seed=1,checkpoint_timestep=0,success=False,truncated=True,end_reason='timestep_limit',episode_length=1,parent_id=str(i),invalid_moves=0,damage_taken=0))
                writer.writerow(dict(split='FINAL_ID',cell='synthetic',seed=1,success=True))
            rows,warnings=legacy_csv.episode_records(path,allowed_root=root)
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['source'],'episodes.csv')
            self.assertTrue(any('Excluded' in message for message in warnings))
            self.assertFalse(hasattr(legacy_csv,'recent_studies'))
            self.assertFalse(hasattr(legacy_csv,'MANUAL_ROOTS'))

    def test_normalized_fixture_retains_strict_final_gate(self):
        document=json.loads((CODE/'examples/normalized_studies.json').read_text())
        normalized_model.validate(document)
        document['studies'][0]['records'][0]['split']='FINAL_ID'
        with self.assertRaises(ValueError):normalized_model.validate(document)


if __name__=='__main__':unittest.main()
