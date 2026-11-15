"""Count/join checks and explicit-release replay checks; no numerical execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parents[1]
EXAMPLES=CODE/'examples'
EXAMPLE=EXAMPLES/'research_results.json'
RELEASES_ROOT=Path(os.environ['DASHBOARD_RELEASES_ROOT']) if os.environ.get('DASHBOARD_RELEASES_ROOT') else None
sys.path.insert(0,str(ROOT))
from research_support.dashboard_reference.readers.results_adapter import adapt,load_bundle_export,local_results_path,read_results,FINAL_ROLES
from research_support.dashboard_reference.readers.bundle_paths import BundleIndex
from research_support.dashboard_reference.readers.replay import load_selected

class AdapterTests(unittest.TestCase):
    def setUp(self): self.doc, self.sha = read_results(EXAMPLE, allowed_root=EXAMPLES)
    def parse(self, d=None): return adapt(d or self.doc, EXAMPLE.name, self.sha)

    def test_final_counts_curves_missing_and_actual_seeds(self):
        s = self.parse()
        rows = s['records']
        final = [r for r in rows if r['split'] in FINAL_ROLES]
        self.assertEqual(len(final), 50)
        self.assertEqual({r['n_expected'] for r in final}, {200})
        self.assertEqual(len({r['seed'] for r in final}), 2)
        missing = [r for r in final if r['n_completed'] == 0]
        self.assertEqual(len(missing), 1)
        self.assertIsNone(missing[0]['successes']); self.assertIsNone(missing[0]['episodes'])
        partial = next(r for r in final if r['n_completed'] == 173)
        self.assertEqual((partial['episodes'], partial['n_missing']), (173, 27))
        self.assertEqual(len([r for r in rows if r['split'] == 'LEARNING_VALIDATION']), 20)
        self.assertEqual(s['results']['comparisons'], self.doc['comparisons'])
        self.assertIsNone(s['results']['comparisons'][0]['holm_p'])
        self.assertEqual(s['results']['extensions']['learning_metrics'], self.doc['learning_metrics'])

    def test_reject_broken_counts_and_joins(self):
        for defect in ('duplicate_model','duplicate_row','unknown_model','unknown_role','bad_n','bad_sum','bool','negative','nan','learning_step','duplicate_trace','fixture_type','bad_unit_type'):
            with self.subTest(defect=defect):
                d=copy.deepcopy(self.doc)
                if defect=='duplicate_model': d['models'].append(d['models'][0])
                elif defect=='duplicate_row': d['outcomes'].append(d['outcomes'][0])
                elif defect=='unknown_model': d['outcomes'][0]['model_id']='missing'
                elif defect=='unknown_role': d['outcomes'][0]['role']='missing'
                elif defect=='bad_n': d['outcomes'][0]['n_expected']=100
                elif defect=='bad_sum': d['outcomes'][0]['success_count']=199
                elif defect=='bool': d['outcomes'][0]['n_completed']=True
                elif defect=='negative': d['models'][0]['checkpoint_step']=-1
                elif defect=='nan': d['provenance']['x']=float('nan')
                elif defect=='learning_step': d['learning'][0]['checkpoint_step']=25
                elif defect=='duplicate_trace': d['traces'].append(d['traces'][0])
                elif defect=='fixture_type': d['fixture']='true'
                elif defect=='bad_unit_type': d['outcomes'][0]['mean_damage']='unknown'
                with self.assertRaises(ValueError): self.parse(d)

    def test_minimum_real_contract_and_role_denominators(self):
        d=copy.deepcopy(self.doc);d['fixture']=False
        for ds in d['datasets']: ds['manifest_sha256']='a'*64
        s=self.parse(d);self.assertFalse(s['fixture'])
        for role in FINAL_ROLES:
            x=copy.deepcopy(d);next(ds for ds in x['datasets'] if ds['role']==role)['scenario_count']=100
            with self.assertRaises(ValueError): self.parse(x)
        d['datasets'][0]['manifest_sha256']=None
        with self.assertRaises(ValueError): self.parse(d)

    def test_future_additions_and_missing_shared_outputs(self):
        self.doc['new_shared_metric']={'result':None,'reason':'pending'}
        self.doc['comparisons']=[];self.doc['learning']=[];self.doc['traces']=[]
        self.doc['datasets'].append(dict(role='FUTURE_ACCEPTED_ROLE',scenario_count=7,manifest_path='future.json',manifest_sha256=None))
        s=self.parse();self.assertEqual(s['results']['comparisons'],[])
        self.assertEqual(s['results']['extensions']['new_shared_metric'],self.doc['new_shared_metric'])

    def test_optional_shapes_and_final_trace_semantics(self):
        for defect in ('units', 'selection', 'model_type', 'role_type', 'trace_role', 'trace_checkpoint'):
            with self.subTest(defect=defect):
                d=copy.deepcopy(self.doc)
                if defect=='units': d['metric_units']=None
                elif defect=='selection': d['trace_selection']=[]
                elif defect=='model_type': d['outcomes'][0]['model_id']=[]
                elif defect=='role_type': d['outcomes'][0]['role']=[]
                elif defect=='trace_role': d['traces'][0]['role']='LEARNING_VALIDATION'
                else: d['traces'][0]['model_id']='sparse-0-0'
                with self.assertRaises(ValueError): self.parse(d)

    def test_explicit_results_paths_preserve_legacy_final_gates(self):
        self.assertEqual(local_results_path('FINAL_ID/results.json', allowed_root=EXAMPLES).name,'results.json')
        from research_support.dashboard_reference.readers.legacy_csv import local_path
        for value in ('FINAL_ID/results.json','final/results.json','heldout/data.csv','ood/results.csv'):
            with self.assertRaises(ValueError): local_path(value, allowed_root=EXAMPLES)
        for value in (EXAMPLES.parent.parent/'outside.json','../outside.json','https://invalid/a.json',r'\\server\share\x.json','.git/a.json'):
            with self.subTest(path=value),self.assertRaises(ValueError): local_results_path(value,allowed_root=EXAMPLES)

    def test_legacy_wire_id_preserves_count_rules(self):
        self.doc['schema_version']='m4-results-v1'
        self.assertEqual(len(self.parse()['records']),len(adapt({**self.doc,'schema_version':'research-results-v1'},EXAMPLE.name,self.sha)['records']))
        self.doc['outcomes'][0]['n_expected']=1
        with self.assertRaises(ValueError):self.parse()

class ReplayTests(unittest.TestCase):
    def test_metadata_loads_without_release_but_replay_is_unavailable(self):
        bundle=BundleIndex(EXAMPLES)
        study=load_bundle_export(bundle,'synthetic_results_export')
        self.assertTrue(study['records'])
        replay=load_selected(study,0,bundle=bundle)
        self.assertFalse(replay['available'])
        self.assertIn('Release unavailable',replay['reason_unavailable'])

    @unittest.skipUnless(RELEASES_ROOT,'Set DASHBOARD_RELEASES_ROOT explicitly to test external fixture bytes')
    def test_recorded_map_frames_categories_and_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary=Path(temporary)
            shutil.copytree(EXAMPLES,temporary/'metadata')
            shutil.copytree(RELEASES_ROOT/'dashboard_synthetic_examples_v1',temporary/'assets/dashboard_synthetic_examples_v1')
            bundle=BundleIndex(temporary/'metadata',releases_root=temporary/'assets')
            self.assertEqual(bundle.verify_all()['files'],3)
            study=load_bundle_export(bundle,'synthetic_results_export')
            for category in ('matched','representative'):
                self.assertEqual(sum(t['selection_category']==category for t in study['results']['traces']),50)
                index=next(i for i,t in enumerate(study['results']['traces']) if t['available'] and t['selection_category']==category)
                replay=load_selected(study,index,bundle=bundle)
                self.assertTrue(replay['available'],replay)
                self.assertEqual(len(replay['frames']),3)
                self.assertEqual(replay['map']['tiles'][1][0]['elevation'],1)
                self.assertEqual(replay['frames'][1]['agent_hp'],10)
            self.assertFalse(load_selected(study,1,bundle=bundle)['available'])

    @unittest.skipUnless(RELEASES_ROOT,'Set DASHBOARD_RELEASES_ROOT explicitly to test external fixture bytes')
    def test_bad_trace_snapshot_identity_and_hash_are_unavailable(self):
        source=BundleIndex(EXAMPLES,releases_root=RELEASES_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            temporary=Path(temporary)
            shutil.copytree(EXAMPLES,temporary/'metadata')
            shutil.copytree(RELEASES_ROOT/'dashboard_synthetic_examples_v1',temporary/'assets/dashboard_synthetic_examples_v1')
            for defect in ('model','role','terrain','checksum','frame_position','frame_unknown','summary','episode','verification','hash','step','outcome','missing_ids','root_type','manifest_type','outcome_type'):
                with self.subTest(defect=defect):
                    bundle=BundleIndex(temporary/'metadata',releases_root=temporary/'assets')
                    study=load_bundle_export(bundle,'synthetic_results_export');selected=study['results']['traces'][0]
                    trace=source.read_json('synthetic_trajectory');snapshot=source.read_json('synthetic_snapshot')
                    selected['trajectory_sha256']=selected['snapshot_sha256']=None
                    if defect=='model':trace['model_id']='wrong'
                    elif defect=='role':trace['distribution_label']='FINAL_OOD_COMBINED'
                    elif defect=='terrain':trace['scenario_manifest']['map']['tiles'][0][0]['elevation']=99
                    elif defect=='checksum':snapshot['checksum']='b'*64
                    elif defect=='frame_position':trace['frames'][1]['agent_position']=[999,1]
                    elif defect=='frame_unknown':del trace['frames'][1]['agent_hp']
                    elif defect=='summary':selected['episode_summary']['episode_length']=99
                    elif defect=='episode':selected['episode_id']='wrong'
                    elif defect=='verification':selected['verification']={'status':'replay_exact','summary_match':True,'events_match':False}
                    elif defect=='hash':selected['trajectory_sha256']='b'*64
                    elif defect=='step':trace['frames'][1]['step']=3
                    elif defect=='outcome':trace['outcome']['final_hp']=0
                    elif defect=='missing_ids':
                        selected.pop('episode_id');trace.pop('episode_id');trace.pop('episode_record_id')
                    elif defect=='root_type':trace=[]
                    elif defect=='manifest_type':trace['scenario_manifest']=[]
                    elif defect=='outcome_type':trace['outcome']=[]
                    for artifact,value in (('synthetic_trajectory',trace),('synthetic_snapshot',snapshot)):
                        raw=json.dumps(value).encode()
                        bundle.path_for_artifact(artifact).write_bytes(raw)
                        bundle.entries[artifact].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                    result=load_selected(study,0,bundle=bundle)
                    self.assertFalse(result['available'],defect)
                    self.assertTrue(result['reason_unavailable'])


if __name__=='__main__':unittest.main(verbosity=2)
