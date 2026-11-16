"""Synthetic producer-to-dashboard integration; no measured outcomes or learning."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_support.dashboard_reference.protocols import RECURRENT_PROTOCOL_ID, get_protocol
from research_support.dashboard_reference.readers.recurrent_export import build_transport, export_campaign, byte_hash, write, digest, engine_digest
from research_support.dashboard_reference.readers.results_adapter import adapt, load_bundle_export
from research_support.dashboard_reference.readers.bundle_paths import BundleIndex


def producer_fixture():
    p = get_protocol(RECURRENT_PROTOCOL_ID)
    models, outcomes, curves = [], [], []
    for slot, seed in enumerate(p['primary_seeds']):
        for reward in p['rewards']:
            job = f'synthetic_{slot}_{reward}'
            cps = [dict(nominal_training_steps=t, actual_training_steps=t+(123 if 0<t<p['budget'] else 0),
                        optimized_training_steps=t+(123 if 0<t<p['budget'] else -17 if t else 0),
                        sha256='a'*64) for t in p['grid']]
            models.append(dict(model_id=job, reward=reward, seed=seed, job_sha256='b'*64, checkpoints=cps))
            def point(role, cp, n):
                return dict(model_id=job, reward=reward, seed=seed, role=role, n=n, complete=True,
                            nominal_steps=cp['nominal_training_steps'], actual_steps=cp['actual_training_steps'],
                            optimized_steps=cp['optimized_training_steps'], success_count=n//2,
                            death_count=0, timeout_count=n-n//2, success_fraction=.5,
                            evaluation_steps=n*20, damage_mean=2.5, length_mean=20., length_success_mean=20.,
                            length_failure_mean=20., raw_return_mean=-3.2, illegal_moves_mean=1.5,
                            illegal_moves_fraction_pooled=.075, wait_mean=.5)
            outcomes.append(point('TRAIN900', cps[-1], 900))
            for role in p['role_counts']:
                if role.startswith('FINAL_'): outcomes.append(point(role, cps[-1], 200))
            curves.append(dict(model_id=job, reward=reward, seed=seed, points=[point('LEARNING_VALIDATION', cp, 100) for cp in cps],
                               normalized_aulc=.5, time_to_sustained_80=None, threshold_status='censored',
                               coordinate='actual_consumed_interactions', censoring_steps=2000000))
    datasets = [dict(role=role, scenario_count=count, manifest_path=f'datasets/{role}/manifest.json', manifest_sha256='c'*64)
                for role, count in p['role_counts'].items() if role != 'DEV']
    report = dict(status='complete', scientific_results_emitted=True, unique_evaluation_cases=65000,
                  logical_training_interactions=50000000, matrix_sha256='d'*64, contract_sha256='e'*64,
                  models=models, outcomes=outcomes, learning_curves=curves,
                  primary_contrasts=[dict(reward=r, reference='sparse', n_pairs=5, p_holm=None,
                                         reason='zero_variance_t_inference_undefined') for r in p['rewards'][1:]],
                  robustness_models=[dict(model_id=m['model_id'], passes=False) for m in models],
                  robustness_rewards=[dict(reward=r, passes=False) for r in p['rewards']],
                  limitations=['Synthetic integration fixture; not measured evidence.'])
    return report, datasets, dict(selected=[], absent=[])


class RecurrentExportTests(unittest.TestCase):
    def setUp(self):
        self.report, self.datasets, self.selection = producer_fixture()

    def convert(self):
        return build_transport(self.report, self.datasets, self.selection, study_id='synthetic_recurrent',
                               provenance={'fixture': True}, fixture=True)

    def test_complete_translation_preserves_authoritative_statistics_and_units(self):
        value = self.convert()
        parsed = adapt(value, 'research_results.json', 'f'*64)
        self.assertTrue(parsed['results']['inventory']['complete'])
        self.assertEqual(len(value['models']), 175)
        self.assertEqual(len(value['outcomes']), 150)
        self.assertEqual(len(value['learning']), 175)
        self.assertEqual(sum(r['n_completed'] for r in value['outcomes']+value['learning']), 65000)
        self.assertEqual(value['comparisons'], self.report['primary_contrasts'])
        self.assertEqual(value['producer_learning_metrics'], self.report['learning_curves'])
        self.assertEqual(value['producer_robustness_models'], self.report['robustness_models'])
        self.assertEqual(value['outcomes'][0]['mean_damage'], 2.5)
        self.assertEqual(value['outcomes'][0]['source_metrics']['raw_return_mean'], -3.2)
        self.assertNotIn('invalid_actions', value['outcomes'][0])  # Never round a mean into an invented count.
        self.assertEqual(value['learning'][1]['actual_training_steps'], 100123)

    def test_incomplete_or_wrong_role_panels_cannot_be_published_complete(self):
        self.report['learning_curves'][0]['points'][0]['role'] = 'DEV'
        self.datasets.append(dict(role='DEV', scenario_count=100, manifest_path='datasets/dev/manifest.json', manifest_sha256='c'*64))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.convert()

    def test_duplicate_or_wrong_counter_panels_fail(self):
        self.report['outcomes'][1] = copy.deepcopy(self.report['outcomes'][0])
        with self.assertRaises(ValueError): self.convert()
        self.setUp()
        self.report['learning_curves'][0]['points'][1]['actual_steps'] += 1
        with self.assertRaisesRegex(ValueError, 'counters'): self.convert()

    def test_incomplete_campaign_and_output_collision_leave_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root/'campaign', root/'export'
            write(source/'state/campaign.json', {'status': 'running'})
            before = byte_hash(source/'state/campaign.json')
            with self.assertRaisesRegex(ValueError, 'running or incomplete'):
                export_campaign(source, target, study_id='synthetic_recurrent')
            self.assertFalse(target.exists())
            self.assertEqual(byte_hash(source/'state/campaign.json'), before)
            target.mkdir()
            with self.assertRaisesRegex(ValueError, 'new export destination'):
                export_campaign(source, target, study_id='synthetic_recurrent')

    def test_results_only_export_survives_relocation_and_indexes_no_worlds(self):
        self.export_roundtrip(False)

    def test_replay_export_keeps_worlds_separate_and_loads_recorded_frames(self):
        self.export_roundtrip(True)

    def test_unicode_job_metadata_preserves_both_producer_hash_domains(self):
        job = {'description': 'terrain café'}
        utf8 = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        escaped = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
        self.assertEqual(digest(job), utf8)
        self.assertEqual(engine_digest(job), escaped)
        self.assertNotEqual(utf8, escaped)
        self.export_roundtrip(True, unicode_metadata=True)

    def export_roundtrip(self, include_replay, unicode_metadata=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root/'source', root/'release'
            source.mkdir()
            p = get_protocol(RECURRENT_PROTOCOL_ID)
            roles = {role: source/'data'/role.lower() for role in p['required_roles']}
            for role, directory in roles.items():
                write(directory/'manifest.json', {'entries': [{'scenario_id': f'synthetic_{i}'} for i in range(p['role_counts'][role])]})
            jobs = [dict(id=m['model_id'], train_dir=str(roles['TRAIN']), dev_dir=str(roles['LEARNING_VALIDATION']),
                         final_suites={role: str(directory) for role, directory in roles.items() if role.startswith('FINAL_')})
                    for m in self.report['models']]
            if unicode_metadata:
                for job in jobs: job['description'] = 'terrain café'
            write(source/'matrix.json', {'jobs': jobs})
            for model, job in zip(self.report['models'], jobs): model['job_sha256'] = digest(job)
            write(source/'contract.json', {'method': 'recurrent_2m_complete900'})
            self.report['matrix_sha256'] = byte_hash(source/'matrix.json')
            self.report['contract_sha256'] = byte_hash(source/'contract.json')
            write(source/'state/campaign.json', {'status': 'complete', 'seal_sha256': 'a'*64})
            if include_replay:
                from test_recurrent_replay import synthetic_documents, seal
                native = synthetic_documents()
                m = self.report['models'][0]
                cp = dict(m['checkpoints'][-1], model_seed=m['seed'])
                sid = native['snapshot']['scenario_id']
                write(roles['FINAL_ID']/'manifest.json', native['collection'])
                world_path = roles['FINAL_ID']/(sid+'.json')
                write(world_path, native['snapshot'])
                identity = native['case']['identity']
                identity.update(job_id=m['model_id'], job_sha256=engine_digest(jobs[0]), checkpoint=cp,
                                checkpoint_sha256=cp['sha256'], scenario_sha256=byte_hash(world_path))
                native['trajectory']['identity'] = copy.deepcopy(identity)
                seal(native['trajectory'])
                native['case']['metrics']['trace_sha256'] = native['trajectory']['sha256']
                seal(native['case'])
                case_rel = f'results/{m["model_id"]}/evaluation/FINAL_ID/2000000/{sid}.json'
                trace_rel = f'results/{m["model_id"]}/evaluation_traces/FINAL_ID/2000000/{sid}.json'
                write(source/case_rel, native['case'])
                write(source/trace_rel, native['trajectory'])
                self.selection['selected'].append(dict(job_id=m['model_id'], reward=m['reward'], seed=m['seed'],
                    checkpoint_step=2000000, checkpoint_sha256=cp['sha256'], scenario_id=sid, role='FINAL_ID',
                    selection_category='matched', case_path=case_rel, trajectory_path=trace_rel))
            write(source/'results/RESEARCH_RESULTS.json', self.report)
            write(source/'results/selected_traces.json', self.selection)
            write(source/'results/replay_verification/receipt.json', {'status': 'passed', 'selection_sha256': byte_hash(source/'results/selected_traces.json')})
            write(source/'results/scientific_case_index.json', {'count': 65000, 'cases': [{'path': f'results/synthetic/cases/{i}.json', 'sha256': 'f'*64} for i in range(65000)]})
            source_hashes = {p.relative_to(source).as_posix(): byte_hash(p) for p in source.rglob('*.json')}
            output = export_campaign(source, target, study_id='synthetic_recurrent', include_replay=include_replay)
            moved = root/'relocated'
            output.rename(moved)
            bundle = BundleIndex(moved/'synthetic_recurrent_results', releases_root=moved)
            loaded = load_bundle_export(bundle, 'research_results')
            self.assertTrue(loaded['results']['inventory']['complete'])
            self.assertEqual(loaded['results']['final_model_inventory'], 25)
            self.assertEqual((moved/'synthetic_recurrent_worlds').exists(), include_replay)
            self.assertEqual(any(e['storage'] == 'release' for e in bundle.entries.values()), include_replay)
            if include_replay:
                from research_support.dashboard_reference.readers.replay import load_selected
                self.assertEqual(len(load_selected(loaded, 0, bundle=bundle)['frames']), 4)
                no_worlds = BundleIndex(moved/'synthetic_recurrent_results')
                no_world_study = load_bundle_export(no_worlds, 'research_results')
                self.assertTrue(no_world_study['results']['inventory']['complete'])
                self.assertFalse(load_selected(no_world_study, 0, bundle=no_worlds)['available'])
            self.assertNotIn(str(source), (moved/'synthetic_recurrent_results/research_results.json').read_text())
            self.assertEqual({p.relative_to(source).as_posix(): byte_hash(p) for p in source.rglob('*.json')}, source_hashes)


if __name__ == '__main__':
    unittest.main()
