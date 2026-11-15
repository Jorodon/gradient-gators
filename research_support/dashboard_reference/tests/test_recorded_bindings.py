"""Default recorded examples; optional full release checks. No training or network.

DASHBOARD_RESULTS_BUNDLE adds full numerical checks (without requiring worlds).
DASHBOARD_RELEASES_ROOT adds world-backed replay joins. Small checks always run.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_support.dashboard_reference.readers.bundle_paths import BundleIndex
from research_support.dashboard_reference.readers.replay import load_recorded_binding

FIXTURE = Path(__file__).parent / 'fixtures/recorded_examples'
FULL_BUNDLE = Path(os.environ['DASHBOARD_RESULTS_BUNDLE']) if os.environ.get('DASHBOARD_RESULTS_BUNDLE') else None
RELEASES_ROOT = Path(os.environ['DASHBOARD_RELEASES_ROOT']) if os.environ.get('DASHBOARD_RELEASES_ROOT') else None


def assert_seal(test, document, field='sha256'):
    unsigned = dict(document)
    saved = unsigned.pop(field)
    actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    test.assertEqual(saved, actual)


class RecordedFixtureTests(unittest.TestCase):
    def test_fixture_is_explicit_subset_with_unchanged_sealed_records(self):
        bundle = BundleIndex(FIXTURE)
        bundle.verify_all(include_releases=False)
        scope = json.loads((FIXTURE / 'fixture_scope.json').read_text())
        self.assertFalse(scope['metric_aggregation_permitted'])
        self.assertNotIn('expected_results', bundle.entries)
        bindings = bundle.read_json('replay_bindings')['bindings']
        self.assertEqual(len(bindings), 3)
        self.assertEqual(sum(e['category'] == 'case' for e in bundle.entries.values()), 3)
        self.assertEqual(sum(e['category'] == 'trace' for e in bundle.entries.values()), 3)
        self.assertTrue(all(e['storage'] == 'release' for e in bundle.entries.values() if e['category'] == 'scenario_snapshot'))
        expected = {row['binding_id']: row for row in scope['selected_behavior_checks']}
        for binding in bindings:
            case = bundle.read_json(binding['case_artifact_id'])
            trace = bundle.read_json(binding['trace_artifact_id'])
            assert_seal(self, case)
            assert_seal(self, trace)
            self.assertEqual(case['identity'], trace['identity'])
            for key, value in expected[binding['binding_id']].items():
                if key != 'binding_id':
                    self.assertEqual(case['metrics'][key], value)
        self.assertEqual({row['timeout'] for row in expected.values()}, {False, True})
        self.assertTrue(any(row['illegal_moves'] > 0 for row in expected.values()))
        self.assertEqual(sum(row['wait'] for row in expected.values()), 1)

    def test_world_release_missing_is_unavailable_without_fallback(self):
        bundle = BundleIndex(FIXTURE)
        result = load_recorded_binding(bundle, bundle.read_json('replay_bindings')['bindings'][-1])
        self.assertFalse(result['available'])
        self.assertIn('Release unavailable', result['reason_unavailable'])

    @unittest.skipUnless(RELEASES_ROOT, 'Set DASHBOARD_RELEASES_ROOT explicitly for recorded-world verification')
    def test_selected_replays_and_wait_separation(self):
        bundle = BundleIndex(FIXTURE, releases_root=RELEASES_ROOT)
        bindings = bundle.read_json('replay_bindings')['bindings']
        for binding in bindings:
            result = load_recorded_binding(bundle, binding)
            self.assertTrue(result['available'], result.get('reason_unavailable'))
            self.assertIsNone(result['frames'][0]['action'])
            self.assertEqual([frame['step'] for frame in result['frames']], list(range(len(result['frames']))))
        wait = load_recorded_binding(bundle, next(b for b in bindings if b['evidence_scope'] == 'supplemental_wait'))
        self.assertEqual(sum(bool(f['events'] and f['events']['waited']) for f in wait['frames']), 1)
        self.assertEqual(sum(bool(f['events'] and f['events']['invalid_move']) for f in wait['frames']), 3)
        self.assertEqual(wait['outcome']['end_reason'], 'goal')

    @unittest.skipUnless(RELEASES_ROOT, 'Set DASHBOARD_RELEASES_ROOT explicitly for recorded-world verification')
    def test_join_tamper_and_final_role_rejected(self):
        bundle = BundleIndex(FIXTURE, releases_root=RELEASES_ROOT)
        binding = bundle.read_json('replay_bindings')['bindings'][-1]
        for key, value in (('source_job_sha256', '0'*64), ('checkpoint_sha256', '0'*64), ('evaluation_seed', 0), ('role', 'FINAL_ID')):
            changed = copy.deepcopy(binding)
            changed[key] = value
            self.assertFalse(load_recorded_binding(bundle, changed)['available'], key)
        original = bundle.read_json
        def wrong_trace(artifact):
            value = original(artifact)
            if artifact == binding['trace_artifact_id']:
                value['steps'][0]['hp'] += 1
            return value
        with patch.object(bundle, 'read_json', side_effect=wrong_trace):
            result = load_recorded_binding(bundle, binding)
            self.assertFalse(result['available'])
            self.assertIn('seal mismatch', result['reason_unavailable'])


@unittest.skipUnless(FULL_BUNDLE, 'Set DASHBOARD_RESULTS_BUNDLE for optional full-corpus validation')
class FullRecordedReleaseTests(unittest.TestCase):
    def test_full_case_summaries_and_learning_curves(self):
        bundle = BundleIndex(FULL_BUNDLE)
        bundle.verify_all(include_releases=False)
        self.assertFalse(bundle.manifest.get('fixture', False))
        expected = bundle.read_json('expected_results')['baseline']
        self.assertEqual(expected['cases'], 3500)
        self.assertEqual(expected['policies'], 5)
        case_entries = {key: entry for key, entry in bundle.entries.items() if entry['category'] == 'case'}
        self.assertEqual(len(case_entries), 3501)
        grouped = {}
        for artifact_id in case_entries:
            case = bundle.read_json(artifact_id)
            assert_seal(self, case)
            self.assertEqual(case['status'], 'complete')
            identity = case['identity']
            group = grouped.setdefault(identity['job_id'], {})
            panel = identity['role'] + '_' + str(identity['checkpoint']['nominal_training_steps'])
            group.setdefault(panel, []).append(case)
        panel_count = curve_count = 0
        for job in expected['jobs']:
            panels = grouped[job['job_id']]
            result = bundle.read_json('entropy_0_01_experiment_' + job['reward'] + '_result')
            self.assertEqual(sum(len(rows) for rows in panels.values()), 700)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(set(panels), set(job['panels']))
            for panel, rows in panels.items():
                recorded = result['metrics'][panel]
                oracle = job['panels'][panel]
                count = len(rows)
                self.assertEqual(count, 100 if panel.startswith('DEV_') else 300)
                self.assertEqual(count, oracle['completed'])
                self.assertEqual(len({r['identity']['scenario_id'] for r in rows}), count)
                metrics = [r['metrics'] for r in rows]
                recomputed = {'episodes': count, 'steps': sum(r['length'] for r in metrics)}
                for field in ('success', 'zero_damage_success', 'death', 'timeout', 'damage', 'wait', 'illegal_moves', 'raw_return', 'length'):
                    total = math.fsum(r[field] for r in metrics)
                    recomputed[field+'_sum'] = total
                    recomputed[field+'_mean'] = total/count
                self.assertEqual(set(recomputed), set(recorded))
                for key, value in recomputed.items():
                    self.assertTrue(math.isclose(value, recorded[key], rel_tol=1e-10, abs_tol=1e-8), (panel, key))
                    self.assertTrue(math.isclose(value, oracle['recomputed'][key], rel_tol=1e-10, abs_tol=1e-8), (panel, key))
                for field, value in (
                    ('pooled_invalid_fraction', recomputed['illegal_moves_sum']/recomputed['steps']),
                    ('episode_mean_invalid_fraction', math.fsum(r['illegal_moves']/r['length'] for r in metrics)/count),
                    ('pooled_wait_fraction', recomputed['wait_sum']/recomputed['steps'])):
                    self.assertTrue(math.isclose(value, oracle[field], rel_tol=1e-10, abs_tol=1e-8), (panel, field))
                panel_count += 1
            grid = job['checkpoint_grid']
            self.assertEqual(grid, [0, 100000, 250000, 500000])
            counts = [sum(r['metrics']['success'] for r in panels['DEV_'+str(step)]) for step in grid]
            self.assertEqual(counts, job['learning']['success_counts'])
            self.assertEqual(job['learning']['denominators'], [100]*4)
            aulc = sum((grid[i+1]-grid[i])*(counts[i]+counts[i+1])/200 for i in range(len(grid)-1))/grid[-1]
            self.assertAlmostEqual(aulc, job['learning']['normalized_aulc_fraction'], places=10)
            curve_count += len(grid)
        self.assertEqual(panel_count, 25)
        self.assertEqual(curve_count, 20)
        supplemental = grouped['anchor_confirmation_s2_recurrent_progress']
        self.assertEqual(sum(len(rows) for rows in supplemental.values()), 1)

    @unittest.skipUnless(RELEASES_ROOT, 'Set DASHBOARD_RELEASES_ROOT for all 36 world-backed replay joins')
    def test_full_replay_joins(self):
        bundle = BundleIndex(FULL_BUNDLE, releases_root=RELEASES_ROOT)
        bundle.verify_all()
        bindings = bundle.read_json('replay_bindings')['bindings']
        self.assertEqual(len(bindings), 36)
        self.assertEqual(sum(b['evidence_scope'] == 'baseline' for b in bindings), 35)
        self.assertEqual(sum(b['evidence_scope'] == 'supplemental_wait' for b in bindings), 1)
        for binding in bindings:
            result = load_recorded_binding(bundle, binding)
            self.assertTrue(result['available'], (binding['binding_id'], result.get('reason_unavailable')))


if __name__ == '__main__':
    unittest.main()
