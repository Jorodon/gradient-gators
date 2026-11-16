"""Synthetic saved-result protocol, counter, inventory and compatibility checks."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_support.dashboard_reference.protocols import (
    FINAL_ROLES, LEGACY_PROTOCOL_ID, RECURRENT_PROTOCOL_ID, get_protocol,
    validate_training_steps,
)
from research_support.dashboard_reference.readers.results_adapter import adapt


def recurrent_fixture(*, full=False, fixture=True):
    """Invented aggregates only; no simulator or policy execution."""
    protocol = get_protocol(RECURRENT_PROTOCOL_ID)
    datasets = [dict(role=role, scenario_count=2 if fixture else count,
                     manifest_path=f'{role}.json', manifest_sha256=None if fixture else 'a' * 64)
                for role, count in protocol['role_counts'].items() if role != 'DEV']
    models, outcomes, learning = [], [], []
    rewards = protocol['rewards'] if full else protocol['rewards'][:1]
    seeds = protocol['primary_seeds'] if full else protocol['primary_seeds'][:1]
    for reward in rewards:
        for slot, seed in enumerate(seeds):
            for step in protocol['grid']:
                actual = step if step in (0, protocol['budget']) else step + 7
                model = dict(model_id=f'{reward}-{slot}-{step}', reward=reward, seed=seed,
                             slot=slot, checkpoint_step=step, actual_training_steps=actual,
                             optimized_training_steps=actual)
                models.append(model)
                for dataset in datasets:
                    role = dataset['role']
                    if role != 'LEARNING_VALIDATION' and step != protocol['budget']:
                        continue
                    count = dataset['scenario_count']
                    row = dict(model_id=model['model_id'], role=role, n_expected=count,
                               n_completed=count, success_count=1, death_count=0,
                               timeout_count=count - 1)
                    outcomes.append(row)
                    if role == 'LEARNING_VALIDATION':
                        learning.append(dict(row, checkpoint_step=step,
                                             actual_training_steps=actual,
                                             optimized_training_steps=actual))
    return dict(schema_version='research-results-v2', protocol_id=RECURRENT_PROTOCOL_ID,
                study_id='synthetic-recurrent-contract', fixture=fixture, status='completed',
                provenance={'source': 'Invented test aggregates'}, datasets=datasets,
                models=models, outcomes=outcomes, learning=learning, comparisons=[], traces=[],
                limitations=['SYNTHETIC FIXTURE ONLY; no policy or research result.'])


def parse(document):
    return adapt(document, 'synthetic-results.json', 'b' * 64)


class ProtocolSelectionTests(unittest.TestCase):
    def test_historical_default_and_immutable_protocol_copies(self):
        self.assertEqual(get_protocol()['id'], LEGACY_PROTOCOL_ID)
        self.assertEqual(get_protocol()['grid'], (0, 250000, 500000, 750000, 1000000))
        copy_of_spec = get_protocol(RECURRENT_PROTOCOL_ID)
        copy_of_spec['role_counts']['TRAIN'] = 1
        self.assertEqual(get_protocol(RECURRENT_PROTOCOL_ID)['role_counts']['TRAIN'], 900)
        self.assertEqual(get_protocol()['role_counts']['TRAIN'], 300)

    def test_legacy_examples_and_alias_keep_original_coordinates(self):
        document = json.loads((CODE / 'examples' / 'research_results.json').read_text())
        expected = parse(document)
        for version in ('research-results-v1', 'm4-results-v1'):
            d = copy.deepcopy(document)
            d['schema_version'] = version
            d.pop('protocol_id')
            result = parse(d)
            self.assertEqual(result['protocol'], LEGACY_PROTOCOL_ID)
            self.assertEqual(result['records'], expected['records'])
            self.assertEqual(result['results']['training_budget'], 1000000)

    def test_reject_unknown_missing_and_cross_version_protocols(self):
        for version, protocol in (
            ('research-results-v2', None), ('research-results-v2', LEGACY_PROTOCOL_ID),
            ('research-results-v1', RECURRENT_PROTOCOL_ID), ('m4-results-v1', RECURRENT_PROTOCOL_ID),
            ('research-results-v3', RECURRENT_PROTOCOL_ID), ('research-results-v2', 'guessed-from-name'),
        ):
            with self.subTest(version=version, protocol=protocol):
                d = recurrent_fixture()
                d['schema_version'] = version
                if protocol is None:
                    d.pop('protocol_id')
                else:
                    d['protocol_id'] = protocol
                with self.assertRaises(ValueError):
                    parse(d)


class TrainingCoordinateTests(unittest.TestCase):
    def test_nominal_and_actual_coordinates_survive_adaptation(self):
        result = parse(recurrent_fixture())
        record = next(r for r in result['records'] if r['checkpoint'] == 100000)
        self.assertEqual(record['actual_training_steps'], 100007)
        self.assertEqual(record['optimized_training_steps'], 100007)
        self.assertEqual(result['results']['checkpoint_grid'], [0, 100000, 250000, 500000, 1000000, 1500000, 2000000])
        self.assertEqual(result['results']['final_model_inventory'], 1)

    def test_overshoot_boundary_and_final_unoptimized_reconciliation(self):
        row = dict(checkpoint_step=100000, actual_training_steps=101398,
                   optimized_training_steps=100000, final_unoptimized_steps=1398)
        validate_training_steps(row, RECURRENT_PROTOCOL_ID)
        for changes in (
            {'actual_training_steps': 101399}, {'actual_training_steps': 99999},
            {'optimized_training_steps': 101399}, {'final_unoptimized_steps': 1397},
            {'actual_training_steps': True}, {'optimized_training_steps': -1},
            {'checkpoint_step': 750000}, {'checkpoint_step': 0},
            {'checkpoint_step': 2000000, 'actual_training_steps': 2000001},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_training_steps(dict(row, **changes), RECURRENT_PROTOCOL_ID)

    def test_counter_presence_and_learning_model_join(self):
        for field in ('actual_training_steps', 'optimized_training_steps'):
            d = recurrent_fixture()
            del d['models'][1][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse(d)
        d = recurrent_fixture()
        d['learning'][1]['actual_training_steps'] += 1
        with self.assertRaisesRegex(ValueError, 'join mismatch'):
            parse(d)


class RecurrentInventoryTests(unittest.TestCase):
    def test_incomplete_matrix_cannot_keep_completed_status(self):
        result = parse(recurrent_fixture())
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['metadata']['reported_status'], 'completed')
        inventory = result['results']['inventory']
        self.assertFalse(inventory['complete'])
        self.assertEqual(inventory['expected_models'], 175)
        self.assertEqual(len(inventory['missing_models']), 168)
        self.assertEqual(inventory['scientific_validation'], 'not_assessed')

    def test_complete_inventory_with_optional_dev_absent(self):
        result = parse(recurrent_fixture(full=True))
        self.assertTrue(result['results']['inventory']['complete'])
        self.assertEqual(result['results']['inventory']['missing_dataset_roles'], [])
        self.assertEqual(result['results']['final_model_inventory'], 25)
        self.assertEqual(result['results']['inventory']['scientific_validation'], 'not_assessed')

    def test_missing_learning_final_rows_and_roles_are_explicit_gaps(self):
        d = recurrent_fixture(full=True)
        missing = next(r for r in d['outcomes'] if r['role'] == 'FINAL_ID')
        d['outcomes'].remove(missing)
        d['learning'] = []
        d['expected_evaluations'] = []
        result = parse(d)
        self.assertEqual(result['status'], 'partial')
        record = next(r for r in result['records'] if r['id'] == missing['model_id'] + ':FINAL_ID')
        self.assertEqual(record['data_status'], 'unavailable')
        self.assertIsNone(record['successes'])
        self.assertEqual(len(result['results']['inventory']['missing_evaluations']), 176)
        d = recurrent_fixture(full=True)
        d['datasets'] = [r for r in d['datasets'] if r['role'] != 'FINAL_OOD_HAZARD']
        d['outcomes'] = [r for r in d['outcomes'] if r['role'] != 'FINAL_OOD_HAZARD']
        result = parse(d)
        self.assertEqual(result['results']['inventory']['missing_dataset_roles'], ['FINAL_OOD_HAZARD'])
        self.assertEqual(result['status'], 'partial')

    def test_production_denominators_and_declared_dimensions(self):
        parse(recurrent_fixture(fixture=False))
        defects = ('train_count', 'reward', 'seed', 'slot', 'conflicting_slot', 'off_grid', 'final_at_1m', 'join', 'complete')
        for defect in defects:
            with self.subTest(defect=defect):
                d = recurrent_fixture(fixture=False)
                if defect == 'train_count':
                    d['datasets'][0]['scenario_count'] = 300
                elif defect == 'reward':
                    d['models'][0]['reward'] = 'inferred-recurrent'
                elif defect == 'seed':
                    d['models'][0]['seed'] = 0
                elif defect == 'slot':
                    d['models'][0]['slot'] = 1
                elif defect == 'conflicting_slot':
                    d['models'][0]['seed_slot'] = 1
                elif defect == 'off_grid':
                    d['models'][0]['checkpoint_step'] = 750000
                elif defect == 'final_at_1m':
                    next(r for r in d['outcomes'] if r['role'] == 'FINAL_ID')['model_id'] = d['models'][4]['model_id']
                elif defect == 'join':
                    d['outcomes'][0]['model_id'] = 'not-declared'
                elif defect == 'complete':
                    d['outcomes'][0]['complete'] = False
                with self.assertRaises(ValueError):
                    parse(d)

    def test_matched_trace_requires_explicit_slot_zero_at_2m(self):
        d = recurrent_fixture()
        model = d['models'][-1]
        d['traces'] = [dict(selection_category='matched', model_id=model['model_id'],
                            role='FINAL_ID', scenario_id='synthetic-case', available=False)]
        result = parse(d)
        self.assertEqual(result['results']['traces'][0]['checkpoint'], 2000000)
        model.pop('slot')
        with self.assertRaisesRegex(ValueError, 'explicit primary slot 0'):
            parse(d)
        model['seed_slot'] = 0
        parse(d)
        d['traces'][0]['model_id'] = d['models'][4]['model_id']
        with self.assertRaisesRegex(ValueError, '2000000 checkpoint'):
            parse(d)


if __name__ == '__main__':
    unittest.main()
