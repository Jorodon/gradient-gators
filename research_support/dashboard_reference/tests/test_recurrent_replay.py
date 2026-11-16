"""Invented native replay bytes; no policies, scenarios or outcomes are executed."""
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
from research_support.dashboard_reference.readers.bundle_paths import BundleIndex, SCHEMA_VERSION
from research_support.dashboard_reference.readers.recurrent_replay import digest
from research_support.dashboard_reference.readers.replay import load_selected
from research_support.dashboard_reference.readers.results_adapter import load_bundle_export


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def seal(value, field='sha256', *, world=False):
    value.pop(field, None)
    value[field] = digest(value, world=world)
    return value


def synthetic_documents(*, death=False, enemy=False):
    """Three stored actions: WAIT, invalid move, then goal contact."""
    seed = get_protocol(RECURRENT_PROTOCOL_ID)['primary_seeds'][0]
    grid = dict(width=2, height=1, start=[0, 0], goal=[1, 0], tiles=[[
        dict(elevation=0, obstacle=False, hazard=None, special_traversal=False) for _ in range(2)]])
    enemy_spec = dict(patrol_route=[[1, 0]], initial_patrol_index=0, behavior='patrol_v1') if enemy else None
    world_input = dict(map=grid, enemy_spec=enemy_spec, environment_config={'max_hp': 100}, horizon=8)
    scenario = 'scn_' + digest(world_input, world=True)
    map_id = 'map_' + digest(grid, world=True)
    world = seal(dict(schema_version=2, scenario_id=scenario, **world_input,
                      provenance={'collection_id': 'synthetic_final', 'split': 'FINAL_ID'}, witness={}),
                 'checksum', world=True)
    members = [dict(scenario_id=scenario, map_id=map_id, checksum=world['checksum'], area_band='synthetic')]
    members += [dict(scenario_id=f'synthetic-member-{i}', map_id=f'synthetic-map-{i}',
                     checksum='c' * 64, area_band='synthetic') for i in range(199)]
    collection = seal(dict(schema_version=1, collection_id='synthetic_final', split='FINAL_ID',
                           code_version='synthetic', entries=members), 'checksum', world=True)
    checkpoint = dict(nominal_training_steps=2000000, actual_training_steps=2000000,
                      optimized_training_steps=1999900, model_seed=seed, sha256='d' * 64)
    identity = dict(job_id='synthetic_job', job_sha256='e' * 64, checkpoint=checkpoint,
                    checkpoint_sha256='d' * 64, algorithm={'algorithm': 'recurrent_ppo'},
                    reward={'family': 'sparse'}, mode='deterministic', trained_on_scenario=False,
                    engine_semantics='recurrent-primary-v1-frozen-C2-modelseed-per-slot',
                    seed=seed, role='FINAL_ID', scenario_id=scenario,
                    scenario_sha256=hashlib.sha256(encode(world)).hexdigest(), scenario_checksum=world['checksum'],
                    collection_checksum=collection['checksum'], map_id=map_id, trace_requested=True)
    initial = dict(steps=0, agent_position=[0, 0], agent_hp=100, enemy_position=None, enemy_patrol_index=None,
                   scenario_id=scenario, snapshot_checksum=world['checksum'], max_episode_steps=8,
                   collection_id='synthetic_final', split='FINAL_ID')
    if enemy:
        initial.update(enemy_position=[1, 0], enemy_patrol_index=0, enemy_behavior='patrol_v1')
    steps = []
    for action, invalid, terminal in ((4, False, False), (3, True, False), (1, False, True)):
        damage = 100 if death and terminal else 0
        events = dict(waited=action == 4, invalid_move=invalid, reached_goal=terminal,
                      hazard_contact=damage > 0, enemy_contact=False, damage_taken=damage,
                      fall_damage=0, hazard_damage=damage, enemy_damage=0)
        steps.append(dict(action=action, reward=-1 if not terminal else 2,
                          terminated=terminal, truncated=False, events=events,
                          position=[1, 0] if terminal else [0, 0], hp=100-damage,
                          enemy_position=[1, 0] if enemy else None, enemy_patrol_index=0 if enemy else None))
    trace = seal(dict(identity=copy.deepcopy(identity), initial_state=initial, steps=steps))
    metrics = dict(scenario_id=scenario, length=3, success=not death, death=death, timeout=False,
                   zero_damage_success=not death, terminated=True, truncated=False,
                   final_hp=0 if death else 100, damage=100 if death else 0, fall_damage=0,
                   hazard_damage=100 if death else 0, enemy_damage=0, wait=1, illegal_moves=1,
                   hazard_contacts=int(death), enemy_contacts=0, invalid_move_rate=1/3,
                   raw_return=0, trace_sha256=trace['sha256'], trajectory_sha256=digest(steps))
    case = seal(dict(status='complete', identity=identity, metrics=metrics))
    model = dict(model_id='synthetic_job__t2000000', source_model_id='synthetic_job',
                 source_job_sha256='e' * 64, checkpoint_sha256='d' * 64, reward='sparse', seed=seed, slot=0,
                 checkpoint_step=2000000, actual_training_steps=2000000, optimized_training_steps=1999900)
    selected = dict(model_id=model['model_id'], role='FINAL_ID', scenario_id=scenario,
                    selection_category='matched', available=True, replay_format='recurrent-primary-v1',
                    source_job_id='synthetic_job', source_job_sha256='e' * 64, checkpoint_sha256='d' * 64,
                    evaluation_seed=seed, case_artifact_id='case', trajectory_artifact_id='trajectory',
                    snapshot_artifact_id='snapshot', collection_artifact_id='collection')
    document = dict(schema_version='research-results-v2', protocol_id=RECURRENT_PROTOCOL_ID,
                    study_id='synthetic-native-replay', fixture=True, provenance={'source': 'invented test records'},
                    models=[model], datasets=[dict(role='FINAL_ID', scenario_count=200,
                        manifest_path='collection.json', manifest_sha256=hashlib.sha256(encode(collection)).hexdigest())],
                    outcomes=[], learning=[], comparisons=[], traces=[selected],
                    limitations=['Synthetic replay fixture; no scientific result.'])
    return dict(research_results=document, case=case, trajectory=trace, snapshot=world, collection=collection)


def write_bundle(root, documents, *, release_world=False):
    files = []
    for name, value in documents.items():
        raw = encode(value)
        relative = name + '.json'
        path = root / relative
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        entry = dict(artifact_id=name, storage='release' if release_world and name == 'snapshot' else 'repository',
                     relative_path=relative, category=name, bytes=len(raw), sha256=sha, source_byte_identical=True,
                     source_ref={'experiment_id': 'synthetic', 'artifact_path': relative, 'sha256': sha})
        if entry['storage'] == 'release':
            entry['release_id'] = 'synthetic_worlds'
        files.append(entry)
    (root / 'bundle_manifest.json').write_bytes(encode(dict(schema_version=SCHEMA_VERSION,
                                                          bundle_id='synthetic_replay', fixture=True, files=files)))
    return BundleIndex(root)


def reseal_case_and_trace(documents):
    trace = documents['trajectory']
    seal(trace)
    documents['case']['metrics'].update(trace_sha256=trace['sha256'], trajectory_sha256=digest(trace['steps']))
    seal(documents['case'])


class NativeReplayTests(unittest.TestCase):
    def load(self, documents, *, release_world=False):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_bundle(Path(directory), documents, release_world=release_world)
            study = load_bundle_export(bundle, 'research_results')
            return load_selected(study, 0, bundle=bundle)

    def test_wait_invalid_and_player_frame_shape(self):
        documents = synthetic_documents()
        before = copy.deepcopy(documents)
        result = self.load(documents)
        self.assertTrue(result['available'], result.get('reason_unavailable'))
        self.assertEqual(documents, before)
        frames = result['frames']
        self.assertEqual(len(frames), 4)
        self.assertEqual([f['step'] for f in frames], [0, 1, 2, 3])
        self.assertIsNone(frames[0]['action'])
        self.assertEqual(frames[0]['agent_position'], [0, 0])
        self.assertTrue(frames[1]['events']['waited'])
        self.assertFalse(frames[1]['events']['invalid_move'])
        self.assertTrue(frames[2]['events']['invalid_move'])
        self.assertEqual(frames[-1]['agent_position'], [1, 0])
        self.assertEqual(result['outcome']['end_reason'], 'goal')
        self.assertEqual(set(frames[-1]), {'step', 'action', 'reward', 'events', 'agent_position', 'agent_hp',
                                        'enemy_position', 'enemy_patrol_index', 'terminated', 'truncated', 'end_reason'})
        self.assertEqual(result['verification']['status'], 'recorded_artifact_joins_checked')
        self.assertTrue(result['fixture'])

    def test_fatal_goal_contact_stays_death(self):
        result = self.load(synthetic_documents(death=True))
        self.assertTrue(result['available'], result.get('reason_unavailable'))
        self.assertTrue(result['frames'][-1]['events']['reached_goal'])
        self.assertEqual(result['outcome']['final_position'], [1, 0])
        self.assertEqual(result['outcome']['end_reason'], 'death')
        self.assertFalse(result['outcome']['success'])

    def test_enemy_reset_joins_world_and_absent_enemy_keys_stay_null(self):
        d = synthetic_documents(enemy=True)
        result = self.load(d)
        self.assertTrue(result['available'], result.get('reason_unavailable'))
        self.assertEqual(result['frames'][0]['enemy_position'], [1, 0])
        d['trajectory']['initial_state']['enemy_position'] = [0, 0]
        reseal_case_and_trace(d)
        self.assertIn('Reset enemy/world mismatch', self.load(d)['reason_unavailable'])
        d = synthetic_documents()
        for field in ('enemy_position', 'enemy_patrol_index'):
            del d['trajectory']['initial_state'][field]
        reseal_case_and_trace(d)
        result = self.load(d)
        self.assertTrue(result['available'], result.get('reason_unavailable'))
        self.assertIsNone(result['frames'][0]['enemy_position'])

    def test_missing_world_requires_explicit_release_without_host_fallback(self):
        result = self.load(synthetic_documents(), release_world=True)
        self.assertFalse(result['available'])
        self.assertIn('Release unavailable', result['reason_unavailable'])

    def test_export_hash_and_in_memory_binding_cannot_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_bundle(Path(directory), synthetic_documents())
            study = load_bundle_export(bundle, 'research_results')
            self.assertFalse(load_selected(study, 0)['available'])
            changed = copy.deepcopy(study)
            changed['results']['traces'][0]['evaluation_seed'] += 1
            result = load_selected(changed, 0, bundle=bundle)
            self.assertIn('binding differs', result['reason_unavailable'])
            study['results']['export_sha256'] = '0' * 64
            result = load_selected(study, 0, bundle=bundle)
            self.assertIn('source hash', result['reason_unavailable'])

    def test_unsealed_changes_are_rejected(self):
        d = synthetic_documents()
        d['trajectory']['steps'][0]['hp'] -= 1
        result = self.load(d)
        self.assertFalse(result['available'])
        self.assertIn('seal mismatch', result['reason_unavailable'])

    def test_resealed_semantic_mismatches_are_rejected(self):
        for defect in ('job', 'checkpoint', 'seed', 'algorithm', 'role', 'world_hash', 'collection',
                       'reset', 'wait_invalid', 'return_total', 'damage_total', 'fatal_success', 'extra_action'):
            with self.subTest(defect=defect):
                d = synthetic_documents()
                case, trace = d['case'], d['trajectory']
                identity = case['identity']
                if defect == 'job': identity['job_sha256'] = '0' * 64
                elif defect == 'checkpoint': identity['checkpoint']['optimized_training_steps'] -= 1
                elif defect == 'seed': identity['seed'] += 1
                elif defect == 'algorithm': identity['algorithm']['algorithm'] = 'ppo'
                elif defect == 'role': identity['role'] = 'FINAL_OOD_HAZARD'
                elif defect == 'world_hash': identity['scenario_sha256'] = '0' * 64
                elif defect == 'collection': identity['collection_checksum'] = '0' * 64
                elif defect == 'reset': trace['initial_state']['agent_position'] = [1, 0]
                elif defect == 'wait_invalid': trace['steps'][0]['events']['invalid_move'] = True
                elif defect == 'return_total': case['metrics']['raw_return'] += 1
                elif defect == 'damage_total': case['metrics']['damage'] += 1
                elif defect == 'fatal_success':
                    trace['steps'][-1]['hp'] = case['metrics']['final_hp'] = 0
                elif defect == 'extra_action':
                    trace['steps'][0]['terminated'] = True
                trace['identity'] = copy.deepcopy(identity)
                reseal_case_and_trace(d)
                result = self.load(d)
                self.assertFalse(result['available'], defect)

    def test_production_collection_count_cannot_use_fixture_exception(self):
        d = synthetic_documents()
        d['research_results']['fixture'] = False
        d['collection']['entries'].pop()
        seal(d['collection'], 'checksum', world=True)
        d['case']['identity']['collection_checksum'] = d['collection']['checksum']
        d['trajectory']['identity'] = copy.deepcopy(d['case']['identity'])
        reseal_case_and_trace(d)
        d['research_results']['datasets'][0]['manifest_sha256'] = hashlib.sha256(encode(d['collection'])).hexdigest()
        result = self.load(d)
        self.assertFalse(result['available'])
        self.assertIn('200 scenarios', result['reason_unavailable'])


if __name__ == '__main__':
    unittest.main()
