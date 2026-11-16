"""Validate and display saved recurrent-primary traces without model execution."""
import hashlib
import json
import math

from ..protocols import FINAL_ROLES, RECURRENT_PROTOCOL_ID, get_protocol, validate_training_steps
from .bundle_paths import BundleIndex, strict_json
from .results_adapter import adapt, integer, require, sha256


def digest(value, *, world=False):
    # Native engine seals use escaped Unicode; world serialization preserves it.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=not world, allow_nan=False).encode()).hexdigest()


def sealed(value, field, *, world=False):
    require(isinstance(value, dict), 'Sealed record must be an object')
    unsigned = dict(value)
    seal = unsigned.pop(field, None)
    require(seal == digest(unsigned, world=world), 'Recorded content seal mismatch')
    return seal


def number(value, label, *, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), f'Invalid {label}')
    require(minimum is None or value >= minimum, f'Invalid {label}')
    return value


def load_recurrent_selected(study, index, *, bundle=None):
    """Return verified stored frames, or explicit unavailability with no fallback."""
    selected = study['results']['traces'][index]
    result = dict(available=False, selection=selected)
    try:
        require(selected.get('available') is True, selected.get('reason_unavailable') or 'Trace was not saved')
        require(isinstance(bundle, BundleIndex), 'An explicit BundleIndex is required for recurrent replay')
        transport = study['results']
        require(transport.get('export_artifact_id'), 'A bundle-indexed export is required for recurrent replay')
        raw = bundle.read_bytes(transport['export_artifact_id'])
        require(hashlib.sha256(raw).hexdigest() == transport['export_sha256'], 'Export source hash changed')
        document = strict_json(raw.decode('utf-8-sig'))
        original = adapt(document, 'artifact:' + transport['export_artifact_id'], transport['export_sha256'])
        require(original['results']['traces'][index] == selected, 'Selected binding differs from its saved export')
        require(original['protocol'] == study['protocol'] == RECURRENT_PROTOCOL_ID,
                'Recurrent replay requires the recurrent scientific protocol')
        fixture = original['fixture']
        protocol = get_protocol(RECURRENT_PROTOCOL_ID)
        require(selected['role'] in FINAL_ROLES and selected['checkpoint'] == protocol['budget'],
                'Recurrent selected replay requires a FINAL role at 2M')
        if selected['selection_category'] == 'matched':
            require(type(selected.get('slot')) is int and selected['slot'] == 0,
                    'Matched recurrent replay requires primary slot 0')
        model = next(m for m in document['models'] if m['model_id'] == selected['model_id'])
        validate_training_steps(model, RECURRENT_PROTOCOL_ID)
        require(model['source_model_id'] == selected['source_job_id'] and
                model['source_job_sha256'] == selected['source_job_sha256'], 'Export model/job identity mismatch')
        for field in ('source_job_sha256', 'checkpoint_sha256'):
            sha256(selected.get(field), field, False)
        require(model['checkpoint_sha256'] == selected['checkpoint_sha256'], 'Export checkpoint hash mismatch')

        names = ('case', 'trajectory', 'snapshot', 'collection')
        ids = {name: selected[name + '_artifact_id'] for name in names}
        values, hashes = {}, {}
        for name, artifact in ids.items():
            saved = bundle.read_bytes(artifact)
            hashes[name] = hashlib.sha256(saved).hexdigest()
            if selected.get(name + '_sha256') is not None:
                require(hashes[name] == selected[name + '_sha256'], 'Selected artifact hash differs from export')
            values[name] = strict_json(saved.decode('utf-8-sig'))
        case, trace, world, collection = (values[name] for name in names)
        sealed(case, 'sha256')
        trace_seal = sealed(trace, 'sha256')
        world_seal = sealed(world, 'checksum', world=True)
        collection_seal = sealed(collection, 'checksum', world=True)
        require(case['status'] == 'complete', 'Recorded case is incomplete')
        identity, metrics = case['identity'], case['metrics']
        initial, steps = trace['initial_state'], trace['steps']
        require(identity == trace['identity'], 'Case/trace identity mismatch')
        require(identity['trace_requested'] is True, 'Case did not request a trace')
        require(identity['job_id'] == selected['source_job_id'] and
                identity['job_sha256'] == selected['source_job_sha256'], 'Original job identity mismatch')
        require(identity['algorithm']['algorithm'] == 'recurrent_ppo' and
                identity['reward']['family'] == selected['reward'], 'Algorithm/reward identity mismatch')
        require(identity['mode'] == 'deterministic' and identity['trained_on_scenario'] is False,
                'Final replay requires deterministic evaluation of an untrained scenario')
        require(identity['engine_semantics'] == 'recurrent-primary-v1-frozen-C2-modelseed-per-slot',
                'Unsupported recurrent engine semantics')
        checkpoint = identity['checkpoint']
        require(checkpoint['sha256'] == identity['checkpoint_sha256'] == selected['checkpoint_sha256'],
                'Checkpoint hash mismatch')
        for native, exported in (('nominal_training_steps', 'checkpoint_step'),
                                 ('actual_training_steps', 'actual_training_steps'),
                                 ('optimized_training_steps', 'optimized_training_steps'), ('model_seed', 'seed')):
            require(type(checkpoint[native]) is int and checkpoint[native] == model[exported],
                    'Checkpoint counter/seed identity mismatch')
        require(type(identity['seed']) is int and identity['seed'] == selected['evaluation_seed'] == model['seed'] and
                identity['role'] == selected['role'], 'Role/evaluation-seed identity mismatch')
        scenario = selected['scenario_id']
        require(all(value == scenario for value in (identity['scenario_id'], world['scenario_id'],
                    initial['scenario_id'], metrics['scenario_id'])), 'Scenario identity mismatch')
        require(identity['scenario_sha256'] == hashes['snapshot'], 'World file hash mismatch')
        require(identity['scenario_checksum'] == initial['snapshot_checksum'] == world_seal,
                'World checksum mismatch')
        require(identity['collection_checksum'] == collection_seal, 'Collection checksum mismatch')
        require(type(collection['schema_version']) is int and collection['schema_version'] == 1,
                'Unsupported collection serialization version')
        require(collection['split'] == initial['split'] == world['provenance']['split'] == selected['role'] and
                collection['collection_id'] == initial['collection_id'] == world['provenance']['collection_id'],
                'Collection/reset/world role identity mismatch')
        entries = collection['entries']
        require(isinstance(entries, list) and (1 <= len(entries) <= 200 if fixture else len(entries) == 200),
                'Final collection requires 200 scenarios (small counts only in fixtures)')
        members = {entry['scenario_id']: entry for entry in entries}
        require(len(members) == len(entries) and scenario in members and members[scenario]['checksum'] == world_seal,
                'Collection membership mismatch')
        dataset = next(d for d in document['datasets'] if d['role'] == selected['role'])
        require(dataset['scenario_count'] == len(entries) and dataset['manifest_sha256'] == hashes['collection'],
                'Export/collection manifest identity mismatch')
        world_identity = {key: world[key] for key in ('map', 'enemy_spec', 'environment_config', 'horizon')}
        require(type(world['schema_version']) is int and world['schema_version'] == 2 and
                scenario == 'scn_' + digest(world_identity, world=True), 'World content identity mismatch')
        grid = world['map']
        require(identity['map_id'] == members[scenario]['map_id'] == 'map_' + digest(grid, world=True),
                'Map identity mismatch')
        width, height = grid['width'], grid['height']
        require(type(width) is int and type(height) is int and 1 <= width <= 128 and 1 <= height <= 128,
                'Invalid map dimensions')
        require(isinstance(grid['tiles'], list) and len(grid['tiles']) == height and
                all(isinstance(row, list) and len(row) == width for row in grid['tiles']), 'Invalid map shape')
        for row in grid['tiles']:
            for tile in row:
                integer(tile['elevation'], 'elevation')
                require(type(tile['obstacle']) is bool and type(tile['special_traversal']) is bool and
                        tile['hazard'] in (None, 'damage'), 'Unknown terrain encoding')

        def position(value):
            require(isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value) and
                    0 <= value[0] < width and 0 <= value[1] < height, 'Recorded position outside map')

        position(grid['start']); position(grid['goal'])
        maximum_hp = number(world['environment_config']['max_hp'], 'maximum HP', minimum=0)
        require(type(initial['steps']) is int and initial['steps'] == 0 and initial['agent_position'] == grid['start'] and
                number(initial['agent_hp'], 'reset HP', minimum=0) == maximum_hp and maximum_hp > 0,
                'Initial-state/reset mismatch')
        for field, expected in (('map_width', width), ('map_height', height), ('map_area', width * height)):
            if field in initial:
                require(type(initial[field]) is int and initial[field] == expected, 'Reset map dimensions mismatch')
        if world['enemy_spec'] is not None:
            enemy = world['enemy_spec']
            route, patrol_index = enemy['patrol_route'], enemy['initial_patrol_index']
            require(isinstance(route, list) and type(patrol_index) is int and 0 <= patrol_index < len(route),
                    'Invalid saved enemy specification')
            for location in route:
                position(location)
            require(initial.get('enemy_position') == route[patrol_index] and
                    initial.get('enemy_patrol_index') == patrol_index and
                    initial.get('enemy_behavior') == enemy['behavior'], 'Reset enemy/world mismatch')
        horizon = world['horizon']
        require(type(horizon) is int and horizon > 0 and initial['max_episode_steps'] == horizon,
                'Recorded horizon mismatch')
        require(isinstance(steps, list) and 1 <= len(steps) <= min(horizon, 10000), 'Invalid recorded trace length')
        require(type(metrics['length']) is int and metrics['length'] == len(steps) and
                metrics['trace_sha256'] == trace_seal and metrics['trajectory_sha256'] == digest(steps),
                'Case/trace summary seal mismatch')
        for key in ('success', 'death', 'timeout', 'zero_damage_success', 'terminated', 'truncated'):
            require(type(metrics[key]) is bool, 'Recorded outcome must be Boolean')
        require(sum(metrics[key] for key in ('success', 'death', 'timeout')) == 1,
                'Recorded outcomes do not reconcile')
        end_reason = 'death' if metrics['death'] else 'goal' if metrics['success'] else 'timestep_limit'
        frames = [dict(step=0, action=None, reward=None, events=None, agent_position=initial['agent_position'],
                       agent_hp=initial['agent_hp'], enemy_position=initial.get('enemy_position'),
                       enemy_patrol_index=initial.get('enemy_patrol_index'), terminated=False, truncated=False, end_reason=None)]
        for step_number, step in enumerate(steps, 1):
            require(type(step['action']) is int and 0 <= step['action'] <= 4, 'Unknown recorded action')
            require(type(step['terminated']) is bool and type(step['truncated']) is bool,
                    'Invalid recorded terminal flags')
            require(not (step['terminated'] and step['truncated']), 'Conflicting terminal flags')
            if step_number < len(steps):
                require(not step['terminated'] and not step['truncated'], 'Actions continue after terminal state')
            events = step['events']
            for key in ('waited', 'invalid_move', 'reached_goal', 'hazard_contact', 'enemy_contact'):
                require(type(events[key]) is bool, 'Missing recorded event flags')
            require(events['waited'] == (step['action'] == 4) and not (events['waited'] and events['invalid_move']),
                    'WAIT and invalid actions are not interchangeable')
            number(step['reward'], 'raw reward')
            for key in ('damage_taken', 'fall_damage', 'hazard_damage', 'enemy_damage'):
                number(events[key], key, minimum=0)
            require(math.isclose(events['damage_taken'], sum(events[k] for k in ('fall_damage', 'hazard_damage', 'enemy_damage')),
                                 abs_tol=1e-8), 'Damage sources do not reconcile')
            frames.append(dict(step=step_number, action=step['action'], reward=step['reward'], events=events,
                               agent_position=step['position'], agent_hp=step['hp'], enemy_position=step['enemy_position'],
                               enemy_patrol_index=step['enemy_patrol_index'], terminated=step['terminated'],
                               truncated=step['truncated'], end_reason=end_reason if step_number == len(steps) else None))
        for frame in frames:
            position(frame['agent_position'])
            number(frame['agent_hp'], 'recorded HP', minimum=0)
            require(frame['agent_hp'] <= maximum_hp, 'Recorded HP exceeds maximum')
            require(frame['agent_hp'] > 0 or frame['terminated'], 'Nonterminal recorded state has no HP')
            if world['enemy_spec'] is None:
                require(frame['enemy_position'] is None and frame['enemy_patrol_index'] is None, 'Unexpected recorded enemy')
            else:
                position(frame['enemy_position'])
                integer(frame['enemy_patrol_index'], 'enemy_patrol_index')
        last = frames[-1]
        require(last['agent_hp'] == metrics['final_hp'] and last['terminated'] == metrics['terminated'] and
                last['truncated'] == metrics['truncated'], 'Final-state summary mismatch')
        require(last['terminated'] == (metrics['death'] or metrics['success']) and last['truncated'] == metrics['timeout'],
                'Incomplete or inconsistent recorded episode')
        require(metrics['death'] == (last['agent_hp'] == 0), 'Fatal outcome priority mismatch')
        require(not metrics['success'] or (last['agent_position'] == grid['goal'] and last['agent_hp'] > 0 and
                                           last['events']['reached_goal']), 'Living-goal success mismatch')
        for metric, event in (('wait', 'waited'), ('illegal_moves', 'invalid_move'), ('hazard_contacts', 'hazard_contact'),
                              ('enemy_contacts', 'enemy_contact'), ('damage', 'damage_taken'), ('fall_damage', 'fall_damage'),
                              ('hazard_damage', 'hazard_damage'), ('enemy_damage', 'enemy_damage')):
            number(metrics[metric], metric, minimum=0)
            require(math.isclose(sum(step['events'][event] for step in steps), metrics[metric], abs_tol=1e-8),
                    f'Recorded {metric} total mismatch')
        require(math.isclose(sum(step['reward'] for step in steps), number(metrics['raw_return'], 'raw return'), abs_tol=1e-8),
                'Raw return mismatch')
        require(metrics['zero_damage_success'] == (metrics['success'] and metrics['damage'] == 0),
                'Zero-damage success mismatch')
        require(math.isclose(number(metrics['invalid_move_rate'], 'invalid move rate', minimum=0),
                             metrics['illegal_moves'] / len(steps), abs_tol=1e-8), 'Invalid-action rate mismatch')
        outcome = dict(final_position=last['agent_position'], final_hp=last['agent_hp'], success=metrics['success'],
                       terminated=last['terminated'], truncated=last['truncated'], end_reason=end_reason)
        return dict(available=True, selection=selected, map=grid, frames=frames, outcome=outcome,
                    enemy_spec=world['enemy_spec'], fixture=fixture,
                    verification={'status': 'recorded_artifact_joins_checked'},
                    frame_semantics='Frame 0 is the saved reset; each later frame is the saved state after its action.',
                    identity_checks='Indexed bytes, export, job, checkpoint, collection, world, case, trace and event totals checked. No policy or physics execution.',
                    source_artifact_ids=ids, source_hashes=hashes)
    except (ValueError, OSError, KeyError, TypeError, IndexError, StopIteration, OverflowError) as error:
        result['reason_unavailable'] = str(error)
        return result
