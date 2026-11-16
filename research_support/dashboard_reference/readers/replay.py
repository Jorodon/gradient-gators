"""Render only recorded trajectory_recorder v1 / scenario_serialization v2 data.

This module checks file/identity/summary consistency; it does not verify physics
or run the environment.
"""
import hashlib
import json
import math

from pathlib import Path
from .results_adapter import local_results_path, read_results, require, sha256, integer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_selected(study, index, *, bundle=None, allowed_root=None):
    require(type(index) is int and 0 <= index < len(study['results']['traces']), 'Invalid trace index')

    t = study['results']['traces'][index]
    if t.get('replay_format') == 'recurrent-primary-v1':
        from .recurrent_replay import load_recurrent_selected
        return load_recurrent_selected(study, index, bundle=bundle)
    result = dict(available=False, selection=t)

    try:
        # helpers
        def read_selected(kind):
                    artifact_id = t.get(kind+'_artifact_id')
                    if artifact_id:
                        require(bundle is not None, 'An explicit bundle is required for release artifacts')
                        raw = bundle.read_bytes(artifact_id)
                        actual_hash = hashlib.sha256(raw).hexdigest()
                        if t.get(kind+'_sha256') is not None:
                            require(actual_hash == t[kind+'_sha256'], 'Selected artifact hash differs from export')
                        return bundle.read_json(artifact_id), actual_hash
                    path = local_results_path(t.get(kind+'_path'), allowed_root=allowed_root, base=root)
                    return read_results(path, allowed_root=allowed_root, expected_hash=t.get(kind+'_sha256'))

        def position(p):
                    require(isinstance(p, list) and len(p) == 2 and all(type(v) is int for v in p) and 0 <= p[0] < width and 0 <= p[1] < height, 'Position outside map')
        
        require(t['available'], t.get('reason_unavailable') or 'Trace was not saved')
        fixture = study.get('fixture', False)

        if not fixture and t['selection_category'] == 'matched':
            require(t.get('slot') == 0, 'Matched primary slot 0 identity unavailable')
        transport = study['results']

        if transport.get('export_artifact_id'):
            require(bundle is not None, 'An explicit bundle is required')
            require(hashlib.sha256(bundle.read_bytes(transport['export_artifact_id'])).hexdigest() == transport['export_sha256'], 'Export source hash changed')

        else:
            read_results(transport['export_path'], allowed_root=allowed_root, expected_hash=transport['export_sha256'])

        root = Path(transport['export_path']).parent
        for field in ('trajectory_sha256', 'snapshot_sha256'):
            sha256(t.get(field), field, fixture)

        
        trace, trace_sha = read_selected('trajectory')
        snapshot, snapshot_sha = read_selected('snapshot')
        require(isinstance(trace, dict) and isinstance(snapshot, dict), 'Trace and snapshot must be objects')
        require(trace.get('schema_version') == 1 and snapshot.get('schema_version') == 2, 'Unsupported saved serialization version')

        manifest = trace.get('scenario_manifest', {})
        require(isinstance(manifest, dict), 'Trace manifest must be an object')

        for key in ('trace_id', 'episode_id', 'episode_record_id'):
            require(isinstance(trace.get(key), str) and bool(trace[key]), f'Trace {key} is required')

        require(isinstance(t.get('episode_id'), str) and bool(t['episode_id']), 'Exported episode_id required')
        require(all(v == t['scenario_id'] for v in (trace.get('scenario_id'), snapshot.get('scenario_id'), manifest.get('scenario_id'))), 'Scenario identity mismatch')
        require(trace.get('model_id') == t['model_id'] and trace.get('reward_type') == t['reward'], 'Trace model/reward mismatch')
        require(trace.get('distribution_label') == t['role'], 'Trace condition mismatch')

        unsigned = dict(snapshot); checksum = unsigned.pop('checksum', None)
        require(checksum == digest(unsigned), 'Snapshot content checksum mismatch')

        world = {k: snapshot[k] for k in ('map', 'enemy_spec', 'environment_config', 'horizon')}
        require(snapshot['scenario_id'] == 'scn_' + digest(world), 'Snapshot world identity mismatch')
        require(trace.get('scenario_checksum') == manifest.get('scenario_checksum') == checksum, 'Trace/snapshot checksum join mismatch')

        grid = snapshot['map']
        require(isinstance(grid, dict), 'Map must be an object')
        require(manifest.get('map') == grid and manifest.get('map_id') == 'map_' + digest(grid), 'Trace terrain differs from snapshot')
        require(manifest.get('environment_config') == snapshot['environment_config'], 'Trace configuration differs from snapshot')

        # Legacy recorder omits behavior from enemy_spec, do not infer it
        enemy = manifest.get('enemy_spec'); actual_enemy = snapshot.get('enemy_spec')
        require((enemy is None) == (actual_enemy is None), 'Enemy presence differs')

        if enemy is not None:
            require(isinstance(enemy, dict) and isinstance(actual_enemy, dict), 'Enemy specification must be an object')
            require(all(actual_enemy.get(k) == v for k, v in enemy.items()), 'Enemy specification differs')

        width, height = grid.get('width'), grid.get('height')
        require(type(width) is int and type(height) is int and 1 <= width <= 128 and 1 <= height <= 128, 'Invalid map dimensions')
        require(isinstance(grid.get('tiles'), list) and len(grid['tiles']) == height and all(isinstance(row, list) and len(row) == width for row in grid['tiles']), 'Invalid tile dimensions')

        for row in grid['tiles']:
            for tile in row:
                require(isinstance(tile, dict) and all(k in tile for k in ('elevation', 'obstacle', 'hazard', 'special_traversal')), 'Unknown tile state')
                integer(tile['elevation'], 'elevation')
                require(type(tile['obstacle']) is bool and type(tile['special_traversal']) is bool and tile['hazard'] in (None, 'damage'), 'Unknown terrain encoding')
        
        position(grid['start']); position(grid['goal'])

        frames = trace.get('frames')
        require(isinstance(frames, list) and 1 <= len(frames) <= 10001, 'Missing or excessive frames')

        frame_keys = ('step', 'action', 'agent_position', 'agent_hp', 'enemy_position', 'enemy_patrol_index', 'reward', 'events', 'terminated', 'truncated', 'end_reason')
        for i, f in enumerate(frames):
            require(isinstance(f, dict) and all(k in f for k in frame_keys), 'Incomplete recorded frame')
            require(type(f['step']) is int and f['step'] == i, 'Unordered frames')
            position(f['agent_position'])
            if f['enemy_position'] is not None: position(f['enemy_position'])
            require(type(f['agent_hp']) in (float, int) and f['agent_hp'] >= 0, 'Invalid HP')
            require(type(f['terminated']) is bool and type(f['truncated']) is bool, 'Invalid terminal flags')
            if i == 0:
                require(all(f[k] is None for k in ('action', 'reward', 'events')), 'Reset frame contains transition')
            else:
                require(type(f['action']) is int and 0 <= f['action'] <= 4 and type(f['reward']) in (int, float) and isinstance(f['events'], dict), 'Unknown transition state')
            if i < len(frames)-1: require(not f['terminated'] and not f['truncated'], 'Frames continue after termination')

        summary = t.get('episode_summary')
        require(isinstance(summary, dict) and summary == trace.get('episode_summary'), 'Exported episode summary join unavailable or different')
        require(t.get('episode_id') == trace.get('episode_id') == trace.get('episode_record_id'), 'Episode identity join mismatch')
        require(summary.get('scenario_id') == t['scenario_id'] and summary.get('reward_type') == t['reward'], 'Episode summary identity mismatch')
        require(summary.get('episode_length') == len(frames)-1, 'Summary length differs from trace')

        last = frames[-1]; outcome = trace.get('outcome', {})
        require(isinstance(outcome, dict), 'Outcome must be an object')
        require(outcome.get('final_position') == last['agent_position'] and outcome.get('final_hp') == last['agent_hp'], 'Outcome/final state mismatch')

        for k in ('terminated', 'truncated', 'end_reason'):
            require(outcome.get(k) == last[k] == summary.get(k), f'Summary/trace {k} mismatch')

        require(type(summary.get('success')) is bool and outcome.get('success') == summary['success'], 'Summary success mismatch')
        require(last['terminated'] or last['truncated'], 'Incomplete trajectory')

        verification = t.get('verification', {})
        require(isinstance(verification, dict) and verification.get('status') in ('captured_original', 'replay_exact'), 'Original capture or exact replay verification unavailable')
        if verification['status'] == 'replay_exact':
            require(verification.get('summary_match') is True and verification.get('events_match') is True, 'Replay summary/events match unavailable')

        return dict(available=True, selection=t, map=grid, frames=frames, outcome=outcome,
            enemy_spec=actual_enemy, fixture=fixture, verification=verification,
            identity_checks='File hashes, world identity, terrain, model, role and episode summary joins checked. No physics or policy re-execution.',
            source_hashes=dict(trajectory=trace_sha, snapshot=snapshot_sha))
    
    except (ValueError, OSError, KeyError, TypeError) as error:
        result['reason_unavailable'] = str(error)
        return result


def load_recorded_binding(bundle, binding):
    """
    Validate an indexed TRAIN/DEV case binding and normalize saved frames only.
    """
    result = dict(available=False, selection=binding)
    try:
        require(binding['role'] in ('TRAIN','DEV'), 'Recorded-case reader excludes final/OOD roles')
        names = ('case','trace','world','configuration','checkpoint_index','collection_manifest')
        ids = {name: binding[name+'_artifact_id'] for name in names}
        values = {name: bundle.read_json(artifact) for name,artifact in ids.items()}
        case, trace, world = (values[name] for name in ('case','trace','world'))
        envelope, checkpoints, collection = (values[name] for name in ('configuration','checkpoint_index','collection_manifest'))


        # helpers
        def recorded_digest(value):
            return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        
        def sealed(value, field):
            require(isinstance(value,dict), 'Sealed record must be an object')
            unsigned=dict(value);seal=unsigned.pop(field,None)
            require(seal == recorded_digest(unsigned), 'Recorded content seal mismatch')
            return seal

        def position(value):
            require(isinstance(value,list) and len(value)==2 and all(type(v) is int for v in value)
                    and 0<=value[0]<width and 0<=value[1]<height, 'Recorded position outside map')

        # full validaiton start
        sealed(case,'sha256');trace_seal=sealed(trace,'sha256');sealed(collection,'checksum')
        world_checksum=sealed(world,'checksum')

        require(case['status']=='complete', 'Recorded case is incomplete')
        identity=case['identity'];metrics=case['metrics'];initial=trace['initial_state'];steps=trace['steps']

        require(trace['identity']==identity, 'Case/trace identity mismatch')
        require(identity['trace_requested'] is True, 'Case did not request this trace')
        require(identity['job_id']==envelope['source_job_id']==binding['source_job_id'], 'Original job ID mismatch')
        require(identity['job_sha256']==envelope['source_job_sha256']==binding['source_job_sha256'], 'Original job digest mismatch')
        require(envelope['schema_version']=='gradient-gators-derived-configuration-v1', 'Unsupported configuration envelope')
        require(envelope['source_file_sha256']==bundle.entries[ids['configuration']]['source_ref']['sha256'], 'Original configuration file hash mismatch')
        require(envelope['experiment_id']==binding['experiment_id'] and envelope['condition_id']==binding['condition_id'], 'Portable configuration identity mismatch')

        configuration=envelope['configuration']
        require(identity['algorithm']['algorithm']==configuration['algorithm'] and identity['reward']['family']==configuration['reward'], 'Algorithm/reward configuration mismatch')

        checkpoint=identity['checkpoint']
        require(checkpoint['model_seed']==configuration['model_seed'], 'Model seed mismatch')

        matches=[record for record in checkpoints if record['nominal_training_steps']==binding['checkpoint_step']]
        require(len(matches)==1 and matches[0]==checkpoint, 'Checkpoint index join mismatch')

        require(checkpoint['sha256']==identity['checkpoint_sha256']==binding['checkpoint_sha256'], 'Checkpoint hash mismatch')
        require(identity['role']==binding['role'] and identity['seed']==binding['evaluation_seed'], 'Role/evaluation-seed mismatch')
        require(identity['scenario_id']==binding['scenario_id']==world['scenario_id']==initial['scenario_id']==metrics['scenario_id'], 'Scenario identity mismatch')
        require(identity['scenario_sha256']==binding['scenario_file_sha256']==bundle.entries[ids['world']]['sha256'], 'World file hash mismatch')
        require(identity['scenario_checksum']==binding['scenario_checksum']==world_checksum==initial['snapshot_checksum'], 'World checksum mismatch')
        require(identity['collection_checksum']==collection['checksum'], 'Collection checksum mismatch')

        expected_count=100 if identity['role']=='DEV' else configuration['train_count']
        expected_split='CHECKPOINT_VALIDATION' if identity['role']=='DEV' else 'TRAIN'

        require(collection['split']==initial['split']==expected_split and len(collection['entries'])==expected_count, 'Collection role/count mismatch')

        entries={entry['scenario_id']:entry for entry in collection['entries']}

        require(len(entries)==expected_count and entries[identity['scenario_id']]['checksum']==world_checksum, 'Collection membership mismatch')

        world_identity={key:world[key] for key in ('map','enemy_spec','environment_config','horizon')}
        require(world['schema_version']==2 and world['scenario_id']=='scn_'+recorded_digest(world_identity), 'World identity mismatch')
        grid=world['map'];width,height=grid['width'],grid['height']
        require(identity['map_id']==entries[identity['scenario_id']]['map_id']=='map_'+recorded_digest(grid), 'Map identity mismatch')
        require(type(width) is int and type(height) is int and 1<=width<=128 and 1<=height<=128, 'Invalid map dimensions')
        require(len(grid['tiles'])==height and all(len(row)==width for row in grid['tiles']), 'Invalid map shape')

        for row in grid['tiles']:
            for tile in row:
                integer(tile['elevation'],'elevation')
                require(type(tile['obstacle']) is bool and type(tile['special_traversal']) is bool and tile['hazard'] in (None,'damage'), 'Unknown terrain encoding')

        
        position(grid['start']);position(grid['goal'])
        require(initial['steps']==0 and initial['agent_position']==grid['start'], 'Initial-state/reset mismatch')

        require(initial['max_episode_steps']==world['horizon'], 'Recorded horizon mismatch')
        require(isinstance(steps,list) and 1<=len(steps)<=min(world['horizon'],10000), 'Invalid recorded trace length')
        require(metrics['length']==len(steps) and metrics['trace_sha256']==trace_seal and metrics['trajectory_sha256']==recorded_digest(steps), 'Case/trace summary seal mismatch')

        for key in ('success','death','timeout','zero_damage_success','terminated','truncated'):
            require(type(metrics[key]) is bool, 'Recorded outcome must be Boolean')

        require(sum(metrics[key] for key in ('success','death','timeout'))==1, 'Recorded outcomes do not reconcile')

        # End labels are normalized from saved outcome flags, never from goal contact alone
        end_reason='death' if metrics['death'] else 'goal' if metrics['success'] else 'timestep_limit'
        frames=[dict(step=0,action=None,reward=None,events=None,agent_position=initial['agent_position'],agent_hp=initial['agent_hp'],
                     enemy_position=initial['enemy_position'],enemy_patrol_index=initial['enemy_patrol_index'],terminated=False,truncated=False,end_reason=None)]
        
        for index,step in enumerate(steps,1):
            require(type(step['action']) is int and 0<=step['action']<=4, 'Unknown recorded action')
            require(type(step['terminated']) is bool and type(step['truncated']) is bool, 'Invalid recorded terminal flags')
            if index<len(steps):require(not step['terminated'] and not step['truncated'], 'Actions continue after terminal state')
            events=step['events']
            require(type(events['waited']) is bool and type(events['invalid_move']) is bool, 'Missing WAIT/invalid event flags')
            require(events['waited']==(step['action']==4) and not(events['waited'] and events['invalid_move']), 'WAIT and invalid action are not interchangeable')
            require(type(step['reward']) in (int,float) and math.isfinite(step['reward']), 'Invalid raw reward')
            frames.append(dict(step=index,action=step['action'],reward=step['reward'],events=events,
                agent_position=step['position'],agent_hp=step['hp'],enemy_position=step['enemy_position'],enemy_patrol_index=step['enemy_patrol_index'],
                terminated=step['terminated'],truncated=step['truncated'],end_reason=end_reason if index==len(steps) else None))
            
        for frame in frames:
            position(frame['agent_position'])
            if frame['enemy_position'] is not None:position(frame['enemy_position'])
            require(type(frame['agent_hp']) in (int,float) and math.isfinite(frame['agent_hp']) and frame['agent_hp']>=0, 'Invalid recorded HP')
        last=frames[-1]

        require(last['agent_hp']==metrics['final_hp'] and last['terminated']==metrics['terminated'] and last['truncated']==metrics['truncated'], 'Final-state summary mismatch')
        require(last['terminated'] or last['truncated'], 'Incomplete recorded episode')
        require(metrics['death']==(last['agent_hp']==0), 'Fatal outcome priority mismatch')
        require(not metrics['success'] or (last['agent_position']==grid['goal'] and last['agent_hp']>0), 'Living-goal success mismatch')
        require(metrics['timeout']==last['truncated'], 'Timeout flag mismatch')
        require(sum(step['events']['waited'] for step in steps)==metrics['wait'] and sum(step['events']['invalid_move'] for step in steps)==metrics['illegal_moves'], 'WAIT/invalid totals mismatch')
        require(math.isclose(sum(step['events']['damage_taken'] for step in steps),metrics['damage'],abs_tol=1e-8), 'Damage total mismatch')
        require(math.isclose(sum(step['reward'] for step in steps),metrics['raw_return'],abs_tol=1e-8), 'Raw return mismatch')
        require(metrics['zero_damage_success']==(metrics['success'] and metrics['damage']==0), 'Zero-damage success mismatch')

        outcome=dict(final_position=last['agent_position'],final_hp=last['agent_hp'],success=metrics['success'],
                     terminated=last['terminated'],truncated=last['truncated'],end_reason=end_reason)

        
        return dict(available=True,selection=binding,map=grid,frames=frames,outcome=outcome,enemy_spec=world['enemy_spec'],fixture=False,
            verification={'status':'recorded_artifact_joins_checked'},
            frame_semantics='Frame0 is recorded reset state; frameN is stored state after actionN. No state is simulated.',
            identity_checks='Indexed file hashes, original-job/configuration, checkpoint, collection, world, trace, outcome and event totals checked. No policy or physics execution.',
            source_artifact_ids=ids,source_hashes={name:bundle.entries[artifact]['sha256'] for name,artifact in ids.items()})
    
    except (ValueError,OSError,KeyError,TypeError) as error:
        result['reason_unavailable']=str(error)
        return result
