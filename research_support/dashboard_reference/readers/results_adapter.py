"""Read-only research-results transport adapter. No simulator, policy, or inference imports."""
import hashlib
import json
import math
import re
from pathlib import Path

from .bundle_paths import configured_root, portable_relative
from ..protocols import (FINAL_ROLES, RECURRENT_PROTOCOL_ID, get_protocol,
                         validate_training_steps)

# Historical imports retain their historical meaning.
ROLE_COUNTS = get_protocol()['role_counts']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite JSON number')
    elif isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, list):
        for v in value: finite(v)


def local_results_path(value, *, allowed_root, base=None):
    """Resolve an explicit local export inside a caller-configured root."""
    root = configured_root(allowed_root)
    require(isinstance(value, (str, Path)) and bool(str(value)), 'Missing local path')
    raw = str(value)
    require(not raw.startswith(('\\\\', '//')) and '://' not in raw, 'Network paths are excluded')
    p = Path(value)
    require('..' not in p.parts, 'Traversal paths are excluded')
    if not p.is_absolute():
        base_path = Path(base) if base is not None else root
        if not base_path.is_absolute(): base_path = root / base_path
        p = base_path / p
    p = p.resolve()
    require(p.is_relative_to(root), 'Path must stay inside its configured root')
    require(not any(part.startswith('.') for part in p.relative_to(root).parts), 'Hidden paths are excluded')
    return p


def read_results(path, *, allowed_root, expected_hash=None):
    p = local_results_path(path, allowed_root=allowed_root)
    require(p.suffix.lower() == '.json', 'Result sources must be JSON')
    require(p.stat().st_size <= 48 * 1024**2, 'JSON exceeds 48 MiB')
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None: require(sha == expected_hash, 'Source SHA256 mismatch; refresh after replacing exports')
    value = json.loads(raw.decode('utf-8-sig'))
    finite(value)
    return value, sha


def integer(v, label):
    require(type(v) is int and v >= 0, f'{label} must be a nonnegative integer')


def sha256(v, label, fixture):
    require((fixture and v is None) or (isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)), f'{label}: SHA256 required (null only in fixtures)')


def adapt(document, path, source_hash):
    """Validate joins/counts and expose exported statistics without recomputing them."""
    path = str(path)
    if path.startswith('artifact:'):
        require(re.fullmatch('[a-z][a-z0-9_]*', path[len('artifact:'):]), 'Invalid logical artifact reference')
    else:
        portable_relative(path)

    d = document
    require(isinstance(d, dict), 'Expected a research results object')
    require(d.get('schema_version') in ('research-results-v1', 'm4-results-v1', 'research-results-v2'), 'Unsupported research results schema_version')
    require(d['schema_version'] != 'research-results-v2' or 'protocol_id' in d,
            'research-results-v2 requires an explicit protocol_id')
    protocol = get_protocol(d.get('protocol_id'))
    require(d['schema_version'] in protocol['schema_versions'], 'Scientific protocol/schema_version mismatch')
    recurrent = protocol['id'] == RECURRENT_PROTOCOL_ID
    budget = protocol['budget']

    finite(d)
    required = ('study_id', 'provenance', 'datasets', 'models', 'outcomes', 'learning', 'comparisons', 'traces', 'limitations')
    
    require(all(k in d for k in required), 'Incomplete research results export')
    require(isinstance(d['study_id'], str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', d['study_id']), 'Invalid study_id')
    require(type(d.get('fixture', False)) is bool, 'fixture must be boolean')
    fixture = d.get('fixture', False)
    require(isinstance(d['provenance'], dict), 'provenance must be an object')
    for k in ('metric_units', 'trace_selection'):
        require(isinstance(d.get(k, {}), dict), f'{k} must be an object')
    for k in ('title', 'status', 'protocol_id', 'evaluation_mode'):
        if k in d: require(isinstance(d[k], str) and bool(d[k]), f'{k} must be a nonempty string')
    for k in required[2:]: require(isinstance(d[k], list), f'{k} must be an array')
    require(all(isinstance(x, str) for x in d['limitations']), 'limitations must contain strings')
    datasets, models, dimensions = {}, {}, set()
    for ds in d['datasets']:
        require(isinstance(ds, dict), 'Invalid dataset')
        role = ds.get('role')
        require(isinstance(role, str) and role and role not in datasets, 'Missing/duplicate dataset role')
        integer(ds.get('scenario_count'), 'scenario_count')
        if recurrent:
            require(role in protocol['role_counts'], 'Unknown recurrent dataset role')
        if role in protocol['role_counts'] and not fixture:
            require(ds['scenario_count'] == protocol['role_counts'][role], f'{role}: wrong protocol denominator')
        require(isinstance(ds.get('manifest_path'), str), 'manifest_path required')
        sha256(ds.get('manifest_sha256'), 'manifest_sha256', fixture)
        datasets[role] = ds
    for m in d['models']:
        require(isinstance(m, dict), 'Invalid model')
        mid = m.get('model_id')
        require(isinstance(mid, str) and mid and mid not in models, 'Missing/duplicate model_id')
        require(isinstance(m.get('reward'), str) and m['reward'], 'reward required')
        integer(m.get('seed'), 'seed'); integer(m.get('checkpoint_step'), 'checkpoint_step')
        if recurrent:
            require(m['reward'] in protocol['rewards'], 'Unknown recurrent reward condition')
            require(m['seed'] in protocol['primary_seeds'], 'Unknown recurrent primary seed')
            validate_training_steps(m, protocol['id'])
            for slot_field in ('slot', 'seed_slot'):
                if slot_field in m:
                    slot = m[slot_field]
                    require(type(slot) is int and 0 <= slot < len(protocol['primary_seeds']), 'Invalid primary seed slot')
                    require(m['seed'] == protocol['primary_seeds'][slot], 'Primary seed/slot mismatch')
            if 'slot' in m and 'seed_slot' in m:
                require(m['slot'] == m['seed_slot'], 'Conflicting primary seed slots')
        key = (m['reward'], m['seed'], m['checkpoint_step'])
        require(key not in dimensions, 'Duplicate reward/seed/checkpoint models; use separate studies for attempts')
        dimensions.add(key); models[mid] = m
    rows = {}
    seen_by_kind = {}
    def joined(r):
        require(isinstance(r, dict), 'Invalid row')
        require(isinstance(r.get('model_id'), str) and r['model_id'] in models, 'Unknown model_id join')
        require(isinstance(r.get('role'), str) and r['role'] in datasets, 'Unknown dataset role join')
        return models[r['model_id']], datasets[r['role']]
    for kind in ('outcomes', 'learning'):
        seen = set()
        seen_by_kind[kind] = seen
        for r in d[kind]:
            m, ds = joined(r)
            if recurrent:
                require(r['role'] not in (*FINAL_ROLES, 'TRAIN') or m['checkpoint_step'] == budget,
                        'Recurrent TRAIN/final outcomes require the final checkpoint')
                if kind == 'learning':
                    require(r['role'] in ('DEV', 'LEARNING_VALIDATION'), 'Invalid recurrent learning role')
                for field in ('actual_training_steps', 'optimized_training_steps', 'final_unoptimized_steps'):
                    if field in r:
                        require(type(r[field]) is int and r[field] == m.get(field), f'{field}/model join mismatch')
            key = (r['model_id'], r['role'])
            require(key not in seen, f'Duplicate {kind} row'); seen.add(key)
            for field in ('n_expected', 'n_completed', 'success_count'):
                integer(r.get(field), field)
            require(r['n_expected'] == ds['scenario_count'], 'n_expected differs from dataset scenario_count')
            require(r['success_count'] <= r['n_completed'] <= r['n_expected'], 'Invalid completed/success counts')
            if recurrent and 'complete' in r:
                require(type(r['complete']) is bool and r['complete'] == (r['n_completed'] == r['n_expected']),
                        'Completion flag/count mismatch')
            for field in ('mean_damage', 'mean_length', 'mean_length_success', 'mean_length_failure'):
                if r.get(field) is not None:
                    require(type(r[field]) in (float, int) and r[field] >= 0, f'Invalid {field}')
            for field in ('invalid_actions', 'transitions'):
                if r.get(field) is not None: integer(r[field], field)
            if r.get('invalid_actions') is not None:
                require(r.get('transitions') is not None and 0 < r['transitions'] and r['invalid_actions'] <= r['transitions'], 'Invalid transition denominator')
            if kind == 'outcomes':
                for field in ('death_count', 'timeout_count'): integer(r.get(field), field)
                require(sum(r[k] for k in ('success_count', 'death_count', 'timeout_count')) == r['n_completed'], 'Outcome counts must reconcile completed episodes')
            else:
                require(type(r.get('checkpoint_step')) is int and r['checkpoint_step'] == m['checkpoint_step'], 'Learning checkpoint/model join mismatch')
            if key in rows:
                require(all(rows[key][k] == r[k] for k in ('n_expected', 'n_completed', 'success_count')), 'Learning/outcome counts disagree')
            rows[key] = dict(rows.get(key, {}), **r)
    # Missing cells remain inventory gaps rather than failed policies. Recurrent
    # requirements cannot be reduced by an incomplete expected_evaluations list.
    defaults = [dict(model_id=m['model_id'], role=role)
                for m in models.values() if m['checkpoint_step'] == budget
                for role in FINAL_ROLES if role in datasets]
    if recurrent:
        defaults += [dict(model_id=m['model_id'], role=role) for m in models.values()
                     for role in ('LEARNING_VALIDATION', 'TRAIN') if role in datasets
                     and (role == 'LEARNING_VALIDATION' or m['checkpoint_step'] == budget)]
    expected = d.get('expected_evaluations', defaults)
    require(isinstance(expected, list), 'expected_evaluations must be an array')
    expected_seen = set()
    for r in expected:
        m, ds = joined(r)
        key = (r['model_id'], r['role'])
        require(key not in expected_seen, 'Duplicate expected evaluation'); expected_seen.add(key)
        if recurrent:
            require(r['role'] not in (*FINAL_ROLES, 'TRAIN') or m['checkpoint_step'] == budget,
                    'Recurrent TRAIN/final expectations require the final checkpoint')
        rows.setdefault((r['model_id'], r['role']), dict(r, n_expected=ds['scenario_count'], n_completed=0, success_count=None, reason_unavailable='No saved result row'))
    if recurrent:
        for r in defaults:
            ds = datasets[r['role']]
            rows.setdefault((r['model_id'], r['role']), dict(r, n_expected=ds['scenario_count'], n_completed=0, success_count=None, reason_unavailable='No saved result row'))
    records = []
    for (mid, role), r in rows.items():
        m = models[mid]; n = r['n_completed']
        records.append(dict(id=f'{mid}:{role}', model_id=mid, reward=m['reward'], condition=m['reward'],
            split=role, seed=m['seed'], checkpoint=m['checkpoint_step'],
            actual_training_steps=m.get('actual_training_steps'),
            optimized_training_steps=m.get('optimized_training_steps'),
            evaluation_mode=d.get('evaluation_mode', 'unspecified'), orientation='original',
            successes=r['success_count'] if n else None, episodes=n or None,
            deaths=r.get('death_count') if n else None, timeouts=r.get('timeout_count') if n else None,
            n_expected=r['n_expected'], n_completed=n, n_missing=r['n_expected']-n,
            mean_damage=r.get('mean_damage'), invalid_actions=r.get('invalid_actions'), transitions=r.get('transitions'),
            source=str(path), evidence='synthetic fixture' if fixture else 'research evaluator export',
            data_status='complete' if n == r['n_expected'] and n else 'partial' if n else 'unavailable',
            metadata={'exported': r, 'model': m}))
        
    traces = []
    trace_keys = set()

    for t in d['traces']:
        m, ds = joined(t)
        require(t['role'] in FINAL_ROLES and m['checkpoint_step'] == budget, f'Selected final traces require a FINAL role and {budget} checkpoint')
        require(t.get('selection_category') in ('matched', 'representative'), 'Unknown trace selection category')
        require(isinstance(t.get('scenario_id'), str) and t['scenario_id'], 'Trace scenario_id required')
        require(type(t.get('available')) is bool, 'Trace available must be boolean')
        key = (t['selection_category'], t['model_id'], t['role'], t['scenario_id'])
        require(key not in trace_keys, 'Duplicate selected trace'); trace_keys.add(key)
        if t['selection_category'] == 'matched' and (recurrent or not fixture):
            slot = m.get('slot', m.get('seed_slot'))
            require(slot == 0 if recurrent else slot is None or slot == 0, 'Matched traces require explicit primary slot 0' if recurrent else 'Matched traces require primary slot 0')
        traces.append(dict(t, reward=m['reward'], seed=m['seed'], checkpoint=m['checkpoint_step'],
                           actual_training_steps=m.get('actual_training_steps'),
                           optimized_training_steps=m.get('optimized_training_steps'),
                           slot=m.get('slot', m.get('seed_slot'))))

    inventory = None
    status = d.get('status', 'partial')
    if recurrent:
        required_dimensions = {(reward, seed, step) for reward in protocol['rewards']
                               for seed in protocol['primary_seeds'] for step in protocol['grid']}
        missing_models = [dict(reward=reward, seed=seed, checkpoint_step=step)
                          for reward, seed, step in sorted(required_dimensions - dimensions)]
        missing_roles = [role for role in protocol['required_roles'] if role not in datasets]
        missing_evaluations = []
        for r in defaults:
            key = (r['model_id'], r['role'])
            kind = 'learning' if r['role'] == 'LEARNING_VALIDATION' else 'outcomes'
            saved = rows[key]
            if key not in seen_by_kind[kind] or not saved['n_completed'] or saved['n_completed'] != saved['n_expected']:
                missing_evaluations.append(dict(r, n_expected=saved['n_expected'], n_completed=saved['n_completed']))
        complete = not missing_models and not missing_roles and not missing_evaluations
        inventory = dict(complete=complete, scientific_validation='not_assessed',
                         expected_models=len(required_dimensions), observed_models=len(models),
                         expected_final_models=len(protocol['rewards']) * len(protocol['primary_seeds']),
                         missing_models=missing_models, missing_dataset_roles=missing_roles,
                         missing_evaluations=missing_evaluations)
        if not complete and status.lower() in ('complete', 'completed', 'pass', 'passed', 'success', 'qualified'):
            status = 'partial'

    return dict(id=d['study_id'], title=d.get('title', d['study_id']),
        status=status, protocol=protocol['id'],
        group='Synthetic fixtures' if fixture else 'Research results', fixture=fixture,
        sources=[str(path)], caveats=d['limitations'], records=records,
        metadata={'provenance': d['provenance'], 'datasets': d['datasets'], 'source_sha256': source_hash,
                  'reported_status': d.get('status'), 'schema_version': d['schema_version']},
        results={'export_path': str(path), 'export_sha256': source_hash, 'traces': traces,
            'protocol_id': protocol['id'], 'checkpoint_grid': list(protocol['grid']),
            'training_budget': budget, 'inventory': inventory,
            'final_roles': list(FINAL_ROLES), 'learning': d['learning'], 'comparisons': d['comparisons'],
            'outcomes': d['outcomes'], 'extensions': {k: v for k, v in d.items() if k not in required and k != 'fixture'},
            'units': d.get('metric_units', {}), 'selection': d.get('trace_selection', {}),
            'final_model_inventory': sum(m['checkpoint_step'] == budget for m in models.values()),
            'missing_final_roles': [r for r in FINAL_ROLES if r not in datasets]})


def load_export(path, *, allowed_root):
    p = local_results_path(path, allowed_root=allowed_root)
    d, sha = read_results(p, allowed_root=allowed_root)
    return adapt(d, p.relative_to(configured_root(allowed_root)).as_posix(), sha)


def load_bundle_export(bundle, artifact_id):
    raw = bundle.read_bytes(artifact_id)
    from .bundle_paths import strict_json
    document = strict_json(raw.decode('utf-8-sig'))
    study = adapt(document, 'artifact:'+artifact_id, hashlib.sha256(raw).hexdigest())
    study['results']['export_artifact_id'] = artifact_id
    return study
