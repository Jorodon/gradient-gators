"""Read-only research-results transport adapter. No simulator, policy, or inference imports."""
import hashlib
import json
import math
import re
from pathlib import Path

from .bundle_paths import configured_root, portable_relative

FINAL_ROLES = ('FINAL_ID', 'FINAL_OOD_HAZARD', 'FINAL_OOD_OBSTACLE',
               'FINAL_OOD_PURSUIT', 'FINAL_OOD_COMBINED')
ROLE_COUNTS = dict(TRAIN=300, DEV=100, LEARNING_VALIDATION=100,
                   **dict.fromkeys(FINAL_ROLES, 200))


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
    require(isinstance(d, dict) and d.get('schema_version') in ('research-results-v1', 'm4-results-v1'), 'Expected research-results-v1 or its documented legacy wire ID')

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
        if role in ROLE_COUNTS and not fixture:
            require(ds['scenario_count'] == ROLE_COUNTS[role], f'{role}: wrong protocol denominator')
        require(isinstance(ds.get('manifest_path'), str), 'manifest_path required')
        sha256(ds.get('manifest_sha256'), 'manifest_sha256', fixture)
        datasets[role] = ds
    for m in d['models']:
        require(isinstance(m, dict), 'Invalid model')
        mid = m.get('model_id')
        require(isinstance(mid, str) and mid and mid not in models, 'Missing/duplicate model_id')
        require(isinstance(m.get('reward'), str) and m['reward'], 'reward required')
        integer(m.get('seed'), 'seed'); integer(m.get('checkpoint_step'), 'checkpoint_step')
        key = (m['reward'], m['seed'], m['checkpoint_step'])
        require(key not in dimensions, 'Duplicate reward/seed/checkpoint models; use separate studies for attempts')
        dimensions.add(key); models[mid] = m
    rows = {}
    def joined(r):
        require(isinstance(r, dict), 'Invalid row')
        require(isinstance(r.get('model_id'), str) and r['model_id'] in models, 'Unknown model_id join')
        require(isinstance(r.get('role'), str) and r['role'] in datasets, 'Unknown dataset role join')
        return models[r['model_id']], datasets[r['role']]
    for kind in ('outcomes', 'learning'):
        seen = set()
        for r in d[kind]:
            m, ds = joined(r)
            key = (r['model_id'], r['role'])
            require(key not in seen, f'Duplicate {kind} row'); seen.add(key)
            for field in ('n_expected', 'n_completed', 'success_count'):
                integer(r.get(field), field)
            require(r['n_expected'] == ds['scenario_count'], 'n_expected differs from dataset scenario_count')
            require(r['success_count'] <= r['n_completed'] <= r['n_expected'], 'Invalid completed/success counts')
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
    # Missing final cells are inventory gaps, not failed policies. Other schedules can
    # explicitly declare expected_evaluations; never guess legacy TRAIN/DEV counts.
    expected = d.get('expected_evaluations', [dict(model_id=m['model_id'], role=role)
                for m in models.values() if m['checkpoint_step'] == 1000000
                for role in FINAL_ROLES if role in datasets])
    require(isinstance(expected, list), 'expected_evaluations must be an array')
    for r in expected:
        m, ds = joined(r)
        rows.setdefault((r['model_id'], r['role']), dict(r, n_expected=ds['scenario_count'], n_completed=0, success_count=None, reason_unavailable='No saved result row'))
    records = []
    for (mid, role), r in rows.items():
        m = models[mid]; n = r['n_completed']
        records.append(dict(id=f'{mid}:{role}', model_id=mid, reward=m['reward'], condition=m['reward'],
            split=role, seed=m['seed'], checkpoint=m['checkpoint_step'],
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
        require(t['role'] in FINAL_ROLES and m['checkpoint_step'] == 1000000, 'Selected final traces require a FINAL role and 1M checkpoint')
        require(t.get('selection_category') in ('matched', 'representative'), 'Unknown trace selection category')
        require(isinstance(t.get('scenario_id'), str) and t['scenario_id'], 'Trace scenario_id required')
        require(type(t.get('available')) is bool, 'Trace available must be boolean')
        key = (t['selection_category'], t['model_id'], t['role'], t['scenario_id'])
        require(key not in trace_keys, 'Duplicate selected trace'); trace_keys.add(key)
        if t['selection_category'] == 'matched' and not fixture:
            slot = m.get('slot', m.get('seed_slot'))
            require(slot is None or slot == 0, 'Matched traces require primary slot 0')
        traces.append(dict(t, reward=m['reward'], seed=m['seed'], checkpoint=m['checkpoint_step'], slot=m.get('slot', m.get('seed_slot'))))

    return dict(id=d['study_id'], title=d.get('title', d['study_id']),
        status=d.get('status', 'partial'), protocol=d.get('protocol_id', 'unspecified'),
        group='Synthetic fixtures' if fixture else 'Research results', fixture=fixture,
        sources=[str(path)], caveats=d['limitations'], records=records,
        metadata={'provenance': d['provenance'], 'datasets': d['datasets'], 'source_sha256': source_hash},
        results={'export_path': str(path), 'export_sha256': source_hash, 'traces': traces,
            'final_roles': list(FINAL_ROLES), 'learning': d['learning'], 'comparisons': d['comparisons'],
            'outcomes': d['outcomes'], 'extensions': {k: v for k, v in d.items() if k not in required and k != 'fixture'},
            'units': d.get('metric_units', {}), 'selection': d.get('trace_selection', {}),
            'final_model_inventory': sum(m['checkpoint_step'] == 1000000 for m in models.values()),
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
