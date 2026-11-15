"""Bounded saved-file readers; never load models or execute experiment code."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import re

from .results_adapter import local_results_path
from .bundle_paths import configured_root
MAX_FILE = 48 * 1024 * 1024

def source_reference(path, *, allowed_root):
    return local_path(path, allowed_root=allowed_root).relative_to(configured_root(allowed_root)).as_posix()

def local_path(value, *, allowed_root):
    p = local_results_path(value, allowed_root=allowed_root)
    for part in p.parts:
        low = part.lower()
        if low.startswith('.') or re.search(r'(^|[_ -])(ood|final|heldout|held-out)([_ .-]|$)', low):
            raise ValueError('Hidden/runtime and final/held-out/OOD paths are excluded')
    return p


def read_bytes(path, *, allowed_root):
    p = local_path(path, allowed_root=allowed_root)
    if p.stat().st_size > MAX_FILE:
        raise ValueError(f'Source exceeds {MAX_FILE // 1024**2} MiB limit: {p}')
    return p.read_bytes()

def read_json(path, *, allowed_root):
    return json.loads(read_bytes(path, allowed_root=allowed_root).decode('utf-8-sig'))

def csv_snapshot(path, *, allowed_root):
    raw = read_bytes(path, allowed_root=allowed_root)
    warnings = []
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
        warnings.append('Incomplete/non-newline-terminated CSV tail ignored')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')), strict=True)
    rows = []
    try:
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                warnings.append('Incomplete/misaligned CSV row ignored')
                continue
            rows.append(row)
    except csv.Error:
        warnings.append('Unfinished/malformed CSV tail ignored')
    return rows, warnings, hashlib.sha256(raw).hexdigest()

def boolean(value):
    if str(value).lower() not in ('true', 'false', '1', '0'):
        raise ValueError('Invalid boolean')
    return str(value).lower() in ('true', '1')

def episode_records(path, metadata=None, provisional=False, *, allowed_root):
    rows, warnings, digest = csv_snapshot(path, allowed_root=allowed_root)
    groups = defaultdict(list)
    metadata = metadata or {}
    for r in rows:
        # Only TRAIN and CHECKPOINT_VALIDATION rows are measurement inputs here.
        if r.get('split') not in ('TRAIN', 'CHECKPOINT_VALIDATION'):
            warnings.append('Excluded non-TRAIN/development row')
            continue
        try:
            condition = r.get('cell') or r.get('arm') or r.get('treatment') or r.get('experiment_id')
            seed = int(r.get('model_seed') or r.get('training_seed') or r.get('seed'))
            checkpoint = int(r.get('checkpoint_timestep') or r.get('target') or 0)
            mode = r.get('action_mode') or 'deterministic'
            key = (condition, r['split'], seed, checkpoint, mode, r.get('orientation') or 'original', r.get('base_sampling_seed') or r.get('action_seed') or '')
            boolean(r['success']); boolean(r['truncated'])
            if not r.get('end_reason') or int(r['episode_length']) <= 0:
                raise ValueError('Incomplete episode')
            groups[key].append(r)
        except (KeyError, TypeError, ValueError):
            warnings.append('Invalid/incomplete episode row ignored')
    records = []
    for key, group in groups.items():
        condition, split, seed, checkpoint, mode, orientation, action_seed = key
        expected = (300 if split == 'TRAIN' else 100) if mode == 'deterministic' else 20
        parents = [r.get('parent_id') or r.get('scenario_id') for r in group]
        if len(group) != expected or len(set(parents)) != expected:
            warnings.append(f'{condition} {split} {checkpoint} {mode}: {len(group)}/{expected} rows; incomplete or duplicate group withheld')
            continue
        try:
            optional_sum = lambda k: sum(int(r[k]) for r in group) if all(r.get(k) not in (None, '') for r in group) else None
            damage = sum(float(r['damage_taken']) for r in group) / len(group) if all(r.get('damage_taken') not in (None, '') for r in group) else None
            if damage is not None and (not math.isfinite(damage) or damage < 0):
                raise ValueError('Invalid damage')
            invalid, transitions = optional_sum('invalid_moves'), optional_sum('episode_length')
            if invalid is not None and (invalid < 0 or transitions is None or invalid > transitions):
                raise ValueError('Invalid transition counts')
            details = dict(metadata.get(condition, {}))
            details['reward'] = group[0].get('reward_type') or details.get('reward')
            for k in ('actual_checkpoint_step', 'checkpoint_consumed', 'checkpoint_unique_optimized'):
                if group[0].get(k): details[k] = int(group[0][k])
            records.append(dict(id=hashlib.sha256(repr(key).encode()).hexdigest()[:20], condition=condition,
                split='TRAIN' if split == 'TRAIN' else 'DEV', seed=seed, checkpoint=checkpoint,
                evaluation_mode=mode, orientation=orientation, sampling_seed=action_seed or None,
                successes=sum(boolean(r['success']) for r in group), episodes=len(group),
                deaths=sum(r['end_reason']=='death' for r in group), timeouts=sum(boolean(r['truncated']) for r in group),
                mean_damage=damage, invalid_actions=invalid, transitions=transitions,
                metadata=details, source=source_reference(path,allowed_root=allowed_root), sha256=digest, evidence='saved episode rows',
                data_status='provisional complete group' if provisional else 'complete group',
                selection='full parent set' if mode == 'deterministic' else 'selected 20-map diagnostic; action seed is not a training seed'))
        except (ValueError, TypeError):
            warnings.append(f'{condition}: malformed numeric group withheld')
    return records, sorted(set(warnings))

def make_study(sid, title, protocol, paths, metadata, caveats, *, allowed_root):
    s = dict(id=sid, title=title, group='Recent', status='completed', protocol=protocol, metadata={}, caveats=caveats,
             sources=[source_reference(p,allowed_root=allowed_root) for p in paths], records=[], warnings=[])
    for path in paths:
        try:
            records, warnings = episode_records(path, metadata, allowed_root=allowed_root)
            s['records'].extend(records); s['warnings'].extend(warnings)
        except (OSError, ValueError) as e:
            s['warnings'].append(str(e))
    if s['warnings']: s['data_status'] = 'source gaps; see warnings'
    return s

def manual_study(root, *, allowed_root):
    root = local_path(root, allowed_root=allowed_root)
    contract = read_json(root/'experiment_contract.json', allowed_root=allowed_root)
    arms = contract.get('cells') or contract.get('arms')
    if not isinstance(arms, list) or not arms:
        raise ValueError('Unsupported run: contract needs cells[] or arms[]')
    layout = 'cells' if contract.get('cells') else 'artifacts'
    try: state = read_json(root/'state/execution_status.json', allowed_root=allowed_root)
    except (OSError, ValueError): state = {}
    raw_status = str(state.get('status', 'unknown'))
    status = raw_status if raw_status in {'completed','running','partial','failed'} else ('failed' if 'fail' in raw_status else 'unknown')
    sid = 'manual-' + root.name
    s = dict(id=sid, title=('Reward × entropy' if layout=='cells' else 'Complete collector × entropy')+' · '+root.name,
        group='Manual runs', status=status, protocol='300 TRAIN / 100 development parents; fixed target budgets; greedy original evaluation. Each launch attempt is separate.',
        sources=[source_reference(root/'experiment_contract.json',allowed_root=allowed_root),source_reference(root/'state/execution_status.json',allowed_root=allowed_root)],
        metadata={'run_id':contract.get('run_id',root.name),'recorded_status':raw_status,'status_timestamp':state.get('heartbeat') or state.get('completed_utc'),
                  'phase_progress':[dict(condition=p.get('cell'),phase=p.get('phase'),status=p.get('status'),training_steps=p.get('progress',{}).get('training_steps')) for p in state.get('phases',[])]},
        caveats=['Status is the saved controller report, not a live-process check. Refresh is manual.', 'Complete numerical groups can be provisional while the attempt remains running. Missing arms are never zero-success policies.'], records=[],warnings=[])
    for arm in arms:
        aid = arm.get('id','')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',aid): raise ValueError('Unsafe arm id in contract')
        meta = dict(arm)
        meta['entropy'] = arm.get('ent_coef')
        meta['collector'] = arm.get('collector', 'stock250' if layout=='cells' else 'complete1000')
        meta['gae_lambda'] = arm.get('gae_lambda',contract.get('ppo',{}).get('gae_lambda'))
        meta['width'] = contract.get('architecture',{}).get('global_projection_width',contract.get('scientific_method',{}).get('architecture',{}).get('global_context_dim'))
        path = root/layout/aid/'evaluation_episodes.csv'
        s['sources'].append(source_reference(path,allowed_root=allowed_root))
        try:
            rows, warnings = episode_records(path,{aid:meta},provisional=status!='completed', allowed_root=allowed_root)
            s['records'].extend(rows); s['warnings'].extend(warnings)
        except FileNotFoundError: s['warnings'].append(f'{aid}: evaluation rows not available')
        except (OSError,ValueError) as e: s['warnings'].append(str(e))
    return s


def discover_manual(parent, *, allowed_root):
    parent = local_path(parent, allowed_root=allowed_root)
    if not parent.is_dir(): return []
    # One directory level only, no recursion or environment discovery.
    return [p for p in sorted(parent.iterdir()) if p.is_dir() and (p/'experiment_contract.json').is_file()]
