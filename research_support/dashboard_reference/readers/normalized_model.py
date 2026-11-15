"""Version 1 interchange validation. No experiment/runtime imports."""
import math
import re

STATUSES = {'completed', 'running', 'partial', 'failed', 'historical', 'unknown'}
SPLITS = {'TRAIN', 'DEV', 'DIAGNOSTIC'}

def validate(document):
    def require(ok, message):
        if not ok:
            raise ValueError(message)
        
    require(isinstance(document, dict) and document.get('schema_version') == 1, 'schema_version must be 1')

    def finite_tree(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'JSON must not contain NaN or Infinity')
        elif isinstance(item, dict):
            for v in item.values(): finite_tree(v)
        elif isinstance(item, list):
            for v in item: finite_tree(v)

    finite_tree(document)
    require(isinstance(document.get('studies'), list) and document['studies'], 'studies must be a nonempty array')

    ids = set()
    for s in document['studies']:
        require(isinstance(s, dict), 'Each study must be an object')

        sid = s.get('id')
        require(isinstance(sid, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', sid), 'Study id: use 1–160 letters, digits, _, . or -')
        require(sid not in ids, f'Duplicate study id: {sid}')
        ids.add(sid)

        for key in ('title', 'protocol'):
            require(isinstance(s.get(key), str) and s[key].strip(), f'{sid}: {key} is required')

        require(s.get('status') in STATUSES, f'{sid}: invalid status')

        for key in ('sources', 'caveats'):
            require(isinstance(s.get(key), list) and all(isinstance(x, str) for x in s[key]), f'{sid}: {key} must be an array of strings')

        require(bool(s['sources']), f'{sid}: source provenance is required')
        require(isinstance(s.get('records'), list), f'{sid}: records must be an array')
        require(isinstance(s.get('metadata', {}), dict), f'{sid}: metadata must be an object')

        rids = set()
        dimensions = set()
        for r in s['records']:
            require(isinstance(r, dict), f'{sid}: each record must be an object')

            rid = r.get('id')
            require(isinstance(rid, str) and rid and rid not in rids, f'{sid}: missing/duplicate record id {rid}')
            rids.add(rid)

            require(r.get('split') in SPLITS, f'{sid}/{rid}: split must be TRAIN, DEV or DIAGNOSTIC; final ID/OOD is excluded')

            for key in ('condition', 'evaluation_mode', 'orientation', 'source', 'evidence', 'data_status'):
                require(isinstance(r.get(key), str) and r[key], f'{sid}/{rid}: {key} is required')

            require(r.get('seed') is None or (type(r['seed']) in (int, str)), f'{rid}: invalid seed')

            for key in ('checkpoint', 'successes', 'episodes', 'deaths', 'timeouts', 'invalid_actions', 'transitions'):
                v = r.get(key)
                require(v is None or (type(v) is int and v >= 0), f'{rid}: {key} must be a nonnegative integer or null')

            n, successes = r.get('episodes'), r.get('successes')
            require(n is None or n > 0, f'{rid}: episodes must be positive or null; absent results are not zero episodes')
            require(successes is None or (n is not None and successes <= n), f'{rid}: successes require episodes and cannot exceed episodes')

            for key in ('deaths', 'timeouts'):
                require(r.get(key) is None or (n is not None and r[key] <= n), f'{rid}: {key} exceeds/misses denominator')

            known = [r.get(k) for k in ('successes', 'deaths', 'timeouts')]
            require(n is None or sum(v for v in known if v is not None) <= n, f'{rid}: outcome counts exceed episodes')

            inv, steps = r.get('invalid_actions'), r.get('transitions')
            require(inv is None or (steps is not None and steps > 0 and inv <= steps), f'{rid}: invalid_actions requires a positive transition denominator')

            d = r.get('mean_damage')
            require(d is None or (type(d) in (int, float) and math.isfinite(d) and d >= 0), f'{rid}: invalid mean_damage')

            key = tuple(str(r.get(k)) for k in ('condition', 'split', 'evaluation_mode', 'orientation', 'seed', 'checkpoint', 'sampling_seed'))
            require(key not in dimensions, f'{sid}/{rid}: duplicate comparison dimensions')
            
            dimensions.add(key)
    return document
