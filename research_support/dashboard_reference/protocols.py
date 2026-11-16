"""Explicit scientific protocols shared by saved-result readers and metrics."""

LEGACY_PROTOCOL_ID = 'm4-common-ppo-v1'
RECURRENT_PROTOCOL_ID = 'recurrent-complete-900-2m-v1'
FINAL_ROLES = ('FINAL_ID', 'FINAL_OOD_HAZARD', 'FINAL_OOD_OBSTACLE',
               'FINAL_OOD_PURSUIT', 'FINAL_OOD_COMBINED')
REWARDS = ('sparse', 'efficiency', 'safety', 'balanced', 'progress')
PRIMARY_SEEDS = (189707373, 845630960, 1696036709, 3447176590, 1584805801)

_PROTOCOLS = {
    LEGACY_PROTOCOL_ID: {
        'grid': (0, 250_000, 500_000, 750_000, 1_000_000),
        'budget': 1_000_000,
        'train_count': 300,
        'schema_versions': ('research-results-v1', 'm4-results-v1'),
    },
    RECURRENT_PROTOCOL_ID: {
        'grid': (0, 100_000, 250_000, 500_000, 1_000_000, 1_500_000, 2_000_000),
        'budget': 2_000_000,
        'train_count': 900,
        'schema_versions': ('research-results-v2',),
    },
}


def get_protocol(protocol_id=None):
    """Return a protocol copy; only an omitted ID selects historical defaults."""
    if protocol_id is None:
        protocol_id = LEGACY_PROTOCOL_ID
    if not isinstance(protocol_id, str) or protocol_id not in _PROTOCOLS:
        raise ValueError(f'Unsupported scientific protocol: {protocol_id!r}')
    spec = _PROTOCOLS[protocol_id]
    return {
        'id': protocol_id, 'grid': spec['grid'], 'budget': spec['budget'],
        'role_counts': dict(TRAIN=spec['train_count'], DEV=100,
                            LEARNING_VALIDATION=100, **dict.fromkeys(FINAL_ROLES, 200)),
        'primary_seeds': PRIMARY_SEEDS, 'rewards': REWARDS,
        'schema_versions': spec['schema_versions'],
        'required_roles': ('TRAIN', 'LEARNING_VALIDATION', *FINAL_ROLES),
    }


def validate_training_steps(row, protocol_id=None):
    """Check the recurrent nominal coordinate and saved interaction counters."""
    protocol = get_protocol(protocol_id)
    if protocol['id'] == LEGACY_PROTOCOL_ID:
        return
    for field in ('checkpoint_step', 'actual_training_steps', 'optimized_training_steps'):
        if type(row.get(field)) is not int or row[field] < 0:
            raise ValueError(f'{field} must be a nonnegative integer')
    nominal = row['checkpoint_step']
    actual = row['actual_training_steps']
    optimized = row['optimized_training_steps']
    if nominal not in protocol['grid']:
        raise ValueError('checkpoint_step must be a nominal protocol checkpoint')
    maximum = nominal if nominal in (0, protocol['budget']) else nominal + 1398
    if not nominal <= actual <= maximum:
        raise ValueError('actual_training_steps outside the nominal checkpoint window')
    if optimized > actual:
        raise ValueError('optimized_training_steps exceeds actual_training_steps')
    if 'final_unoptimized_steps' in row:
        final = row['final_unoptimized_steps']
        if type(final) is not int or final < 0 or final != actual - optimized:
            raise ValueError('final_unoptimized_steps must reconcile actual and optimized steps')
