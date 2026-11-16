"""Export a completed recurrent campaign to portable dashboard records.

Only saved JSON is read. Scientific statistics are forwarded from the producer;
this module does not execute policies, environments, training or evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from ..protocols import RECURRENT_PROTOCOL_ID, get_protocol, validate_training_steps
from .bundle_paths import SCHEMA_VERSION, portable_relative, strict_json, within
from .results_adapter import adapt, require


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def engine_digest(value):
    # The report and native case recorder have distinct canonical Unicode encodings.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def byte_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return strict_json(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def model_id(job, step):
    return f'{job}__t{step}'


def build_transport(report, datasets, selection, *, study_id, provenance, fixture=False, native_job_hashes=None):
    """Translate the complete native report, retaining the producer's statistics."""
    protocol = get_protocol(RECURRENT_PROTOCOL_ID)
    require(re.fullmatch('[a-z][a-z0-9_]*', study_id) is not None, 'Use a lowercase snake_case study ID')
    require(report.get('status') == 'complete' and report.get('scientific_results_emitted') is True,
            'A complete scientific report is required')
    require(report.get('unique_evaluation_cases') == 65000 and report.get('logical_training_interactions') == 50000000,
            'Unexpected campaign coverage')
    expected = {(reward, seed) for reward in protocol['rewards'] for seed in protocol['primary_seeds']}
    native_models = report['models']
    require(len(native_models) == 25 and {(m['reward'], m['seed']) for m in native_models} == expected,
            'Expected five reward families by five primary seeds')
    by_job = {m['model_id']: m for m in native_models}
    require(len(by_job) == 25, 'Duplicate native model identity')
    if native_job_hashes is not None:
        require(set(native_job_hashes) == set(by_job), 'Native job hash inventory differs')
    models, checkpoints = [], {}
    for m in native_models:
        cps = m['checkpoints']
        require(len(cps) == 7 and {c['nominal_training_steps'] for c in cps} == set(protocol['grid']),
                'Incomplete native checkpoint inventory')
        for cp in cps:
            step = cp['nominal_training_steps']
            item = dict(model_id=model_id(m['model_id'], step), source_model_id=m['model_id'],
                        reward=m['reward'], seed=m['seed'], slot=protocol['primary_seeds'].index(m['seed']),
                        checkpoint_step=step, actual_training_steps=cp['actual_training_steps'],
                        optimized_training_steps=cp['optimized_training_steps'], checkpoint_sha256=cp['sha256'],
                        source_job_sha256=(native_job_hashes or {}).get(m['model_id'], m['job_sha256']),
                        producer_report_job_sha256=m['job_sha256'])
            validate_training_steps(item, RECURRENT_PROTOCOL_ID)
            models.append(item)
            checkpoints[m['model_id'], step] = item

    def row(point):
        job, step = point['model_id'], point['nominal_steps']
        require(job in by_job and (job, step) in checkpoints, 'Unknown report checkpoint')
        m = checkpoints[job, step]
        role = 'TRAIN' if point['role'] == 'TRAIN900' else point['role']
        require(point['reward'] == m['reward'] and point['seed'] == m['seed'], 'Report model dimensions differ')
        require(point['actual_steps'] == m['actual_training_steps'] and point['optimized_steps'] == m['optimized_training_steps'],
                'Report/checkpoint step counters differ')
        require(point['n'] == protocol['role_counts'][role] and point['complete'] is True,
                'Incomplete scientific panel')
        require(point['success_fraction'] == point['success_count']/point['n'], 'Success count/rate differs')
        item = dict(model_id=m['model_id'], role=role, checkpoint_step=step,
                    actual_training_steps=m['actual_training_steps'], optimized_training_steps=m['optimized_training_steps'],
                    n_expected=point['n'], n_completed=point['n'], success_count=point['success_count'],
                    death_count=point['death_count'], timeout_count=point['timeout_count'],
                    mean_damage=point.get('damage_mean'), mean_length=point.get('length_mean'),
                    mean_length_success=point.get('length_success_mean'), mean_length_failure=point.get('length_failure_mean'),
                    transitions=point['evaluation_steps'], source_metrics=copy.deepcopy(point))
        return item

    outcomes = [row(p) for p in report['outcomes']]
    require(len(outcomes) == 150, 'Expected TRAIN plus five final panels per policy')
    curves = report['learning_curves']
    require(len(curves) == 25 and {c['model_id'] for c in curves} == set(by_job), 'Incomplete learning-curve inventory')
    for curve in curves:
        native = by_job[curve['model_id']]
        require(curve['reward'] == native['reward'] and curve['seed'] == native['seed'] and
                all(p['model_id'] == curve['model_id'] for p in curve['points']), 'Learning-curve model identity differs')
    learning = [row(p) for curve in curves for p in curve['points']]
    require(len(learning) == 175, 'Expected seven learning panels per policy')
    require(len(report['primary_contrasts']) == 4 and {c['reward'] for c in report['primary_contrasts']} == set(protocol['rewards'][1:]),
            'Expected the four prespecified shaped-minus-sparse contrasts')
    require(len(report['robustness_models']) == 25 and len(report['robustness_rewards']) == 5,
            'Incomplete producer robustness assessment')
    require({r['model_id'] for r in report['robustness_models']} == set(by_job) and
            {r['reward'] for r in report['robustness_rewards']} == set(protocol['rewards']),
            'Producer robustness identities differ')
    for contrast in report['primary_contrasts']:
        require(contrast.get('reference') == 'sparse' and contrast.get('n_pairs') == 5,
                'Primary comparison requires five paired seeds against sparse')
    traces = []
    for selected in selection['selected']:
        m = checkpoints[selected['job_id'], protocol['budget']]
        require(selected['reward'] == m['reward'] and selected['seed'] == m['seed'] and
                selected['checkpoint_step'] == protocol['budget'] and selected['checkpoint_sha256'] == m['checkpoint_sha256'],
                'Selected trace does not join its final model')
        traces.append(dict(model_id=m['model_id'], role=selected['role'], scenario_id=selected['scenario_id'],
                           selection_category=selected['selection_category'], available=False,
                           reason_unavailable='Optional saved replay assets are not included in this results bundle.',
                           replay_format='recurrent-primary-v1', source_job_id=selected['job_id'],
                           source_job_sha256=m['source_job_sha256'], checkpoint_sha256=m['checkpoint_sha256'],
                           evaluation_seed=m['seed']))
    result = dict(schema_version='research-results-v2', protocol_id=RECURRENT_PROTOCOL_ID, study_id=study_id,
                  title=study_id.replace('_', ' ').title(), status='complete', fixture=fixture,
                  evaluation_mode='deterministic', provenance=provenance, datasets=datasets, models=models,
                  outcomes=outcomes, learning=learning, comparisons=copy.deepcopy(report['primary_contrasts']), traces=traces,
                  limitations=copy.deepcopy(report['limitations']),
                  metric_units={'success': 'fraction', 'damage': 'HP', 'training_steps': 'consumed interactions',
                                'raw_return': 'original reward units; compare only within the same reward definition'},
                  trace_selection={k: copy.deepcopy(v) for k, v in selection.items() if k != 'selected'},
                  producer_learning_metrics=copy.deepcopy(curves), producer_robustness_models=copy.deepcopy(report['robustness_models']),
                  producer_robustness_rewards=copy.deepcopy(report['robustness_rewards']),
                  compute_accounting={'logical_training_interactions': report['logical_training_interactions'],
                                      'unique_evaluation_cases': report['unique_evaluation_cases'],
                                      'note': report.get('consumed_compute_note')})
    # The reader validates duplicate joins, complete coverage and all role-specific counts.
    parsed = adapt(result, 'research_results.json', '0'*64)
    require(parsed['results']['inventory']['complete'], 'Incomplete required recurrent campaign panels')
    return result


def export_campaign(source, destination, *, study_id, include_replay=False):
    """Create separate results/worlds bundle folders after controller completion.

    A fresh destination is required. Interrupted exports retain a partial sibling
    directory for inspection; they never replace a previously accepted release.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(not destination.exists(), 'Choose a new export destination')
    require(not destination.is_relative_to(source) and not source.is_relative_to(destination),
            'Export and source directories must be separate')
    state = read(source/'state/campaign.json')
    require(state.get('status') == 'complete', 'Campaign is still running or incomplete; no export created')
    report_path = source/'results/RESEARCH_RESULTS.json'
    report = read(report_path)
    matrix, contract = read(source/'matrix.json'), read(source/'contract.json')
    require(report['matrix_sha256'] == byte_hash(source/'matrix.json') and report['contract_sha256'] == byte_hash(source/'contract.json'),
            'Report configuration hashes differ')
    selection = read(source/'results/selected_traces.json')
    replay_receipt = read(source/'results/replay_verification/receipt.json')
    require(replay_receipt.get('status') == 'passed' and replay_receipt['selection_sha256'] == byte_hash(source/'results/selected_traces.json'),
            'Selected replay verification is missing or inconsistent')
    case_index = read(source/'results/scientific_case_index.json')
    require(case_index['count'] == len(case_index['cases']) == 65000 and
            len({r['path'] for r in case_index['cases']}) == 65000, 'Incomplete unique-case index')
    for case in case_index['cases']:
        portable_relative(case['path'].replace('\\', '/'))
    require(contract['method'] == 'recurrent_2m_complete900', 'Unsupported native producer method')
    jobs = {j['id']: j for j in matrix['jobs']}
    require(len(jobs) == 25, 'Unexpected native job inventory')
    require({m['model_id'] for m in report['models']} == set(jobs), 'Report/matrix policy inventory differs')
    for m in report['models']:
        require(digest(jobs[m['model_id']]) == m['job_sha256'], 'Report/matrix job content differs')
    first = matrix['jobs'][0]

    def locate(raw):
        # Old producer roots are provenance only. Resolve the known data/results
        # suffix inside the explicitly supplied source root, including relocation.
        value = str(raw).replace('\\', '/')
        if ':' in value or value.startswith('/'):
            matches = [value.index('/'+part+'/') for part in ('data', 'results') if '/'+part+'/' in value]
            require(len(matches) == 1, 'Producer path has no unambiguous data/results suffix')
            value = value[matches[0]+1:]
        return within(source, value)

    locations = {'TRAIN': locate(first['train_dir']), 'LEARNING_VALIDATION': locate(first['dev_dir']),
                 **{role: locate(path) for role, path in first['final_suites'].items()}}
    for job in jobs.values():
        require(locate(job['train_dir']) == locations['TRAIN'] and locate(job['dev_dir']) == locations['LEARNING_VALIDATION'] and
                all(locate(job['final_suites'][role]) == locations[role] for role in first['final_suites']),
                'Policies use different evaluation collections')
    datasets = []
    for role, directory in locations.items():
        manifest = read(directory/'manifest.json')
        datasets.append(dict(role=role, scenario_count=len(manifest['entries']),
                             manifest_path=f'datasets/{role.lower()}/manifest.json', manifest_sha256=byte_hash(directory/'manifest.json')))
    provenance = {'producer': 'recurrent_primary', 'source_report': 'evidence/producer_results.json',
                  'source_report_sha256': byte_hash(report_path), 'source_matrix_sha256': report['matrix_sha256'],
                  'source_contract_sha256': report['contract_sha256'], 'source_seal_sha256': state['seal_sha256'],
                  'validation_scope': 'Completed producer report, coverage/joins, configuration hashes and selected replay receipt; no raw-case metric recomputation or model execution.'}
    document = build_transport(report, datasets, selection, study_id=study_id, provenance=provenance,
                               native_job_hashes={job_id: engine_digest(job) for job_id, job in jobs.items()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name+'.partial-', dir=destination.parent))
    bundle_id, world_id = study_id+'_results', study_id+'_worlds'
    bundle, worlds = staging/bundle_id, staging/world_id
    bundle.mkdir()
    files, indexed = [], {}

    def add(src, relative, category, *, world=False):
        portable_relative(relative)
        storage = 'release' if world else 'repository'
        location = (storage, relative)
        if location in indexed:
            return indexed[location]
        target = (worlds if world else bundle)/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        artifact = 'a_'+hashlib.sha256((storage+':'+relative).encode()).hexdigest()[:24]
        entry = dict(artifact_id=artifact, storage=storage, relative_path=relative, category=category,
                     bytes=target.stat().st_size, sha256=byte_hash(target), source_byte_identical=True,
                     source_ref=dict(experiment_id=study_id, artifact_path=Path(src).relative_to(source).as_posix(), sha256=byte_hash(src)))
        if world: entry['release_id'] = world_id
        files.append(entry)
        indexed[location] = artifact
        return artifact

    add(report_path, 'evidence/producer_results.json', 'producer_results')
    add(source/'results/scientific_case_index.json', 'evidence/scientific_case_index.json', 'case_index')
    add(source/'results/selected_traces.json', 'evidence/selected_traces.json', 'trace_selection')
    add(source/'results/replay_verification/receipt.json', 'evidence/replay_receipt.json', 'replay_verification')
    for ds in datasets:
        add(locations[ds['role']]/'manifest.json', ds['manifest_path'], 'collection_manifest')
    if include_replay:
        for selected, trace in zip(selection['selected'], document['traces']):
            job = jobs[selected['job_id']]
            require(engine_digest(job) == trace['source_job_sha256'], 'Native job content differs from export')
            case_path = locate(selected['case_path'])
            trace_path = locate(selected['trajectory_path'])
            world_path = within(locations[selected['role']], selected['scenario_id']+'.json')
            trace.update(available=True, reason_unavailable=None,
                         case_artifact_id=add(case_path, 'replay/'+case_path.relative_to(source).as_posix(), 'case'),
                         trajectory_artifact_id=add(trace_path, 'replay/'+trace_path.relative_to(source).as_posix(), 'trace'),
                         snapshot_artifact_id=add(world_path, 'worlds/'+selected['role'].lower()+'/'+selected['scenario_id']+'.json', 'world', world=True),
                         collection_artifact_id=indexed['repository', f"datasets/{selected['role'].lower()}/manifest.json"])
    write(bundle/'research_results.json', document)
    normalized_hash = byte_hash(bundle/'research_results.json')
    files.append(dict(artifact_id='research_results', storage='repository', relative_path='research_results.json',
                      category='research_results', bytes=(bundle/'research_results.json').stat().st_size,
                      sha256=normalized_hash, source_byte_identical=False,
                      source_ref=dict(experiment_id=study_id, artifact_path='exports/research_results.json', sha256=normalized_hash)))
    write(bundle/'bundle_manifest.json', dict(schema_version=SCHEMA_VERSION, bundle_id=bundle_id, files=files))
    from .bundle_paths import BundleIndex
    from .results_adapter import load_bundle_export
    index = BundleIndex(bundle, releases_root=staging)
    index.verify_all()
    study = load_bundle_export(index, 'research_results')
    require(study['results']['inventory']['complete'], 'Incomplete exported campaign inventory')
    if include_replay:
        from .replay import load_selected
        for i in range(len(study['results']['traces'])):
            loaded = load_selected(study, i, bundle=index)
            require(loaded['available'], loaded.get('reason_unavailable', 'Replay join failed'))
    for directory in (bundle, worlds) if include_replay else (bundle,):
        write(directory/'release_manifest.json', dict(schema_version=1, bundle_id=directory.name,
              files=[{'path': p.relative_to(directory).as_posix(), 'bytes': p.stat().st_size, 'sha256': byte_hash(p)}
                     for p in sorted(directory.rglob('*')) if p.is_file()]))
    write(staging/'EXPORT_RECEIPT.json', dict(status='passed', study_id=study_id, protocol_id=RECURRENT_PROTOCOL_ID,
          results_bundle=bundle_id, worlds_bundle=world_id if include_replay else None,
          scientific_cases=65000, replay_selections=len(document['traces']), replay_included=include_replay,
          source_report_sha256=byte_hash(report_path), output_sha256=normalized_hash))
    # Publish the complete directory in one filesystem rename; no existing output is overwritten.
    require(not destination.exists(), 'Export destination appeared during preparation')
    staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Completed campaign directory')
    parser.add_argument('--output', type=Path, required=True, help='New output directory, outside the campaign')
    parser.add_argument('--study-id', required=True)
    parser.add_argument('--include-replay', action='store_true', help='Include saved cases/traces and a separate selected-worlds bundle')
    args = parser.parse_args()
    print(export_campaign(args.source, args.output, study_id=args.study_id, include_replay=args.include_replay))


if __name__ == '__main__':
    main()
