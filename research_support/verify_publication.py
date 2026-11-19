"""Verify downloaded publication assets and optional saved dashboard evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Expected a portable relative path')
    if PurePosixPath(value).is_absolute() or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Unsafe relative path')
    return value


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if manifest.get('schema_version') != 'frozen-study-publication-v1':
        raise ValueError('Unsupported publication manifest')
    assets = manifest.get('assets')
    if not isinstance(assets, list) or not assets:
        raise ValueError('Publication has no assets')
    seen = set()
    for entry in assets:
        name = safe_name(entry['filename'])
        if '/' in name or name in seen:
            raise ValueError('Asset filenames must be unique basenames')
        seen.add(name)
        if not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Invalid asset checksum')
        if type(entry['bytes']) is not int or entry['bytes'] <= 0:
            raise ValueError('Invalid asset byte count')
        if '/' in safe_name(entry['top_level_folder']):
            raise ValueError('Archive root must be one folder name')
    return manifest


def verify_assets(manifest, assets_root):
    root = Path(assets_root).resolve()
    verified = []
    for entry in manifest['assets']:
        path = root / entry['filename']
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Asset escapes download directory')
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Asset bytes differ: {entry["filename"]}')
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            all_names = [item.filename for item in entries]
            names = [item.filename for item in entries if not item.is_dir()]
            if len(set(all_names)) != len(all_names):
                raise ValueError('Duplicate ZIP member')
            if '/' in safe_name(entry['top_level_folder']):
                raise ValueError('Archive root must be one folder name')
            for item in entries:
                name = item.filename.removesuffix('/') if item.is_dir() else item.filename
                safe_name(name)
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('ZIP symbolic links are not supported')
                if name != entry['top_level_folder'] and not name.startswith(entry['top_level_folder'] + '/'):
                    raise ValueError('Unexpected ZIP top-level directory')
            if archive.testzip() is not None:
                raise ValueError('ZIP member integrity failed')
        verified.append(dict(filename=entry['filename'], sha256=entry['sha256'], files=len(names)))
    return verified


def verify_dashboard(manifest, releases_root, expected_path, include_replay=False):
    from research_support.dashboard_reference.readers.bundle_paths import BundleIndex
    from research_support.dashboard_reference.readers.results_adapter import load_bundle_export
    from research_support.dashboard_reference.readers.replay import load_selected

    root = Path(releases_root).resolve()
    bundle_name = safe_name(manifest['dashboard']['results_bundle'])
    if '/' in bundle_name:
        raise ValueError('Results must be an immediate bundle child')
    selected = root/bundle_name
    if selected.is_symlink() or not selected.resolve().is_relative_to(root):
        raise ValueError('Results bundle escapes selected directory')
    bundle = BundleIndex(selected, releases_root=root)
    bundle.verify_all(include_releases=include_replay)
    study = load_bundle_export(bundle, 'research_results')
    wire = bundle.read_json('research_results')
    expected = json.loads(Path(expected_path).read_text(encoding='utf-8-sig'))
    if expected['study_id'] != manifest['study_id'] or expected['protocol_id'] != manifest['protocol_id']:
        raise ValueError('Reference belongs to another study/protocol')
    if wire['study_id'] != manifest['study_id'] or wire['protocol_id'] != manifest['protocol_id']:
        raise ValueError('Dashboard export belongs to another study/protocol')
    if digest(root/bundle_name/'research_results.json') != expected['dashboard_export_sha256']:
        raise ValueError('Result export differs from accepted evidence')
    if len(wire['models']) != expected['inventory']['checkpoint_models']:
        raise ValueError('Checkpoint inventory differs')
    all_rows = wire['outcomes'] + wire['learning']
    if sum(r['n_completed'] for r in all_rows) != expected['inventory']['unique_cases']:
        raise ValueError('Case inventory differs')
    actual = {(r['source_metrics']['model_id'],r['source_metrics']['role'],r['source_metrics']['nominal_steps']):r['source_metrics'] for r in all_rows}
    oracle = {(r['model_id'],r['role'],r['nominal_steps']):r for r in expected['panels']}
    if len(actual) != len(all_rows) or actual != oracle:
        raise ValueError('Panel metrics differ from the accepted reference')
    for key, oracle_key in [('comparisons','primary_contrasts'),('producer_robustness_models','robustness_models'),('producer_robustness_rewards','robustness_rewards')]:
        if wire[key] != expected[oracle_key]:
            raise ValueError(f'Authoritative scientific fields differ: {key}')
    replay_count = 0
    if include_replay:
        if len(study['results']['traces']) != expected['inventory']['selected_replays']:
            raise ValueError('Replay inventory differs')
        for i in range(len(study['results']['traces'])):
            result = load_selected(study, i, bundle=bundle)
            if not result['available']:
                raise ValueError(result.get('reason_unavailable', 'Replay unavailable'))
            replay_count += 1
    return dict(panels=len(actual), unique_cases=sum(r['n_completed'] for r in all_rows), verified_saved_replays=replay_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--dashboard-root', type=Path)
    parser.add_argument('--include-replay', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if not args.assets and not args.dashboard_root:
        parser.error('Supply --assets, --dashboard-root, or both')
    if args.include_replay and not args.dashboard_root:
        parser.error('--include-replay requires --dashboard-root')
    manifest = load_manifest(args.manifest)
    receipt = dict(status='passed', study_id=manifest['study_id'], model_or_environment_execution=False)
    if args.assets:
        receipt['assets'] = verify_assets(manifest, args.assets)
    if args.dashboard_root:
        manifest_root = args.manifest.resolve().parent
        expected = manifest_root / safe_name(manifest['expected_results_file'])
        if expected.is_symlink() or not expected.resolve().is_relative_to(manifest_root):
            raise ValueError('Expected-results file escapes manifest directory')
        receipt['dashboard'] = verify_dashboard(manifest,args.dashboard_root,expected,args.include_replay)
    text = json.dumps(receipt,indent=2)+'\n'
    if args.receipt:
        # A verification receipt is a new output, never an overwritten evidence file.
        with args.receipt.open('x',encoding='utf-8') as stream: stream.write(text)
    print(text,end='')


if __name__ == '__main__': main()
