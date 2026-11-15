"""Explicit, offline, hash-checked repository/release artifact resolution.

Artifact IDs and producer-relative source references identify evidence.
Files resolve only within explicitly configured roots, without network access
or automatic filesystem discovery.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

SCHEMA_VERSION = 'gradient-gators-reference-bundle-v2'
MAX_FILE_BYTES = 48 * 1024**2


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid_constant(value): raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)


def configured_root(value):
    if not isinstance(value,(str,Path)) or not str(value):
        raise ValueError('An explicit local root is required')
    if str(value).startswith(('\\\\','//')) or '://' in str(value):
        raise ValueError('Network roots are excluded')
    return Path(value).resolve()


def portable_relative(value):
    if not isinstance(value,str) or not value or '\\' in value or ':' in value:
        raise ValueError('Use portable relative paths')
    relative = PurePosixPath(value)
    if (relative.is_absolute() or not relative.parts or relative.as_posix() != value
            or any(part.startswith('.') for part in relative.parts)):
        raise ValueError('Absolute, hidden, ambiguous and traversal paths are excluded')
    return relative


def within(root, relative):
    portable_relative(relative)
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('Artifact path or symlink escapes its configured root')
    return candidate


def file_hash(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('A lowercase file-byte SHA256 is required')
    return value


class BundleIndex:
    def __init__(self, bundle_directory, *, releases_root=None):
        self.root = configured_root(bundle_directory)
        self.releases_root = configured_root(releases_root) if releases_root is not None else None
        manifest_path = within(self.root,'bundle_manifest.json')
        if manifest_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('Oversized bundle manifest')
        self.manifest = strict_json(manifest_path.read_text(encoding='utf-8-sig'))
        if self.manifest.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported reference bundle schema')
        if not isinstance(self.manifest.get('files'),list):
            raise ValueError('Bundle files must be an array')
        self.entries, self.sources = {}, {}
        locations, artifact_keys = set(), set()
        for entry in self.manifest['files']:
            artifact_id = entry.get('artifact_id')
            if not isinstance(artifact_id,str) or not re.fullmatch('[a-z][a-z0-9_]*',artifact_id):
                raise ValueError('Artifact ID must use lowercase snake_case')
            if artifact_id.casefold() in artifact_keys:
                raise ValueError('Duplicate artifact identity')
            artifact_keys.add(artifact_id.casefold())
            storage = entry.get('storage')
            if storage not in ('repository','release'): raise ValueError('Unsupported artifact storage')
            portable_relative(entry.get('relative_path'))
            release_id = entry.get('release_id')
            if storage == 'release':
                if not isinstance(release_id,str) or not re.fullmatch('[a-z][a-z0-9_]*',release_id):
                    raise ValueError('Release ID must use lowercase snake_case')
            elif release_id is not None:
                raise ValueError('Repository artifacts cannot declare a release ID')
            location = (storage,release_id,entry['relative_path'].casefold())
            if location in locations: raise ValueError('Ambiguous storage identity')
            locations.add(location)
            file_hash(entry.get('sha256'))
            if type(entry.get('bytes')) is not int or not 0 <= entry['bytes'] <= MAX_FILE_BYTES:
                raise ValueError('Invalid or oversized artifact length')
            if type(entry.get('source_byte_identical')) is not bool:
                raise ValueError('Source-byte identity must be explicit')
            if not isinstance(entry.get('category'),str) or not entry['category']:
                raise ValueError('Artifact category is required')
            self.entries[artifact_id] = entry
            self._add_source(artifact_id,entry.get('source_ref'), require_identical=entry['source_byte_identical'])
            # Resolve every configured location at construction, and again on read.
            if storage == 'repository' or self.releases_root is not None:
                self.path_for_artifact(artifact_id)
        aliases = self.manifest.get('source_aliases',[])
        if not isinstance(aliases,list): raise ValueError('Source aliases must be an array')
        for alias in aliases:
            artifact_id = alias['artifact_id']
            if artifact_id not in self.entries: raise ValueError('Alias references an unknown artifact')
            self._add_source(artifact_id,alias.get('source_ref'),require_identical=True)

    def _add_source(self, artifact_id, source, *, require_identical):
        if not isinstance(source,dict): raise ValueError('Portable source_ref is required')
        experiment = source.get('experiment_id')
        if not isinstance(experiment,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',experiment):
            raise ValueError('Invalid producer experiment identity')
        portable_relative(source.get('artifact_path'))
        file_hash(source.get('sha256'))
        if require_identical and source['sha256'] != self.entries[artifact_id]['sha256']:
            raise ValueError('Byte-identical source hash differs from artifact hash')
        key = (experiment.casefold(),source['artifact_path'].casefold())
        if key in self.sources: raise ValueError('Ambiguous producer source identity')
        self.sources[key] = (artifact_id,source['sha256'])

    def artifact_for_source(self, experiment_id, artifact_path, sha256=None):
        portable_relative(artifact_path)
        artifact_id, source_hash = self.sources[(experiment_id.casefold(),artifact_path.casefold())]
        if sha256 is not None and source_hash != sha256:
            raise ValueError('Producer source hash differs')
        return artifact_id

    def path_for_artifact(self, artifact_id):
        entry = self.entries[artifact_id]
        if entry['storage'] == 'repository':
            root = self.root
        else:
            if self.releases_root is None:
                raise FileNotFoundError('Release unavailable: explicitly configure releases_root for '+entry['release_id'])
            root = within(self.releases_root,entry['release_id'])
        return within(root,entry['relative_path'])

    def read_bytes(self, artifact_id):
        entry = self.entries[artifact_id]
        path = self.path_for_artifact(artifact_id)
        if path.stat().st_size != entry['bytes']:
            raise ValueError('Indexed artifact size changed')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Indexed artifact hash changed')
        return data

    def read_json(self, artifact_id):
        return strict_json(self.read_bytes(artifact_id).decode('utf-8-sig'))

    def verify_all(self, *, include_releases=True):
        """Verify indexed bytes; this does not certify metrics or replay physics."""
        selected = [key for key,entry in self.entries.items() if include_releases or entry['storage']=='repository']
        total = sum(len(self.read_bytes(key)) for key in selected)
        return dict(files=len(selected),bytes=total,skipped_release_files=len(self.entries)-len(selected))
