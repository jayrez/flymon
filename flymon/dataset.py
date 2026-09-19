"""Immutable framebuffer-sequence datasets; never accesses game memory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .spatiotemporal import SpatiotemporalConfig, spatial_grid, temporal_features

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'datasets/experiment-06'
RESULTS = ROOT / 'results/experiment-06-generalization'
CAPTURES = ROOT / 'captures/experiment-06'
CLASSES = ('bedroom', 'dialogue', 'menu', 'title', 'intro')
SEEDS = tuple(range(201, 221))
SIZES = (1314, 500, 250, 100, 50)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def encode(frames, projection):
    grids = np.stack([spatial_grid(f, projection.config)[1] for f in frames])
    dark, change = temporal_features(grids)
    return np.stack([projection.encode(d, c) for d, c in zip(dark, change)])


def save_instance(label, instance_id, frames, procedure, root=DATASET, role='primary'):
    """Never overwrite an instance; reject shared frames across distinct instances."""
    frames = np.asarray(frames)
    if frames.shape != (10, 144, 160, 4) or frames.dtype != np.uint8:
        raise ValueError('Expected ten RGBA framebuffer samples')
    if label not in CLASSES and role == 'primary':
        raise ValueError(label)
    if Path(instance_id).name != instance_id or Path(label).name != label:
        raise ValueError('Invalid name')
    folder = root / label / instance_id
    hashes = [digest(f.tobytes()) for f in frames]
    if folder.exists():
        raise FileExistsError(folder)
    for path in root.glob('*/*/metadata.json'):
        other = json.loads(path.read_text())
        if set(hashes) & set(other['frame_sha256']):
            raise ValueError(f'Duplicate framebuffer shared with {other["instance_id"]}')
    folder.mkdir(parents=True)
    files = []
    for i, frame in enumerate(frames):
        path = folder / f'frame-{i:02d}.png'
        Image.fromarray(frame).save(path)
        files.append(dict(path=str(path.relative_to(root)), file_sha256=digest(path.read_bytes())))
    meta = dict(class_label=label, instance_id=instance_id, role=role,
                frame_sha256=hashes, sequence_sha256=digest(frames.tobytes()),
                frames=files, capture=procedure)
    write_json(folder / 'metadata.json', meta)
    return meta


def load_dataset(root=DATASET):
    records, sequences = [], []
    seen = set()
    for label in (*CLASSES, 'ood'):
        for path in sorted((root / label).glob('*/metadata.json')):
            meta = json.loads(path.read_text())
            frames = []
            for item, expected in zip(meta['frames'], meta['frame_sha256'], strict=True):
                p = root / item['path']
                if digest(p.read_bytes()) != item['file_sha256']:
                    raise ValueError(f'PNG changed: {p}')
                frame = np.asarray(Image.open(p).convert('RGBA'))
                if digest(frame.tobytes()) != expected:
                    raise ValueError(f'Pixels changed: {p}')
                frames.append(frame)
            frames = np.stack(frames)
            if frames.shape != (10, 144, 160, 4) or digest(frames.tobytes()) != meta['sequence_sha256']:
                raise ValueError('Sequence changed')
            if seen & set(meta['frame_sha256']):
                raise ValueError('Cross-instance duplicate pixels')
            seen.update(meta['frame_sha256'])
            records.append(meta); sequences.append(frames)
    if len({r['instance_id'] for r in records}) != len(records):
        raise ValueError('Duplicate instance ID')
    return records, sequences


def regression(projection):
    """Bit-exact regression against all five original saved encoder sequences."""
    old = json.loads((ROOT / 'results/experiment-03-spatiotemporal/trials.json').read_text())
    if projection.ids.tolist() != old['projection_neuron_indices']:
        raise AssertionError('Projection mapping changed')
    checks = {}
    for name, metrics in old['frame_metrics'].items():
        frames = np.stack([np.asarray(Image.open(ROOT / f'captures/experiment-03/{name}/frame-{i:02d}-original.png').convert('RGBA')) for i in range(10)])
        assert [digest(f.tobytes()) for f in frames] == metrics['frame_sha256']
        actual = encode(frames, projection)
        assert np.array_equal(actual, np.asarray(metrics['dynamic_vectors'], np.float32)), name
        checks[name] = digest(actual.tobytes())
    return checks
