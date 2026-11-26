"""Plan or execute the approved V4 official test protocol; dry-run by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_HASH = 'ca12f33e158e30c08279ff8be5bd5b2a69be75e688a01eee59195fc3f198cb06'
CALIBRATION_HASH = 'c04ebcc0534cf814f660be6357479d44c948a65f6daa0e8d631274915201d0e4'
MANIFEST_HASH = 'a0666d0cb771f7ce8ee0f45600d099b68d412bf8bf76cfeddca25c7fbc3b0b33'
CONFIGURATIONS = ('dense', 'unmasked', *(f'{v}_{t}' for t in (25, 50) for v in 'ABCD'))


def raw_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')).hexdigest()


def check_inputs(checkpoint, calibration, manifest):
    hashes = {'checkpoint_sha256': raw_hash(checkpoint),
              'calibration_canonical_sha256': canonical_hash(calibration),
              'manifest_canonical_sha256': canonical_hash(manifest)}
    for key, expected in zip(hashes, (CHECKPOINT_HASH, CALIBRATION_HASH, MANIFEST_HASH)):
        if hashes[key] != expected:
            raise ValueError(f'Frozen input mismatch: {key}')
    return hashes


def commands(checkpoint, data_root, calibration, manifest, output, workers=0, download=False):
    common = [sys.executable, '-m', 'sparse_pet.evaluate', '--checkpoint', str(checkpoint),
              '--data-root', str(data_root), '--manifest', str(manifest), '--split', 'test',
              '--frozen-config', str(calibration), '--batch-size', '16', '--workers', str(workers)]
    if download:
        common += ['--download']
    plan = []
    for label in CONFIGURATIONS:
        selector = ['--variant', label] if label in ('dense', 'unmasked') else ['--label', label]
        plan.append(common + selector + ['--output', str(output / f'{label}.json'),
                                        '--predictions-output', str(output / f'{label}.csv')])
    analysis = [sys.executable, '-m', 'sparse_pet.paired_analysis', '--input-dir', str(output),
                '--split', 'test', '--configurations', *CONFIGURATIONS, '--repetitions', '10000',
                '--seed', '42', '--output', str(output / 'paired_analysis.json')]
    return plan, analysis


def check_approval(execute, approval, approved_revision, revision, dirty):
    if execute:
        if not approval or not approval.strip() or not approved_revision or approved_revision != revision:
            raise ValueError('Execution requires a team approval reference and exact approved full Git SHA matching HEAD')
        if dirty:
            raise ValueError('Execution requires a clean Git working tree')


def check_result(payload, label, selected):
    if payload['split'] != 'test' or payload['scores']['n'] != 3669:
        raise ValueError(f'{label}: incomplete or wrong split')
    if payload['variant'] != label.split('_')[0]:
        raise ValueError(f'{label}: wrong variant')
    if payload['layer_indices'] != (None if label == 'dense' else [3, 6, 9]):
        raise ValueError(f'{label}: wrong layer indices')
    if label in ('dense', 'unmasked'):
        if payload['mask_config'] is not None:
            raise ValueError(f'{label}: unexpected mask')
    else:
        c = selected[label]
        expected = {'method': 'top_p' if label[0] in 'AB' else 'hybrid', 'p': c['p'],
                    'top_k': c['top_k'], 'similarity_threshold': c['similarity_threshold'],
                    'grid_size': 14, 'tile_size': 2}
        if payload['mask_config'] != expected:
            raise ValueError(f'{label}: unexpected mask settings')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def runtime_metadata(allow_cpu):
    # This only constructs an untrained model to resolve its preprocessing; no inference.
    import torch
    import torchvision
    import timm
    import numpy
    import sklearn
    from timm.data import resolve_model_data_config, create_transform
    if not torch.cuda.is_available() and not allow_cpu:
        raise ValueError('CUDA unavailable. Use a GPU or explicitly pass --allow-cpu.')
    model = timm.create_model('deit_tiny_patch16_224.fb_in1k', pretrained=False, num_classes=37)
    cfg = resolve_model_data_config(model)
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'torchvision': torchvision.__version__, 'timm': timm.__version__, 'numpy': numpy.__version__,
            'scikit_learn': sklearn.__version__, 'cuda_runtime': torch.version.cuda,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'default_dtype': str(torch.get_default_dtype()),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'eval_data_config': cfg, 'eval_transform': repr(create_transform(**cfg, is_training=False))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path, help='Fresh directory outside the Git repository')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--download', action='store_true', help='Allow dataset download on execution only')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--approved-revision', help='Full approved Git commit SHA')
    p.add_argument('--approval-reference', help='Team approval date and PR comment/reference')
    p.add_argument('--allow-cpu', action='store_true')
    args = p.parse_args()
    checkpoint, data_root, output = (x.expanduser().resolve() for x in (args.checkpoint, args.data_root, args.output_dir))
    if args.workers < 0:
        p.error('workers must be nonnegative')
    if output == ROOT or ROOT in output.parents:
        p.error('Use an output directory outside the repository')
    calibration, manifest = (ROOT / 'outputs' / f for f in ('calibration_64_v4.json', 'split_manifest.json'))
    hashes = check_inputs(checkpoint, calibration, manifest)
    revision, dirty = git('rev-parse', 'HEAD'), git('status', '--porcelain')
    check_approval(args.execute, args.approval_reference, args.approved_revision, revision, dirty)
    plan, analysis = commands(checkpoint, data_root, calibration, manifest, output, args.workers, args.download)
    record = {'revision': revision, 'working_tree_clean': not bool(dirty), 'input_hashes': hashes,
              'checkpoint': str(checkpoint), 'data_root': str(data_root), 'output_dir': str(output),
              'batch_size': 16, 'workers': args.workers, 'configurations': list(CONFIGURATIONS),
              'evaluation_commands': plan, 'analysis_command': analysis}
    if not args.execute:
        print(json.dumps(record, indent=2))
        print('PREFLIGHT PASS: no model inference, dataset download or output files created.')
        return
    if output.exists():
        raise FileExistsError('Choose a fresh output directory; existing runs are never overwritten')
    # Imports use the checked repository, even when launched from another directory.
    sys.path.insert(0, str(ROOT))
    from sparse_pet.paired_analysis import read_run
    record.update(approval_reference=args.approval_reference, runtime=runtime_metadata(args.allow_cpu),
                  started_utc=datetime.now(timezone.utc).isoformat(), status='running', completed=[])
    output.mkdir(parents=True, exist_ok=False)
    log = output / 'run_provenance.json'
    def save():
        log.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    selected = json.loads(calibration.read_text())['selected']
    save()
    try:
        for label, cmd in zip(CONFIGURATIONS, plan):
            check_inputs(checkpoint, calibration, manifest)
            check_approval(True, args.approval_reference, args.approved_revision, git('rev-parse', 'HEAD'), git('status', '--porcelain'))
            print(f'Evaluating {label}', flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
            payload = json.loads((output / f'{label}.json').read_text())
            check_result(payload, label, selected)
            if payload['device'] != record['runtime']['device']:
                raise ValueError(f'{label}: device differs from preflight')
            _, file_hashes = read_run(output, label, 'test')
            record['completed'].append({'configuration': label, 'output_sha256': file_hashes})
            save()
        subprocess.run(analysis, cwd=ROOT, check=True)
        check_inputs(checkpoint, calibration, manifest)
        check_approval(True, args.approval_reference, args.approved_revision, git('rev-parse', 'HEAD'), git('status', '--porcelain'))
        record['paired_analysis_sha256'] = raw_hash(output / 'paired_analysis.json')
        record['status'] = 'complete'
    except BaseException as exc:
        record['status'] = 'failed'
        record['error'] = repr(exc)
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print(f'OFFICIAL TEST WORKFLOW COMPLETE: {output}')


if __name__ == '__main__':
    main()
