# -*- coding: utf-8 -*-
"""Unified evaluation entry for the current ROI/Kaggle mainline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from data_combined import get_combined_loaders
from model_unet import load_model
from utils_metrics import compute_all_metrics, summarize_topology_results


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(config: Dict[str, Any]) -> str:
    requested = config.get('training', {}).get('device', 'cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    return requested


def infer_data_mode(config: Dict[str, Any]) -> Tuple[str, str]:
    use_kaggle = bool(config.get('data', {}).get('use_kaggle_combined', False))
    if use_kaggle:
        return 'roi_kaggle_combined', 'data_combined.get_combined_loaders -> KaggleCombinedDataset'
    return 'drive_legacy_compat', 'data_combined.get_combined_loaders -> data_drive.get_drive_loaders'


def resolve_checkpoint_path(config: Dict[str, Any], checkpoint_path: Optional[str]) -> Tuple[Path, str]:
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path)
        if not resolved.exists():
            raise FileNotFoundError(f'Checkpoint not found: {resolved}')
        return resolved, 'CLI --checkpoint'

    checkpoint_dir = Path(config.get('training', {}).get('checkpoint_dir', './checkpoints'))
    use_kaggle = bool(config.get('data', {}).get('use_kaggle_combined', False))
    topo_enabled = not bool(config.get('topology', {}).get('disable_topo_loss', False))

    candidates = []
    if use_kaggle:
        if topo_enabled:
            candidates.extend([
                'best_model_topo_roi.pth',
                'best_model_baseline_roi.pth',
            ])
        else:
            candidates.extend([
                'best_model_baseline_roi.pth',
                'best_model_topo_roi.pth',
            ])
    else:
        if topo_enabled:
            candidates.extend([
                'best_model_topo.pth',
                'best_model.pth',
            ])
        else:
            candidates.extend([
                'best_model.pth',
                'best_model_baseline_roi.pth',
            ])

    for fallback_name in [
        'best_model_topo_roi.pth',
        'best_model_baseline_roi.pth',
        'best_model_topo.pth',
        'best_model.pth',
        'final_model_topo_roi.pth',
        'final_model_topo.pth',
    ]:
        if fallback_name not in candidates:
            candidates.append(fallback_name)

    for candidate_name in candidates:
        candidate_path = checkpoint_dir / candidate_name
        if candidate_path.exists():
            return candidate_path, f'auto-resolved from {checkpoint_dir} ({candidate_name})'

    raise FileNotFoundError(
        'Unable to auto-resolve checkpoint. Tried: ' + ', '.join(str(checkpoint_dir / name) for name in candidates)
    )


def unpack_batch(batch: Any, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        images, vessels, rois = batch
    else:
        images = batch['image']
        vessels = batch['vessel']
        rois = batch['roi']

    images = images.to(device)
    vessels = vessels.to(device)
    rois = rois.to(device)
    if rois.dim() == 3:
        rois = rois.unsqueeze(1)
    return images, vessels, rois


def dataset_has_labels(dataset: Any, split: str) -> bool:
    if split != 'test':
        return True
    if getattr(dataset, 'mode', None) == 'test':
        return False
    return True


def sample_name(dataset: Any, global_index: int, batch_index: int, sample_index: int) -> str:
    image_ids = getattr(dataset, 'image_ids', None)
    if image_ids is not None and global_index < len(image_ids):
        image_id = image_ids[global_index]
        if isinstance(image_id, (int, np.integer)):
            return f'{int(image_id):02d}'
        return str(image_id)

    image_files = getattr(dataset, 'image_files', None)
    if image_files is not None and global_index < len(image_files):
        return Path(image_files[global_index]).stem

    return f'b{batch_index:03d}_s{sample_index:02d}'


def prepare_image_for_display(image: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
    image_np = image.detach().cpu().numpy()
    if image_np.ndim == 3 and image_np.shape[0] == 1:
        display = np.clip(image_np[0] * 0.5 + 0.5, 0.0, 1.0)
        return display, {'cmap': 'gray'}

    if image_np.ndim == 3 and image_np.shape[0] >= 3:
        display = np.moveaxis(image_np[:3], 0, -1)
        display = np.clip(display * 0.5 + 0.5, 0.0, 1.0)
        return display, {}

    if image_np.ndim == 2:
        return np.clip(image_np, 0.0, 1.0), {'cmap': 'gray'}

    if image_np.ndim == 3:
        return np.clip(image_np[0], 0.0, 1.0), {'cmap': 'gray'}

    raise ValueError(f'Unsupported image shape for visualization: {image_np.shape}')


def fmt_metric(value: Optional[float], digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return 'nan'
    return f'{float(value):.{digits}f}'


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    compute_topology: bool,
    threshold: float,
    save_predictions: bool,
    output_dir: Path,
    has_labels: bool,
) -> Dict[str, Any]:
    model.eval()
    dataset = dataloader.dataset

    basic_sums = {
        'dice': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
    }
    topology_results = []
    num_samples = 0

    pred_dir = output_dir / 'predictions'
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(dataloader, desc='Evaluate')):
            images, vessels, rois = unpack_batch(batch, device)
            outputs = model(images)

            for sample_index in range(images.shape[0]):
                pred_np = torch.sigmoid(outputs[sample_index, 0]).detach().cpu().numpy()
                roi_np = rois[sample_index, 0].detach().cpu().numpy()
                name = sample_name(dataset, num_samples, batch_index, sample_index)

                if has_labels:
                    target_np = vessels[sample_index, 0].detach().cpu().numpy()
                    metrics = compute_all_metrics(
                        pred_np,
                        target_np,
                        roi_np,
                        threshold=threshold,
                        compute_topology=compute_topology,
                    )
                    basic_sums['dice'] += metrics.dice
                    basic_sums['iou'] += metrics.iou
                    basic_sums['precision'] += metrics.precision
                    basic_sums['recall'] += metrics.recall

                    if compute_topology:
                        topology_results.append({
                            'cl_break': metrics.cl_break,
                            'delta_beta0': metrics.delta_beta0,
                            'pred_beta0': metrics.pred_beta0,
                            'target_beta0': metrics.target_beta0,
                            'pred_fragments': metrics.pred_fragments,
                            'target_fragments': metrics.target_fragments,
                            'valid': metrics.topology_valid,
                            'error': metrics.topology_message,
                        })

                if save_predictions:
                    np.save(pred_dir / f'{name}_pred.npy', pred_np.astype(np.float32))

                num_samples += 1

    results: Dict[str, Any] = {
        'num_samples': num_samples,
        'has_labels': has_labels,
        'topology_valid_count': 0,
        'topology_invalid_count': 0,
        'cl_break': float('nan'),
        'delta_beta0': float('nan'),
        'pred_beta0': float('nan'),
        'target_beta0': float('nan'),
    }

    if has_labels and num_samples > 0:
        for key, value in basic_sums.items():
            results[key] = value / num_samples
    else:
        for key in basic_sums:
            results[key] = float('nan')

    if compute_topology and topology_results:
        topo_summary = summarize_topology_results(topology_results)
        results.update({
            'cl_break': topo_summary['cl_break'],
            'delta_beta0': topo_summary['delta_beta0'],
            'pred_beta0': topo_summary['pred_beta0'],
            'target_beta0': topo_summary['target_beta0'],
            'topology_valid_count': topo_summary['valid_count'],
            'topology_invalid_count': topo_summary['invalid_count'],
        })

    return results


def visualize_results(
    model: torch.nn.Module,
    dataset: Any,
    device: str,
    has_labels: bool,
    threshold: float,
    num_samples: int,
    save_path: Path,
) -> None:
    model.eval()
    num_samples = min(num_samples, len(dataset))
    num_columns = 4 if has_labels else 3
    column_titles = ['Image', 'ROI', 'Target', 'Prediction'] if has_labels else ['Image', 'ROI', 'Prediction']

    fig, axes = plt.subplots(num_samples, num_columns, figsize=(4.5 * num_columns, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for idx in range(num_samples):
            sample = dataset[idx]
            if isinstance(sample, (list, tuple)) and len(sample) == 3:
                image, vessel, roi = sample
            else:
                image = sample['image']
                vessel = sample['vessel']
                roi = sample['roi']

            output = model(image.unsqueeze(0).to(device))
            pred_np = torch.sigmoid(output[0, 0]).detach().cpu().numpy()
            roi_np = roi[0].detach().cpu().numpy() if roi.dim() == 3 else roi.detach().cpu().numpy()
            pred_roi = pred_np * roi_np
            image_display, image_kwargs = prepare_image_for_display(image)

            for col_idx, title in enumerate(column_titles):
                axes[idx, col_idx].set_title(title)
                axes[idx, col_idx].axis('off')

            axes[idx, 0].imshow(image_display, **image_kwargs)
            axes[idx, 1].imshow(roi_np, cmap='gray', vmin=0, vmax=1)

            if has_labels:
                target_np = vessel[0].detach().cpu().numpy() if vessel.dim() == 3 else vessel.detach().cpu().numpy()
                axes[idx, 2].imshow(target_np * roi_np, cmap='gray', vmin=0, vmax=1)
                axes[idx, 3].imshow(pred_roi, cmap='gray', vmin=0, vmax=1)
            else:
                axes[idx, 2].imshow((pred_roi > threshold).astype(np.float32), cmap='gray', vmin=0, vmax=1)

            axes[idx, 0].set_ylabel(sample_name(dataset, idx, idx, 0))

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main(config_path: str = 'config.yaml', checkpoint_path: Optional[str] = None, split: str = 'val') -> None:
    config = load_config(config_path)
    device = resolve_device(config)
    data_mode, loader_source = infer_data_mode(config)
    resolved_checkpoint, checkpoint_source = resolve_checkpoint_path(config, checkpoint_path)

    train_loader, val_loader, test_loader = get_combined_loaders(config)
    loader_map = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    loader = loader_map.get(split)
    if loader is None:
        raise ValueError(f'Split "{split}" is not available for the current data mode: {data_mode}')

    has_labels = dataset_has_labels(loader.dataset, split)
    compute_topology = bool(config.get('metrics', {}).get('compute_topology', True)) and has_labels
    threshold = float(config.get('metrics', {}).get('topology_threshold', 0.5))

    print('=' * 72)
    print(f'Config:        {config_path}')
    print(f'Checkpoint:    {resolved_checkpoint} [{checkpoint_source}]')
    print(f'Data mode:     {data_mode}')
    print(f'Loader source: {loader_source}')
    print(f'Split:         {split}')
    print(f'Device:        {device}')
    print(f'Has labels:    {has_labels}')
    print('=' * 72)

    model = load_model(resolved_checkpoint, device)

    results_dir = Path('results') / f'evaluate_{split}'
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_model(
        model=model,
        dataloader=loader,
        device=device,
        compute_topology=compute_topology,
        threshold=threshold,
        save_predictions=True,
        output_dir=results_dir,
        has_labels=has_labels,
    )

    print('\nEvaluation Summary')
    print('-' * 72)
    print(f'Samples:      {metrics["num_samples"]}')
    if has_labels:
        print(f'Dice:         {fmt_metric(metrics.get("dice"))}')
        print(f'IoU:          {fmt_metric(metrics.get("iou"))}')
        print(f'Precision:    {fmt_metric(metrics.get("precision"))}')
        print(f'Recall:       {fmt_metric(metrics.get("recall"))}')
        if compute_topology:
            print(f'CL-Break:     {fmt_metric(metrics.get("cl_break"), digits=1)}')
            print(f'铻栧熬閳р偓:          {fmt_metric(metrics.get("delta_beta0"), digits=1)}')
            print(f'pred_beta0:   {fmt_metric(metrics.get("pred_beta0"), digits=1)}')
            print(f'target_beta0: {fmt_metric(metrics.get("target_beta0"), digits=1)}')
            print(
                f'Topology valid/invalid: {metrics.get("topology_valid_count", 0)}/'
                f'{metrics.get("topology_invalid_count", 0)}'
            )
    else:
        print('Metrics:      skipped because this split has no vessel labels.')
    print(f'Predictions:  {results_dir / "predictions"}')

    visualize_results(
        model=model,
        dataset=loader.dataset,
        device=device,
        has_labels=has_labels,
        threshold=threshold,
        num_samples=min(4, len(loader.dataset)),
        save_path=results_dir / 'evaluation_results.png',
    )
    print(f'Visualization:{results_dir / "evaluation_results.png"}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate the current ROI/Kaggle mainline checkpoint.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Explicit checkpoint path.')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Split to evaluate.')
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.split)
