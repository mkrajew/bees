"""Backfill checkpoint_eval.json for an already-completed training run, by
scanning its checkpoint directory directly -- for runs that finished under
the older train.py, which collected checkpoint paths via ModelCheckpoint's
in-memory best_k_models/last_model_path instead of a directory scan, and on
config 10a's actual run only saw 1 of the 6 files that existed on disk (root
cause not fully pinned down; a network-filesystem quirk interacting with
Lightning's own bookkeeping is the leading suspect, since this runs over an
NFS-mounted home directory -- see train.py's comment at the same spot).

Reuses the exact same per-checkpoint evaluation logic as train.py's own
loop (our held-out test set + DeepWings' published one), just standalone
so it can be pointed at a checkpoint directory after the fact without
re-training.

Usage:
    uv run python wings/modeling/backfill_checkpoint_eval.py \
        wings/modeling/training/lightning-checkpoints/unet-400-online-augmentation-k5-10a
"""
import argparse
import json
from pathlib import Path

import torch
import torch.utils.data as data

from wings.config import PROCESSED_DATA_DIR, COUNTRIES
from wings.dataset import build_mask_datasets, generate_circular_landmark_mask
from wings.transforms import TrainAugmentConfig
from wings.modeling.litnet import LitNet, compute_statistics
from wings.modeling.unet import UNet
from wings.modeling.loss import BCEDiceLoss
from wings.deepwings_eval import evaluate_checkpoint_on_deepwings


def main(checkpoint_dir: Path, sigmoid: bool = False) -> None:
    mean_coords = torch.load(
        PROCESSED_DATA_DIR / "mask_datasets" / "rectangle" / "mean_shape.pth",
        weights_only=False,
    )

    # Only the test split matters here -- train_augment_cfg is irrelevant to
    # it (TransformedMaskDataset uses build_eval_transform for val/test
    # regardless of what augmentation config is passed).
    _, _, test_dataset = build_mask_datasets(
        countries=COUNTRIES,
        data_folder=PROCESSED_DATA_DIR / "cropped",
        output_size=400,
        square_size=7,
        train_augment_cfg=TrainAugmentConfig(),
        mask_fn=generate_circular_landmark_mask,
    )
    test_dataloader = data.DataLoader(test_dataset, batch_size=12, num_workers=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = sorted(str(p) for p in checkpoint_dir.glob("*.ckpt"))
    print(f"Found {len(checkpoint_paths)} checkpoints in {checkpoint_dir}")
    for p in checkpoint_paths:
        print(f"  {p}")

    results = []
    for ckpt_path in checkpoint_paths:
        entry = {"checkpoint": str(ckpt_path)}

        try:
            eval_model = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=sigmoid)
            eval_lit_net = LitNet.load_from_checkpoint(
                ckpt_path,
                model=eval_model,
                criterion=BCEDiceLoss(),
                num_epochs=1,
                mean_coords=mean_coords,
                strict=False,
            )
            eval_lit_net.eval()
            eval_lit_net.to(device)

            error_distances, wrong_spot_count = [], []
            with torch.no_grad():
                for bx, _, bcoords, (bx_size, by_size) in test_dataloader:
                    output = eval_lit_net.model(bx.to(device))
                    dists, wrong = compute_statistics(
                        output=output,
                        coords=bcoords,
                        x_size=bx_size,
                        y_size=by_size,
                        mean_coords=mean_coords,
                        output_is_logits=not eval_model.sigmoid,
                    )
                    error_distances.extend(dists)
                    wrong_spot_count.extend(wrong)

            distances = torch.tensor(error_distances)
            entry["our_test_mean_error_px"] = (
                distances.mean().item() if len(distances) else float("nan")
            )
            entry["our_test_median_error_px"] = (
                distances.median().item() if len(distances) else float("nan")
            )
            entry["our_test_wrong_spot_count_pct"] = (
                torch.tensor(wrong_spot_count).mean().item() * 100.0
                if wrong_spot_count
                else float("nan")
            )

            del eval_lit_net, eval_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Our-test-set evaluation failed for {ckpt_path}: {e!r}")

        try:
            dw_result = evaluate_checkpoint_on_deepwings(ckpt_path, mean_coords, sigmoid=sigmoid)
            entry["deepwings_precision_mean"] = dw_result["precision_mean"]
            entry["deepwings_precision_median"] = dw_result["precision_median"]
            entry["deepwings_reliable_pct"] = dw_result["reliable_pct"]
            entry["deepwings_n_samples"] = dw_result["n_samples"]
        except Exception as e:
            print(f"DeepWings evaluation failed for {ckpt_path}: {e!r}")

        results.append(entry)
        print(f"Done [{ckpt_path}]: {entry}")

    report_path = checkpoint_dir / "checkpoint_eval.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path, help="Directory containing the .ckpt files")
    parser.add_argument(
        "--sigmoid",
        action="store_true",
        help="Pass for configs 8a/8b/8c (WeightedDiceLoss); every other config in this "
        "series uses BCEDiceLoss and needs sigmoid=False (the default).",
    )
    args = parser.parse_args()
    main(args.checkpoint_dir, sigmoid=args.sigmoid)
