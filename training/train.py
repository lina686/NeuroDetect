from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training.dataset import MRISegmentationDataset, collect_image_mask_pairs
from training.model import UNet
from training.utils import (
    bce_dice_loss,
    bce_tversky_loss,
    create_preview_panel,
    dice_per_image_from_logits,
    dice_score_from_logits,
    ensure_dir,
    precision_recall_from_logits,
    save_json,
    search_best_threshold,
    set_seed,
    simple_train_transform,
    split_pairs_by_patient,
    summarize_pairs,
)


DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_ARTIFACT_DIR = ROOT / "models"


@dataclass(slots=True)
class TrainConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    val_size: float = 0.15
    test_size: float = 0.15
    threshold: float = 0.5
    seed: int = 42
    num_workers: int = 0
    demo_case_count: int = 6
    loss_type: str = "bce_tversky"
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    early_stop_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / "unet.pt"

    @property
    def report_path(self) -> Path:
        return self.artifact_dir / "results.json"

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / "run_metadata.json"

    @property
    def demo_manifest_path(self) -> Path:
        return self.artifact_dir / "demo_cases.json"

    @property
    def preview_dir(self) -> Path:
        return self.artifact_dir / "demo_assets"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train NeuroDetect U-Net artifacts.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--demo-case-count", type=int, default=6)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="bce_tversky",
        choices=["bce_dice", "bce_tversky"],
    )
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        test_size=args.test_size,
        threshold=args.threshold,
        seed=args.seed,
        num_workers=args.num_workers,
        demo_case_count=args.demo_case_count,
        loss_type=args.loss_type,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        early_stop_patience=args.early_stop_patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
    )


def _build_loss_fn(config: TrainConfig):
    if config.loss_type == "bce_tversky":
        alpha = config.tversky_alpha
        beta = config.tversky_beta
        return lambda logits, targets: bce_tversky_loss(
            logits, targets, alpha=alpha, beta=beta
        )
    if config.loss_type == "bce_dice":
        return bce_dice_loss
    raise ValueError(f"Unknown loss type: {config.loss_type}")


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    loss_fn,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_dice_sum = 0.0
    running_precision = 0.0
    running_recall = 0.0
    sample_count = 0

    for images, masks in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)
        dice = dice_per_image_from_logits(logits, masks, threshold=threshold)
        precision, recall, _ = precision_recall_from_logits(
            logits, masks, threshold=threshold
        )

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_dice_sum += dice * batch_size
        running_precision += precision * batch_size
        running_recall += recall * batch_size
        sample_count += batch_size

    n = max(sample_count, 1)
    return {
        "loss": running_loss / n,
        "dice": running_dice_sum / n,
        "precision": running_precision / n,
        "recall": running_recall / n,
    }


@torch.no_grad()
def predict_pair(
    model: UNet,
    image_path: Path,
    mask_path: Path,
    image_size: int,
    threshold: float,
    device: torch.device,
) -> dict[str, object]:
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    true_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None or true_mask is None:
        raise ValueError(f"Could not read pair: {image_path} / {mask_path}")

    resized_image = cv2.resize(
        image_gray, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    )
    resized_mask = cv2.resize(
        true_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST
    )
    resized_mask = (resized_mask > 0).astype(np.uint8)

    tensor = torch.from_numpy(resized_image.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
    pred_mask = (probs > threshold).astype(np.uint8)

    target_tensor = torch.from_numpy(resized_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    logits_cpu = logits.detach().cpu()
    dice = dice_score_from_logits(logits_cpu, target_tensor, threshold=threshold)
    precision, recall, _ = precision_recall_from_logits(
        logits_cpu, target_tensor, threshold=threshold
    )

    pred_area_pct = float(pred_mask.mean() * 100.0)
    mean_confidence = float(probs[pred_mask > 0].mean()) if pred_mask.any() else float(probs.mean())

    pred_mask_original = cv2.resize(
        pred_mask * 255,
        (image_gray.shape[1], image_gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    true_mask_original = (true_mask > 0).astype(np.uint8) * 255

    return {
        "image_gray": image_gray,
        "true_mask_original": true_mask_original,
        "pred_mask_original": pred_mask_original,
        "metrics": {
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "pred_area_pct": pred_area_pct,
            "mean_confidence": mean_confidence,
        },
    }


def save_demo_cases(
    config: TrainConfig,
    model: UNet,
    test_pairs: list[tuple[Path, Path]],
    device: torch.device,
) -> list[dict[str, object]]:
    ensure_dir(config.preview_dir)
    ranked_cases: list[dict[str, object]] = []

    for image_path, mask_path in test_pairs:
        prediction = predict_pair(
            model=model,
            image_path=image_path,
            mask_path=mask_path,
            image_size=config.image_size,
            threshold=config.threshold,
            device=device,
        )
        ranked_cases.append(
            {
                "image_path": image_path,
                "mask_path": mask_path,
                "prediction": prediction,
            }
        )

    ranked_cases.sort(
        key=lambda item: (
            item["prediction"]["metrics"]["pred_area_pct"] > 0.25,
            item["prediction"]["metrics"]["dice"],
            item["prediction"]["metrics"]["mean_confidence"],
        ),
        reverse=True,
    )

    manifest: list[dict[str, object]] = []
    for index, item in enumerate(ranked_cases[: config.demo_case_count], start=1):
        image_path = item["image_path"]
        mask_path = item["mask_path"]
        prediction = item["prediction"]
        patient_id = image_path.parent.name
        stem = image_path.stem

        overlay = cv2.cvtColor(prediction["image_gray"], cv2.COLOR_GRAY2BGR)
        overlay[prediction["pred_mask_original"] > 0] = (
            0.35 * overlay[prediction["pred_mask_original"] > 0]
            + 0.65 * np.array([50, 220, 180])
        ).astype(np.uint8)

        preview = create_preview_panel(
            image_gray=prediction["image_gray"],
            true_mask=(prediction["true_mask_original"] > 0).astype(np.uint8),
            pred_mask=(prediction["pred_mask_original"] > 0).astype(np.uint8),
            title=f"{patient_id} / {stem}",
            metrics=prediction["metrics"],
        )

        overlay_path = config.preview_dir / f"{index:02d}_{stem}_overlay.png"
        preview_path = config.preview_dir / f"{index:02d}_{stem}_panel.png"
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(preview_path), preview)

        manifest.append(
            {
                "label": f"Case {index}: {patient_id} / {stem}",
                "patient_id": patient_id,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "overlay_path": str(overlay_path.resolve()),
                "preview_path": str(preview_path.resolve()),
                "metrics": prediction["metrics"],
            }
        )

    save_json(config.demo_manifest_path, {"cases": manifest})
    return manifest


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    if not config.data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {config.data_dir}")

    ensure_dir(config.artifact_dir)
    pairs = collect_image_mask_pairs(config.data_dir)
    train_pairs, val_pairs, test_pairs = split_pairs_by_patient(
        pairs,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.seed,
    )

    print(f"Total pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")

    train_dataset = MRISegmentationDataset(
        train_pairs,
        image_size=config.image_size,
        transform=simple_train_transform,
    )
    val_dataset = MRISegmentationDataset(val_pairs, image_size=config.image_size)
    test_dataset = MRISegmentationDataset(test_pairs, image_size=config.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    loss_fn = _build_loss_fn(config)

    best_val_dice = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn
        )
        val_metrics = evaluate(
            model, val_loader, device, threshold=config.threshold, loss_fn=loss_fn
        )
        scheduler.step(val_metrics["dice"])

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": current_lr,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), config.model_path)
            print(f"Saved best model to {config.model_path}")

        if epoch - best_epoch >= config.early_stop_patience:
            print(
                f"Early stopping: no val_dice improvement for "
                f"{config.early_stop_patience} epochs (best was epoch {best_epoch})."
            )
            break

    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(config.model_path, map_location=device))

    print("Searching best threshold on validation set...")
    best_threshold, best_val_dice_at_threshold = search_best_threshold(
        model, val_loader, device
    )
    print(
        f"Best threshold: {best_threshold:.3f} "
        f"(val dice {best_val_dice_at_threshold:.4f})"
    )
    config.threshold = best_threshold

    test_metrics = evaluate(
        model, test_loader, device, threshold=best_threshold, loss_fn=loss_fn
    )

    print(f"Test loss:      {test_metrics['loss']:.4f}")
    print(f"Test dice:      {test_metrics['dice']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall:    {test_metrics['recall']:.4f}")

    demo_cases = save_demo_cases(config, model, test_pairs, device)

    report_payload = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_threshold": best_threshold,
        "best_val_dice_at_threshold": best_val_dice_at_threshold,
        "test_metrics": test_metrics,
    }
    save_json(config.report_path, report_payload)

    metadata_payload = {
        "project": "NeuroDetect",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "dataset_summary": {
            "overall": summarize_pairs(pairs),
            "train": summarize_pairs(train_pairs),
            "val": summarize_pairs(val_pairs),
            "test": summarize_pairs(test_pairs),
        },
        "artifacts": {
            "model_path": str(config.model_path.resolve()),
            "report_path": str(config.report_path.resolve()),
            "demo_manifest_path": str(config.demo_manifest_path.resolve()),
            "preview_dir": str(config.preview_dir.resolve()),
        },
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_threshold": best_threshold,
        "best_val_dice_at_threshold": best_val_dice_at_threshold,
        "test_metrics": test_metrics,
        "demo_case_count": len(demo_cases),
    }
    save_json(config.metadata_path, metadata_payload)

    print(f"Saved report to {config.report_path}")
    print(f"Saved metadata to {config.metadata_path}")
    print(f"Saved demo manifest to {config.demo_manifest_path}")


if __name__ == "__main__":
    main()
