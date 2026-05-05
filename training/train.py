from __future__ import annotations

import sys
from pathlib import Path

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
    dice_score_from_logits,
    precision_recall_from_logits,  # NEU eingefügt 22.04
    ensure_parent_dir,
    set_seed,
    simple_train_transform,
    split_pairs_by_patient,
)


DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "unet.pt"

IMAGE_SIZE = 256
BATCH_SIZE = 8 # Hier war 8, kleinere batch size kann zu Stabilität führen
EPOCHS = 20 #hier kam vorhin 15 rein für richtiges training
LR = 1e-4
SEED = 42


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:  # ← geändert
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_precision = 0.0  # NEU
    running_recall = 0.0     # NEU

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        dice = dice_score_from_logits(logits, masks)
        precision, recall, _ = precision_recall_from_logits(logits, masks)  # NEU

        running_loss += loss.item() * images.size(0)
        running_dice += dice * images.size(0)
        running_precision += precision * images.size(0)  # NEU
        running_recall += recall * images.size(0)        # NEU

    n = len(loader.dataset)
    return (
        running_loss / n,
        running_dice / n,
        running_precision / n,  # NEU
        running_recall / n,     # NEU
    )


def main() -> None:
    set_seed(SEED)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    pairs = collect_image_mask_pairs(DATA_DIR)
    train_pairs, val_pairs, test_pairs = split_pairs_by_patient(pairs)

    print(f"Total pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")

    train_dataset = MRISegmentationDataset(
        train_pairs, image_size=IMAGE_SIZE, transform=simple_train_transform
    )
    val_dataset = MRISegmentationDataset(val_pairs, image_size=IMAGE_SIZE)
    test_dataset = MRISegmentationDataset(test_pairs, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    #hier war vorhin das hier zu testen drinnen
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR) --> jetzt weight decay hinzugefügt
    # hilft gegen Overfitting

    best_val_dice = -1.0
    train_loss_history = []  # NEU
    val_loss_history = []  # NEU
    ensure_parent_dir(MODEL_PATH)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice, val_precision, val_recall = evaluate(model, val_loader, device)

        # NEU eingefügt am 22.04
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_precision={val_precision:.4f} | "
            f"val_recall={val_recall:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH}")

 #Das auch NEU eingefügt 22.04
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loss, test_dice, test_precision, test_recall = evaluate(model, test_loader, device)

    print(f"Test loss:      {test_loss:.4f}")
    print(f"Test dice:      {test_dice:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall:    {test_recall:.4f}")

    import json
    results = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "test_dice": test_dice,
        "test_precision": test_precision,
        "test_recall": test_recall,
    }
    results_path = ROOT / "models" / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    print("\nLoading best model for final training evaluation...")


if __name__ == "__main__":
    main()