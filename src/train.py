"""Multi-task training loop for CAFN.

Trains the fused cross-attention head jointly with the two auxiliary
unimodal heads (`CAFN.multi_task_loss`). Callable as a library function
(`train_model`) for tests/notebooks, or as a script:

    python -m src.train --epochs 20 --synthetic-samples 64
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset

from src.config import Config, DEFAULT_CONFIG
from src.data.dataset import SyntheticNeuroDataset
from src.models.cafn import CAFN


def train_val_split(dataset, val_split: float, seed: int) -> tuple[Subset, Subset]:
    n = len(dataset)
    n_val = max(1, int(n * val_split)) if n > 1 else 0
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if not train_idx:
        train_idx, val_idx = perm, []
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


@torch.no_grad()
def evaluate(model: CAFN, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        mri = batch["mri"].to(device)
        vep = batch["vep"].to(device)
        labels = batch["label"].to(device)
        outputs = model(mri, vep)
        loss, _ = model.multi_task_loss(outputs, labels)
        preds = outputs["logits_fused"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
    if total == 0:
        return {"loss": float("nan"), "accuracy": float("nan")}
    return {"loss": loss_sum / total, "accuracy": correct / total}


def train_model(
    model: CAFN,
    train_dataset,
    val_dataset=None,
    cfg: Config = DEFAULT_CONFIG,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict:
    """Runs the multi-task training loop. Returns per-epoch history."""
    device = device or torch.device("cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss, n_samples = 0.0, 0
        for batch in train_loader:
            mri = batch["mri"].to(device)
            vep = batch["vep"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(mri, vep)
            loss, _ = model.multi_task_loss(outputs, labels, aux_weight=cfg.train.aux_loss_weight)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        train_loss = epoch_loss / max(n_samples, 1)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            history["val_loss"].append(metrics["loss"])
            history["val_accuracy"].append(metrics["accuracy"])
            if verbose:
                print(
                    f"epoch {epoch + 1}/{cfg.train.epochs} "
                    f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
                    f"val_acc={metrics['accuracy']:.4f}"
                )
        elif verbose:
            print(f"epoch {epoch + 1}/{cfg.train.epochs} train_loss={train_loss:.4f}")

    return history


def save_checkpoint(model: CAFN, path: str, cfg: Config = DEFAULT_CONFIG) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_config": cfg.model}, path)


def load_checkpoint(path: str, cfg: Config = DEFAULT_CONFIG) -> CAFN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CAFN(checkpoint.get("model_config", cfg.model))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train CAFN on synthetic or manifest data")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.train.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.train.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.train.lr)
    parser.add_argument("--synthetic-samples", type=int, default=64)
    parser.add_argument("--checkpoint-name", type=str, default="cafn.pt")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.train.seed)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = Config(
        train=DEFAULT_CONFIG.train.__class__(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=DEFAULT_CONFIG.train.weight_decay,
            aux_loss_weight=DEFAULT_CONFIG.train.aux_loss_weight,
            val_split=DEFAULT_CONFIG.train.val_split,
            seed=args.seed,
            checkpoint_dir=DEFAULT_CONFIG.train.checkpoint_dir,
        )
    )

    dataset = SyntheticNeuroDataset(num_samples=args.synthetic_samples, cfg=cfg, seed=args.seed)
    train_ds, val_ds = train_val_split(dataset, cfg.train.val_split, cfg.train.seed)

    model = CAFN(cfg.model)
    history = train_model(model, train_ds, val_ds, cfg=cfg, verbose=True)

    checkpoint_path = os.path.join(cfg.train.checkpoint_dir, args.checkpoint_name)
    save_checkpoint(model, checkpoint_path, cfg)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
