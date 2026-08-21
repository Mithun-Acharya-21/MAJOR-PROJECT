"""Fine-tune the Phase 5 text architecture on frozen train/validation splits only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from training.text.baseline import assert_no_participant_overlap
from training.text.deberta_model import DebertaBiLSTMAttentionClassifier, DebertaTextConfig
from training.text.preprocess import normalize_text

VARIANTS = {
    "deberta_mean": {"use_bilstm": False, "use_attention": False},
    "deberta_attention": {"use_bilstm": False, "use_attention": True},
    "deberta_bilstm_mean": {"use_bilstm": True, "use_attention": False},
    "deberta_bilstm_attention": {"use_bilstm": True, "use_attention": True},
}


class TextRecords(Dataset):
    def __init__(self, path: Path, expected_split: str, label_to_id: dict[str, int] | None = None) -> None:
        if "test" in path.name.casefold():
            raise ValueError("final test data is intentionally unavailable during fine-tuning")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        needed = {"record_id", "participant_id", "label", "text"}
        if not rows or not needed <= set(rows[0]):
            raise ValueError(f"{path} must contain {sorted(needed)}")
        if any(row.get("split") and row["split"].strip() != expected_split for row in rows):
            raise ValueError(f"{path} has rows outside {expected_split!r}")
        self.texts = [normalize_text(row["text"]) for row in rows]
        self.participants = [row["participant_id"].strip() for row in rows]
        labels = [row["label"].strip() for row in rows]
        self.label_to_id = label_to_id or {label: index for index, label in enumerate(sorted(set(labels)))}
        if set(labels) - set(self.label_to_id) or len(self.label_to_id) != 2:
            raise ValueError("binary train/validation labels must be identical")
        self.labels = [self.label_to_id[label] for label in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def collate(tokenizer, max_length: int):
    def _collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch, strict=True)
        encoded = tokenizer(list(texts), max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded
    return _collate


@torch.no_grad()
def validation_f1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    actual, predicted = [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        outputs = model(**{key: value.to(device) for key, value in batch.items()})
        actual.extend(labels.cpu().tolist())
        predicted.extend(outputs["logits"].argmax(dim=1).cpu().tolist())
    return float(f1_score(actual, predicted, average="binary", zero_division=0))


def train(train_csv: Path, validation_csv: Path, output_dir: Path, config: DebertaTextConfig, seed: int = 42, variant: str = "deberta_bilstm_attention") -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose one of {sorted(VARIANTS)}")
    config = replace(config, **VARIANTS[variant])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_data = TextRecords(train_csv, "train")
    validation_data = TextRecords(validation_csv, "validation", train_data.label_to_id)
    assert_no_participant_overlap(train_data.participants, validation_data.participants)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = DebertaBiLSTMAttentionClassifier(config).to(device)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate(tokenizer, config.max_length))
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate(tokenizer, config.max_length))
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, math.floor(total_steps * config.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    best_f1, stale_epochs = -1.0, 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**{key: value.to(device) for key, value in batch.items()})["logits"]
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer); scaler.update(); scheduler.step()
        score = validation_f1(model, validation_loader, device)
        if score > best_f1:
            best_f1, stale_epochs = score, 0
            torch.save({"state_dict": model.state_dict(), "config": config.to_dict(), "label_to_id": train_data.label_to_id}, output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    result = {"variant": variant, "best_validation_f1": best_f1, "epochs_completed": epoch, "device": str(device), "label_to_id": train_data.label_to_id}
    (output_dir / "training-summary.json").write_text(json.dumps({"config": asdict(config), **result}, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa + BiLSTM + attention on a frozen text split.")
    parser.add_argument("--train", type=Path, required=True); parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="deberta_bilstm_attention")
    args = parser.parse_args()
    config = DebertaTextConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    print(json.dumps(train(args.train, args.validation, args.output_dir, config, variant=args.variant), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
