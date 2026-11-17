"""
Train dedicated seven-class FER-2013 facial expression classifiers (Phase 2).
Compares ResNet-18, MobileNet-V2, and EfficientNet-B0.
"""

import os
import sys
import json
import time
import copy
import random
from pathlib import Path
import numpy as np
from PIL import Image

import builtins
_orig_print = builtins.print
builtins.print = lambda *args, **kwargs: _orig_print(*args, **{**kwargs, 'flush': True})

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fer2013"
MODELS_DIR = PROJECT_ROOT / "models" / "facial_expression"
RESULTS_DIR = PROJECT_ROOT / "results" / "facial_expression"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

print("=== Building Dataset Manifests ===")
train_pool_paths = []
train_pool_labels = []
private_test_paths = []
private_test_labels = []
full_test_paths = []
full_test_labels = []

# 1. From data/fer2013/train
train_dir = DATA_DIR / "train"
if train_dir.exists():
    for c in CLASSES:
        cdir = train_dir / c
        if cdir.exists():
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    train_pool_paths.append(str(cdir / fname))
                    train_pool_labels.append(CLASS_TO_IDX[c])

# 2. From data/fer2013/test
test_dir = DATA_DIR / "test"
for c in CLASSES:
    cdir = test_dir / c
    if cdir.exists():
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = str(cdir / fname)
                full_test_paths.append(fpath)
                full_test_labels.append(CLASS_TO_IDX[c])
                if fname.startswith("PublicTest_"):
                    train_pool_paths.append(fpath)
                    train_pool_labels.append(CLASS_TO_IDX[c])
                elif fname.startswith("PrivateTest_"):
                    private_test_paths.append(fpath)
                    private_test_labels.append(CLASS_TO_IDX[c])

print(f"Total training pool images: {len(train_pool_paths)}")
print(f"Untouched PrivateTest images: {len(private_test_paths)}")
print(f"Full FER-2013 Test images: {len(full_test_paths)}")

# Stratified split of training pool into Train (85%) and Val (15%)
from sklearn.model_selection import train_test_split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_pool_paths, train_pool_labels,
    test_size=0.15,
    random_state=SEED,
    stratify=train_pool_labels
)

print(f"Final Train set: {len(train_paths)}")
print(f"Final Val set: {len(val_paths)}")

# Class counts and class weights
train_class_counts = np.bincount(train_labels, minlength=7)
total_train = len(train_labels)
class_weights = total_train / (7.0 * np.maximum(train_class_counts, 1))
class_weights = torch.FloatTensor(class_weights)
print("Class weights for CrossEntropyLoss:", {c: round(float(w), 3) for c, w in zip(CLASSES, class_weights)})

# PyTorch Dataset
class FERDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

IMG_SIZE = 48
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 32
train_dataset = FERDataset(train_paths, train_labels, transform=train_transform)
val_dataset = FERDataset(val_paths, val_labels, transform=val_transform)
test_dataset = FERDataset(private_test_paths, private_test_labels, transform=val_transform)
full_test_dataset = FERDataset(full_test_paths, full_test_labels, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
full_test_loader = DataLoader(full_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
class_weights = class_weights.to(device)

def train_and_evaluate(model_name, model, epochs=12):
    print(f"\n==========================================")
    print(f"Training Architecture: {model_name}")
    print(f"==========================================")
    torch.cuda.empty_cache()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_macro_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 4
    no_improve_epochs = 0

    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Train phase
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())

        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Val phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_bal_acc = balanced_accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        scheduler.step(val_macro_f1)

        dur = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} [{dur:.1f}s]: Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, BalAcc: {val_bal_acc:.3f}, MacroF1: {val_macro_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_bal_acc": val_bal_acc,
            "val_macro_f1": val_macro_f1
        })

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save model checkpoint
    ckpt_path = MODELS_DIR / f"{model_name.lower().replace('-', '_')}_fer2013.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved best model checkpoint to {ckpt_path}")

    # Evaluate on UNTOUCHED PrivateTest
    def evaluate_loader(loader, desc):
        model.eval()
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                p = outputs.argmax(dim=1).detach().cpu().numpy()
                preds.extend(p)
                targets.extend(labels.numpy())
                probs.extend(probabilities)

        preds = np.array(preds)
        targets = np.array(targets)
        probs = np.array(probs)

        acc = accuracy_score(targets, preds)
        bal_acc = balanced_accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(targets, preds, labels=list(range(7))).tolist()

        per_class = {}
        prec = precision_score(targets, preds, average=None, zero_division=0)
        rec = recall_score(targets, preds, average=None, zero_division=0)
        f1_ind = f1_score(targets, preds, average=None, zero_division=0)
        support = np.bincount(targets, minlength=7).tolist()

        for idx, c in enumerate(CLASSES):
            per_class[c] = {
                "support": support[idx],
                "precision": round(float(prec[idx]), 4),
                "recall": round(float(rec[idx]), 4),
                "f1": round(float(f1_ind[idx]), 4)
            }

        print(f"\n--- {desc} Results ---")
        print(f"Accuracy: {acc:.4f}, Balanced Accuracy: {bal_acc:.4f}, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}")
        for c in CLASSES:
            print(f"  {c:10s} Support: {per_class[c]['support']:4d} | Prec: {per_class[c]['precision']:.3f} | Rec: {per_class[c]['recall']:.3f} | F1: {per_class[c]['f1']:.3f}")

        return {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
            "confusion_matrix": cm,
            "classes": CLASSES
        }

    test_results = evaluate_loader(test_loader, "Untouched PrivateTest (N=3,589)")
    full_test_results = evaluate_loader(full_test_loader, "Full FER-2013 Test (N=7,178)")

    return {
        "model_name": model_name,
        "checkpoint": str(ckpt_path),
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "private_test_evaluation": test_results,
        "full_test_evaluation": full_test_results,
        "history": history
    }

all_results = {}

# 1. Model 1: ResNet-18
resnet18 = models.resnet18()
resnet_cached = Path("C:/Users/mh129/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
if resnet_cached.exists():
    state_dict = torch.load(resnet_cached, map_location="cpu", weights_only=True)
    resnet18.load_state_dict(state_dict)
    print("Loaded cached ResNet-18 ImageNet weights.")
resnet18.fc = nn.Linear(512, 7)
all_results["ResNet-18"] = train_and_evaluate("ResNet-18", resnet18, epochs=12)
del resnet18
torch.cuda.empty_cache()

# 2. Model 2: MobileNet-V2
mobilenet = models.mobilenet_v2()
mobilenet_cached = Path("C:/Users/mh129/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth")
if mobilenet_cached.exists():
    state_dict = torch.load(mobilenet_cached, map_location="cpu", weights_only=True)
    mobilenet.load_state_dict(state_dict)
    print("Loaded cached MobileNet-V2 ImageNet weights.")
mobilenet.classifier[1] = nn.Linear(1280, 7)
all_results["MobileNet-V2"] = train_and_evaluate("MobileNet-V2", mobilenet, epochs=12)
del mobilenet
torch.cuda.empty_cache()

# 3. Model 3: EfficientNet-B0 (Scratch initialization)
effnet = models.efficientnet_b0(weights=None)
effnet.classifier[1] = nn.Linear(1280, 7)
print("Initialized EfficientNet-B0 from scratch.")
all_results["EfficientNet-B0"] = train_and_evaluate("EfficientNet-B0", effnet, epochs=12)
del effnet
torch.cuda.empty_cache()

# Save overall results
comparison_file = RESULTS_DIR / "fer2013_deep_models_comparison.json"
with open(comparison_file, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nAll Phase 2 evaluations successfully written to {comparison_file}")
