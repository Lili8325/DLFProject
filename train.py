import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.cuda.amp as amp
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle
from model import SubmissionModel
from torchvision import transforms
from caption_dataset import CrossModalDataset

batch_size = 16
num_epochs = 15
lr_backbone = 1e-5
lr_head = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train_one_epoch(model, loader, optimizer, scaler, epoch: int | None = None):
    model.train()
    losses = []

    desc = f"Training Epoch {epoch + 1}" if epoch is not None else "Training"
    with tqdm(loader, desc=desc, unit="batch") as pbar:
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast():
                logits = model(
                    batch["image"].to(device),
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device)
                )

                labels = batch["label"].to(device).float().view(-1)
                logits = logits.view(-1)

                loss = F.binary_cross_entropy_with_logits(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, epoch: int | None = None):
    model.eval()

    preds, labels = [], []
    desc = f"Evaluating Epoch {epoch + 1}" if epoch is not None else "Evaluating"
    for batch in tqdm(loader, desc=desc, unit="batch"):
        logits = model(
            batch["image"].to(device),
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )

        probs = torch.sigmoid(logits)
        preds.extend(probs.cpu().numpy())
        labels.extend(batch["label"].numpy())

    preds_bin = (np.array(preds) > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds_bin),
        "f1": f1_score(labels, preds_bin),
        "auc": roc_auc_score(labels, preds)
    }


def train():
    print("Loading dataset from dataset.pkl...")
    with open("dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    print(f"Loaded dataset with {len(dataset)} samples.")

    print("Splitting dataset into train and validation sets...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train set: {train_size} samples, Validation set: {val_size} samples.")

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train loader: {len(train_loader)} batches, Validation loader: {len(val_loader)} batches.")

    print("Initializing model and optimizer...")
    model = SubmissionModel().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": lr_backbone},
        {"params": model.cross_attn.parameters(), "lr": lr_head},
        {"params": model.classifier.parameters(), "lr": lr_head},
    ])
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None
    # scaler = amp.GradScaler()
    print("Model, optimizer, and GradScaler initialized. Using mixed precision (float16).")

    best_auc = 0.0
    UNFREEZE_EPOCH = 2
    N_ROBERTA_LAYERS = 2
    ROBERTA_LR = 1e-5
    roberta_unfrozen = False

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        if epoch == UNFREEZE_EPOCH and not roberta_unfrozen:
            print(f"Unfreezing last {N_ROBERTA_LAYERS} RoBERTa layers")

            model.text_encoder.unfreeze_last_layers(N_ROBERTA_LAYERS)

            optimizer.add_param_group({
                "params": filter(
                    lambda p: p.requires_grad,
                    model.text_encoder.parameters()
                ),
                "lr": ROBERTA_LR,
            })

            roberta_unfrozen = True

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, epoch)
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), "weights.pth")
            print("Saved best model (weights.pth)")

    print("Training complete. Best AUC:", best_auc)


if __name__ == "__main__":
    train()