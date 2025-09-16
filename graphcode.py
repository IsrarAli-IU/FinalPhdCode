# train_all_and_plot.py
# Trains BERT, FABERT, Hierarchical_Bert, and RABERT on lcd.csv
# and saves: accuracy_vs_epoch.png/pdf, loss_vs_epoch.png/pdf

import os, math, random
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = "lcd.csv"          # keep in same folder
EPOCHS   = 10                 # adjust as needed
BATCH    = 16
LR       = 2e-5
MAX_LEN  = 128
SEED     = 42
OUTDIR   = "./plots"
os.makedirs(OUTDIR, exist_ok=True)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- REPRO ----------
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ---------- DATASET ----------
class RowToTextDataset(Dataset):
    """
    Generic dataset: turns each row (except label col) into a compact text string.
    Works across all four models so they share the same split.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, label_col: str = "LargeClass"):
        assert label_col in df.columns, f"'{label_col}' not in CSV"
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_col = label_col

        # Keep only numeric/usable cols to avoid super long strings
        self.feature_cols = [c for c in df.columns if c != label_col]
        # Shorten names a bit in text
        self._alias = {c: c.replace("_", " ").lower() for c in self.feature_cols}

    def _row_to_text(self, row: pd.Series) -> str:
        # Compact "k:v" comma separated
        parts = []
        for c in self.feature_cols:
            v = row[c]
            # coerce NaN -> 0 / "" safely
            if isinstance(v, (int, float)) and (math.isnan(v) if isinstance(v, float) else False):
                v = 0
            parts.append(f"{self._alias[c]}: {v}")
        return ", ".join(parts)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = self._row_to_text(row)
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        label = int(row[self.label_col])
        return {
            "input_ids":        enc["input_ids"].squeeze(0),
            "attention_mask":   enc["attention_mask"].squeeze(0),
            "label":            torch.tensor(label, dtype=torch.long),
            "raw_row":          row,      # for hierarchical fallback
            "text":             text
        }

# ---------- MODELS ----------
# 1) Plain BERT baseline (CLS -> linear)
class BERTClassifier(nn.Module):
    def __init__(self, pretrained="bert-base-uncased", num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.fc   = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.fc(cls)

# 2) FABERT / 3) Hierarchical_Bert / 4) RABERT
# Import your provided architectures (place this script next to those files).
# If you renamed files, update these imports accordingly.
from feature_aware_bert import FeatureAwareBERT          # FABERT
from hierarchical_bert_ import HierarchicalBERT          # Hierarchical_Bert
from relation_aware_bert import RelationAwareBERT        # RABERT

# ---------- TRAIN/EVAL HELPERS ----------
def pub_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3
    })

@torch.no_grad()
def eval_one_epoch(model, loader, device, model_name, tokenizer):
    model.eval()
    y_true, y_pred, val_losses = [], [], []
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        if model_name == "Hierarchical_Bert":
            # Try to build two short texts from likely columns; if absent, fall back to the same text twice.
            row = batch["raw_row"]
            # Build strings from common metric names if present
            # (safe defaults; adapt if your csv has different column names)
            def getv(col, default=0):
                try: 
                    v = row[col].item() if hasattr(row[col], "item") else row[col]
                    return 0 if (isinstance(v,float) and math.isnan(v)) else v
                except Exception:
                    return default
            code_txt = f"loc: {getv('loc')}, wmc: {getv('wmc',0)}, cbo: {getv('cbo',0)}, comments: {getv('comments',0)}, volume: {getv('volume',0)}"
            hals_txt = f"effort: {getv('effort',0)}, difficulty: {getv('difficulty',0)}, bugs: {getv('bugs',0)}"
            code_enc = tokenizer(code_txt, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            hals_enc = tokenizer(hals_txt, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

            code_ids  = code_enc["input_ids"].to(device)
            code_att  = code_enc["attention_mask"].to(device)
            hals_ids  = hals_enc["input_ids"].to(device)
            hals_att  = hals_enc["attention_mask"].to(device)
            labels    = batch["label"].to(device)

            logits = model(code_ids, code_att, hals_ids, hals_att)
        else:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            # RABERT needs relation_ids; FABERT and BERT do not.
            if model_name == "RABERT":
                # Minimal placeholder (single relation id = 0) to match your provided class
                relation_ids = torch.zeros(input_ids.size(0), dtype=torch.long, device=device)
                logits = model(input_ids, attention_mask, relation_ids)
            else:
                logits = model(input_ids, attention_mask)

        loss = ce(logits, labels)
        val_losses.append(loss.item())
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    val_loss = float(np.mean(val_losses)) if val_losses else np.nan
    val_acc  = accuracy_score(y_true, y_pred) if len(y_true) else np.nan
    return val_loss, val_acc

def train_model(model, train_loader, val_loader, device, model_name, tokenizer, epochs=EPOCHS, lr=LR):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        ep_losses = []
        for batch in train_loader:
            optim.zero_grad()
            if model_name == "Hierarchical_Bert":
                row = batch["raw_row"]
                def getv(col, default=0):
                    try:
                        v = row[col].item() if hasattr(row[col], "item") else row[col]
                        return 0 if (isinstance(v,float) and math.isnan(v)) else v
                    except Exception:
                        return default
                code_txt = f"loc: {getv('loc')}, wmc: {getv('wmc',0)}, cbo: {getv('cbo',0)}, comments: {getv('comments',0)}, volume: {getv('volume',0)}"
                hals_txt = f"effort: {getv('effort',0)}, difficulty: {getv('difficulty',0)}, bugs: {getv('bugs',0)}"
                code_enc = tokenizer(code_txt, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
                hals_enc = tokenizer(hals_txt, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

                code_ids  = code_enc["input_ids"].to(device)
                code_att  = code_enc["attention_mask"].to(device)
                hals_ids  = hals_enc["input_ids"].to(device)
                hals_att  = hals_enc["attention_mask"].to(device)
                labels    = batch["label"].to(device)

                logits = model(code_ids, code_att, hals_ids, hals_att)
            else:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                if model_name == "RABERT":
                    relation_ids = torch.zeros(input_ids.size(0), dtype=torch.long, device=device)
                    logits = model(input_ids, attention_mask, relation_ids)
                else:
                    logits = model(input_ids, attention_mask)

            loss = ce(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_losses.append(loss.item())

        val_loss, val_acc = eval_one_epoch(model, val_loader, device, model_name, tokenizer)
        history["train_loss"].append(float(np.mean(ep_losses)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[{model_name}] Epoch {ep:02d}/{epochs} | train_loss={history['train_loss'][-1]:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
    return history

# ---------- MAIN ----------
def main():
    # Load data
    df = pd.read_csv(CSV_PATH)
    assert "LargeClass" in df.columns, "Expected 'LargeClass' column (0/1 labels) in lcd.csv"

    # Split once to share across models
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["LargeClass"]
    )
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = RowToTextDataset(train_df, tok, MAX_LEN)
    val_ds   = RowToTextDataset(val_df, tok, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # Build models
    models: Dict[str, nn.Module] = {
        "BERT":               BERTClassifier(),
        "FABERT":             FeatureAwareBERT(pretrained_model="bert-base-uncased", feature_dim=16, num_labels=2),
        "Hierarchical_Bert":  HierarchicalBERT(pretrained_model="bert-base-uncased", num_labels=2),
        "RABERT":             RelationAwareBERT(pretrained_model="bert-base-uncased", relation_dim=16, num_labels=2),
    }

    # Train each and collect histories
    histories: Dict[str, Dict[str, list]] = {}
    for name, mdl in models.items():
        print(f"\n===== Training {name} =====")
        histories[name] = train_model(mdl, train_loader, val_loader, DEVICE, name, tok, epochs=EPOCHS, lr=LR)

    # Plot: Accuracy vs Epoch & Loss vs Epoch (validation curves)
    pub_style()
    xs = range(1, EPOCHS + 1)

    # Accuracy
    plt.figure(figsize=(7, 4.5))
    for name in models.keys():
        if histories[name]["val_acc"]:
            plt.plot(xs, histories[name]["val_acc"], linewidth=2, label=name)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch")
    plt.legend(frameon=False, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "accuracy_vs_epoch.png"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "accuracy_vs_epoch.pdf"), bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure(figsize=(7, 4.5))
    for name in models.keys():
        if histories[name]["val_loss"]:
            plt.plot(xs, histories[name]["val_loss"], linewidth=2, label=name)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
    plt.legend(frameon=False, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "loss_vs_epoch.png"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "loss_vs_epoch.pdf"), bbox_inches="tight")
    plt.close()

    print(f"\nSaved plots in: {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()
