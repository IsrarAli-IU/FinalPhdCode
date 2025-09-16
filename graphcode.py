# train_transformers_and_plot_no_bert.py
# Trains FABERT, Hierarchical_BERT, and RABERT (no plain BERT) on lcd.csv
# Produces: plots/accuracy_vs_epoch.(png|pdf), plots/loss_vs_epoch.(png|pdf)

import os, math, random
from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = "lcd.csv"          # put in same folder
EPOCHS   = 10                 # adjust as needed
BATCH    = 16
LR       = 2e-5
MAX_LEN  = 128
SEED     = 42
OUTDIR   = "./plots"
os.makedirs(OUTDIR, exist_ok=True)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- MODELS (your files) ----------
# FeatureAwareBERT: forward(input_ids, attention_mask)
from feature_aware_bert import FeatureAwareBERT          # FABERT  :contentReference[oaicite:3]{index=3}
# HierarchicalBERT: forward(code_inputs, code_attention, halstead_inputs, halstead_attention)
from hierarchical_bert_ import HierarchicalBERT          # Hierarchical_BERT  :contentReference[oaicite:4]{index=4}
# RelationAwareBERT: forward(input_ids, attention_mask, relation_ids)
from relation_aware_bert import RelationAwareBERT        # RABERT  :contentReference[oaicite:5]{index=5}

# ---------- REPRO ----------
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ---------- DATASETS ----------
class FlatTextDataset(Dataset):
    """
    For FABERT & RABERT: serialize each row (except label) into a compact text string and tokenize.
    Returns only tensors/ints (no pandas objects).
    """
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, label_col: str = "LargeClass"):
        assert label_col in df.columns, f"'{label_col}' not found in CSV"
        self.df = df.reset_index(drop=True).copy()
        self.tok = tokenizer
        self.max_len = max_len
        self.label_col = label_col
        self.feature_cols = [c for c in self.df.columns if c != label_col]
        # prebuild alias names (short, lowercase, spaces instead of underscores)
        self.alias = {c: c.replace("_", " ").lower() for c in self.feature_cols}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        parts = []
        for c in self.feature_cols:
            v = row[c]
            if isinstance(v, float) and (pd.isna(v)):
                v = 0
            parts.append(f"{self.alias[c]}: {v}")
        text = ", ".join(parts)

        enc = self.tok(text, max_length=self.max_len, padding="max_length",
                       truncation=True, return_tensors="pt")

        label = int(row[self.label_col])
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }

class HierarchicalDataset(Dataset):
    """
    For Hierarchical_BERT: build two short texts (code metrics & halstead group).
    Missing columns default to 0.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, label_col: str = "LargeClass"):
        assert label_col in df.columns, f"'{label_col}' not found in CSV"
        self.df = df.reset_index(drop=True).copy()
        self.tok = tokenizer
        self.max_len = max_len
        self.label_col = label_col

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]

        def getv(k, d=0):
            if k not in r.index: return d
            v = r[k]
            if isinstance(v, float) and pd.isna(v): return d
            return v

        # Build two segments (tweak these feature picks if your CSV uses different names)
        code_txt = f"loc: {getv('loc')}, wmc: {getv('wmc')}, cbo: {getv('cbo')}, comments: {getv('comments')}, volume: {getv('volume')}"
        hals_txt = f"effort: {getv('effort')}, difficulty: {getv('difficulty')}, bugs: {getv('bugs')}"

        code = self.tok(code_txt, max_length=self.max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        hals = self.tok(hals_txt, max_length=self.max_len, padding="max_length",
                        truncation=True, return_tensors="pt")

        label = int(r[self.label_col])
        return {
            "code_input_ids":          code["input_ids"].squeeze(0),
            "code_attention_mask":     code["attention_mask"].squeeze(0),
            "halstead_input_ids":      hals["input_ids"].squeeze(0),
            "halstead_attention_mask": hals["attention_mask"].squeeze(0),
            "label":                   torch.tensor(label, dtype=torch.long),
        }

# ---------- UTIL ----------
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
def evaluate(model, loader, device, model_name):
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses = []
    y_true, y_pred = [], []

    for batch in loader:
        if model_name == "Hierarchical_BERT":
            code_ids  = batch["code_input_ids"].to(device)
            code_att  = batch["code_attention_mask"].to(device)
            hals_ids  = batch["halstead_input_ids"].to(device)
            hals_att  = batch["halstead_attention_mask"].to(device)
            labels    = batch["label"].to(device)
            logits = model(code_ids, code_att, hals_ids, hals_att)
        elif model_name == "RABERT":
            ids   = batch["input_ids"].to(device)
            att   = batch["attention_mask"].to(device)
            labs  = batch["label"].to(device)
            # simple placeholder relation ids (0) to satisfy the signature
            relation_ids = torch.zeros(ids.size(0), dtype=torch.long, device=device)
            labels = labs
            logits = model(ids, att, relation_ids)
        else:  # FABERT
            ids   = batch["input_ids"].to(device)
            att   = batch["attention_mask"].to(device)
            labels= batch["label"].to(device)
            logits = model(ids, att)

        loss = ce(logits, labels)
        losses.append(loss.item())
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    return float(np.mean(losses)), accuracy_score(y_true, y_pred)

def train_one_model(name: str,
                    model: nn.Module,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    device: str,
                    epochs: int = EPOCHS,
                    lr: float = LR):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        ep_losses = []
        for batch in train_loader:
            opt.zero_grad()
            if name == "Hierarchical_BERT":
                code_ids  = batch["code_input_ids"].to(device)
                code_att  = batch["code_attention_mask"].to(device)
                hals_ids  = batch["halstead_input_ids"].to(device)
                hals_att  = batch["halstead_attention_mask"].to(device)
                labels    = batch["label"].to(device)
                logits    = model(code_ids, code_att, hals_ids, hals_att)
            elif name == "RABERT":
                ids   = batch["input_ids"].to(device)
                att   = batch["attention_mask"].to(device)
                labels= batch["label"].to(device)
                relation_ids = torch.zeros(ids.size(0), dtype=torch.long, device=device)
                logits = model(ids, att, relation_ids)
            else:  # FABERT
                ids   = batch["input_ids"].to(device)
                att   = batch["attention_mask"].to(device)
                labels= batch["label"].to(device)
                logits = model(ids, att)

            loss = ce(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())

        val_loss, val_acc = evaluate(model, val_loader, device, name)
        hist["train_loss"].append(float(np.mean(ep_losses)))
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        print(f"[{name}] Epoch {ep:02d}/{epochs} | train_loss={hist['train_loss'][-1]:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
    return hist

# ---------- MAIN ----------
def main():
    # Load & split once (shared across models)
    df = pd.read_csv(CSV_PATH)
    assert "LargeClass" in df.columns, "Expected 'LargeClass' label column in lcd.csv"
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["LargeClass"])

    tok = BertTokenizer.from_pretrained("bert-base-uncased")

    # Datasets/loaders
    flat_train = FlatTextDataset(train_df, tok, MAX_LEN)
    flat_val   = FlatTextDataset(val_df,   tok, MAX_LEN)
    flat_train_loader = DataLoader(flat_train, batch_size=BATCH, shuffle=True)
    flat_val_loader   = DataLoader(flat_val,   batch_size=BATCH, shuffle=False)

    hier_train = HierarchicalDataset(train_df, tok, MAX_LEN)
    hier_val   = HierarchicalDataset(val_df,   tok, MAX_LEN)
    hier_train_loader = DataLoader(hier_train, batch_size=BATCH, shuffle=True)
    hier_val_loader   = DataLoader(hier_val,   batch_size=BATCH, shuffle=False)

    # Build models
    models: Dict[str, nn.Module] = {
        "FABERT":            FeatureAwareBERT(pretrained_model="bert-base-uncased", feature_dim=16, num_labels=2),
        "Hierarchical_BERT": HierarchicalBERT(pretrained_model="bert-base-uncased", num_labels=2),
        "RABERT":            RelationAwareBERT(pretrained_model="bert-base-uncased", relation_dim=16, num_labels=2),
    }

    # Train each model with appropriate loaders
    histories = {}

    print("\n===== Training FABERT =====")
    histories["FABERT"] = train_one_model("FABERT", models["FABERT"],
                                          flat_train_loader, flat_val_loader, DEVICE)

    print("\n===== Training Hierarchical_BERT =====")
    histories["Hierarchical_BERT"] = train_one_model("Hierarchical_BERT", models["Hierarchical_BERT"],
                                                     hier_train_loader, hier_val_loader, DEVICE)

    print("\n===== Training RABERT =====")
    histories["RABERT"] = train_one_model("RABERT", models["RABERT"],
                                          flat_train_loader, flat_val_loader, DEVICE)

    # ---------- PLOTS ----------
    pub_style()
    xs = range(1, EPOCHS + 1)

    # Accuracy vs Epoch
    plt.figure(figsize=(7, 4.5))
    for name in ["FABERT", "Hierarchical_BERT", "RABERT"]:
        plt.plot(xs, histories[name]["val_acc"], linewidth=2, label=name)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch")
    plt.legend(frameon=False, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "accuracy_vs_epoch.png"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "accuracy_vs_epoch.pdf"), bbox_inches="tight")
    plt.close()

    # Loss vs Epoch
    plt.figure(figsize=(7, 4.5))
    for name in ["FABERT", "Hierarchical_BERT", "RABERT"]:
        plt.plot(xs, histories[name]["val_loss"], linewidth=2, label=name)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
    plt.legend(frameon=False, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "loss_vs_epoch.png"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "loss_vs_epoch.pdf"), bbox_inches="tight")
    plt.close()

    print(f"\nSaved plots in: {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()
