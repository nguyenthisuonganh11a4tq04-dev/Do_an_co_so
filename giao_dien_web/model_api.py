# ===========================================================================
# Imports
# ===========================================================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR  # ← fix: bỏ SequentialLR

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    classification_report,
)
from sklearn.preprocessing import label_binarize

# ===========================================================================
# 1. Config
# ===========================================================================
class Config:

    TEENCODE_PATH = "data/teencode_final.xlsx"
    DATA_DIR        ="data/dataset"

    SEED            = 42
    NUM_CLASSES     = 3
    MAX_LENGTH      = 128

    MODEL_NAME      = "uitnlp/visobert"

    BATCH_SIZE      = 16
    EPOCHS          = 10
    LR_BERT         = 2e-5
    LR_HEAD         = 1e-4
    WARMUP_RATIO    = 0.1
    WEIGHT_DECAY    = 1e-2
    DROPOUT         = 0.1
    PATIENCE        = 3

    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

    COL_TEENCODE    = "Teencode"
    COL_NORMALIZED  = "Định dạng chuẩn"
    COL_EMOTION     = "Nhãn cảm xúc"
    COL_INTENSITY   = "Độ mạnh (1=Nhẹ, 2=TB, 3=Mạnh)"

    HIGH_INTENSITY  = {"cao", "rất cao", "high", "very high", "3", "4", "5"}
    REPEAT_STRONG   = 1

cfg = Config()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)

print(f"Device : {cfg.DEVICE}")
print(f"Model  : {cfg.MODEL_NAME}")

LABEL2ID    = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = ["negative", "neutral", "positive"]

# ===========================================================================
# 2. Load teencode dictionary
# ===========================================================================
teencode_df = pd.read_excel(cfg.TEENCODE_PATH)

def _find_col(df, candidates):
    norm = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in norm:
            return norm[key]
    return None

col_tc  = _find_col(teencode_df, [cfg.COL_TEENCODE,   "teencode"])
col_std = _find_col(teencode_df, [cfg.COL_NORMALIZED,  "dạng chuẩn hóa", "dinh dang chuan"])
col_emo = _find_col(teencode_df, [cfg.COL_EMOTION,     "cam xuc", "emotion"])
col_int = _find_col(teencode_df, [cfg.COL_INTENSITY,   "do manh", "intensity", "mức độ"])

missing = [n for n, c in [(cfg.COL_TEENCODE, col_tc), (cfg.COL_NORMALIZED, col_std),
                           (cfg.COL_EMOTION, col_emo), (cfg.COL_INTENSITY, col_int)] if c is None]
if missing:
    print(f"[WARN] Không tìm thấy cột: {missing}")
    print(f"       Các cột hiện có   : {list(teencode_df.columns)}")

required_cols = [c for c in [col_tc, col_std] if c is not None]
teencode_df   = teencode_df.dropna(subset=required_cols)

TEENCODE_DICT      = {}
TEENCODE_EMOTION   = {}
TEENCODE_INTENSITY = {}

for _, row in teencode_df.iterrows():
    key = str(row[col_tc]).strip().lower()
    if not key:
        continue
    val = str(row[col_std]).strip().split("/")[0].strip() if col_std else key
    if not val:
        continue
    TEENCODE_DICT[key] = val
    if col_emo and pd.notna(row[col_emo]):
        TEENCODE_EMOTION[key] = str(row[col_emo]).strip().lower()
    if col_int and pd.notna(row[col_int]):
        TEENCODE_INTENSITY[key] = str(row[col_int]).strip().lower()

PHRASE_LIST = sorted(
    [(k, v) for k, v in TEENCODE_DICT.items() if " " in k],
    key=lambda x: len(x[0]), reverse=True
)

print(f"Loaded {len(TEENCODE_DICT)} teencode entries.")
print(f"  → Cảm xúc  : {len(TEENCODE_EMOTION)} | Độ mạnh: {len(TEENCODE_INTENSITY)}")
if TEENCODE_EMOTION:
    print(f"  → Phân bố  : {dict(Counter(TEENCODE_EMOTION.values()).most_common(8))}")

# ===========================================================================
# 3. Text preprocessing
# ===========================================================================
def _is_high_intensity(key: str) -> bool:
    return TEENCODE_INTENSITY.get(key, "") in cfg.HIGH_INTENSITY


def normalize_teencode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    for phrase, replacement in PHRASE_LIST:
        text = text.replace(phrase, f" {replacement} ")
    tokens = text.split()
    new_tokens = []
    for t in tokens:
        replaced = TEENCODE_DICT.get(t, t)
        new_tokens.append(replaced)
        if _is_high_intensity(t):
            new_tokens.extend([replaced] * cfg.REPEAT_STRONG)
    return re.sub(r"\s+", " ", " ".join(new_tokens)).strip()

# ===========================================================================
# 4. Load & preprocess data
# ===========================================================================
train_df = pd.read_csv(os.path.join(cfg.DATA_DIR, "train.csv")).dropna()
valid_df  = pd.read_csv(os.path.join(cfg.DATA_DIR, "valid.csv")).dropna()
test_df   = pd.read_csv(os.path.join(cfg.DATA_DIR, "test.csv")).dropna()

print("Đang tiền xử lý teencode...")
train_texts = [normalize_teencode(s) for s in train_df["sentence"]]
valid_texts  = [normalize_teencode(s) for s in valid_df["sentence"]]
test_texts   = [normalize_teencode(s) for s in test_df["sentence"]]

train_labels = train_df["sentiment"].map(LABEL2ID).values.astype(int)
valid_labels  = valid_df["sentiment"].map(LABEL2ID).values.astype(int)
test_labels   = test_df["sentiment"].map(LABEL2ID).values.astype(int)

print(f"Train: {len(train_texts)} | Valid: {len(valid_texts)} | Test: {len(test_texts)}")
print(f"Label dist train : {np.bincount(train_labels)}")
print(f"Label dist valid : {np.bincount(valid_labels)}")
print(f"Label dist test  : {np.bincount(test_labels)}")

# ===========================================================================
# 5. Tokenizer
# ===========================================================================
print(f"\nLoading tokenizer: {cfg.MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

# ===========================================================================
# 6. Dataset & DataLoader
# ===========================================================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, cfg.MAX_LENGTH)
valid_dataset  = SentimentDataset(valid_texts,  valid_labels,  tokenizer, cfg.MAX_LENGTH)
test_dataset   = SentimentDataset(test_texts,   test_labels,   tokenizer, cfg.MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,  drop_last=False)
valid_loader  = DataLoader(valid_dataset,  batch_size=cfg.BATCH_SIZE)
test_loader   = DataLoader(test_dataset,   batch_size=cfg.BATCH_SIZE)

# ===========================================================================
# 7. Model
# ===========================================================================
class ViSoBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden    = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 3, num_classes),
        )
        self._init_classifier()

    def _init_classifier(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask):
        outputs   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits    = self.classifier(cls_token)
        attn_last = outputs.attentions
        return logits, attn_last

model = ViSoBERTClassifier(
    model_name  = cfg.MODEL_NAME,
    num_classes = cfg.NUM_CLASSES,
    dropout     = cfg.DROPOUT,
).to(cfg.DEVICE)

bert_params = list(model.bert.parameters())
head_params = list(model.classifier.parameters())

optimizer = AdamW([
    {"params": bert_params, "lr": cfg.LR_BERT, "weight_decay": cfg.WEIGHT_DECAY},
    {"params": head_params, "lr": cfg.LR_HEAD, "weight_decay": 0.0},
])

class_counts  = np.bincount(train_labels)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float)
class_weights = (class_weights / class_weights.sum() * cfg.NUM_CLASSES).to(cfg.DEVICE)
criterion     = nn.CrossEntropyLoss(weight=class_weights)

total_steps  = len(train_loader) * cfg.EPOCHS
warmup_steps = int(total_steps * cfg.WARMUP_RATIO)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda)
reduce_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {trainable:,}")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

def predict_sentiment(text):

    #text = preprocess_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        pred = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    labels = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    return labels[pred]
