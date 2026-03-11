# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — ENVIRONMENT SETUP, IMPORTS & REPRODUCIBILITY                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, sys, re, io, json, math, time, random, heapq, pickle, warnings, itertools
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd

import matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from IPython.display import display, HTML, Markdown

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
warnings.filterwarnings("ignore")

try:
    import google.colab
    IN_COLAB = True
    from google.colab import files
except ImportError:
    IN_COLAB = False

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED);  np.random.seed(SEED)
torch.manual_seed(SEED);  torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR = Path("/content") if IN_COLAB else Path.cwd().parent
OUT_DIR  = BASE_DIR / "outputs"
PLOT_DIR = OUT_DIR / "plots"
CKPT_DIR = OUT_DIR / "checkpoints"
RES_DIR  = OUT_DIR / "results"
for d in [PLOT_DIR, CKPT_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
PALETTE = {"primary":"#1A237E","secondary":"#E65100","accent":"#00897B",
           "warn":"#C62828","light":"#E8EAF6","success":"#2E7D32","grid":"#D0D0D0"}
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#FAFAFA","axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,"grid.color":PALETTE["grid"],
    "grid.linestyle":"--","grid.alpha":0.5,"font.family":"DejaVu Sans",
    "font.size":11,"axes.titlesize":13,"figure.dpi":120,
})

# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 72)
print(f"  Python         : {sys.version.split()[0]}")
print(f"  PyTorch        : {torch.__version__}")
print(f"  NumPy          : {np.__version__}")
print(f"  Device         : {DEVICE}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU            : {p.name}")
    print(f"  VRAM (GB)      : {p.total_memory/1e9:.2f}")
print(f"  In Colab       : {IN_COLAB}")
print(f"  Random seed    : {SEED}")
print("=" * 72)
print("  ✅  Environment ready.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — DATA LOADING & DEEP EXPLORATION                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_dataset() -> pd.DataFrame:
    if IN_COLAB:
        print("📂  Please upload 'english_to_urdu_dataset.xlsx' below ↓")
        uploaded = files.upload()
        if not uploaded: raise FileNotFoundError("No file uploaded.")
        fname = list(uploaded.keys())[0]
        return pd.read_excel(fname, engine="openpyxl")
    else:
        path = BASE_DIR / "data" / "english_to_urdu_dataset.xlsx"
        if not path.exists(): raise FileNotFoundError(f"Not found: {path}")
        print(f"  ✅  Loading from: {path}")
        return pd.read_excel(path, engine="openpyxl")

df_raw = load_dataset()

print("\n" + "=" * 72)
print("  DATASET OVERVIEW")
print("=" * 72)
print(f"  Shape          : {df_raw.shape}")
print(f"  Columns        : {df_raw.columns.tolist()}")
print(f"  Total pairs    : {len(df_raw):,}")
print(f"  Memory usage   : {df_raw.memory_usage(deep=True).sum()/1e6:.2f} MB")
print("\n  Data types:")
print(df_raw.dtypes.to_string())
display(df_raw.head(3))

print("\n" + "=" * 72)
print("  5 RANDOM SAMPLES (seed=42)")
print("=" * 72)
for i, (_, row) in enumerate(df_raw.sample(5, random_state=SEED).iterrows(), 1):
    print(f"  [{i}] ENG : {str(row['eng'])[:100]}")
    print(f"      URD : {str(row['urdu'])[:100]}\n")

print("=" * 72)
print("  MISSING VALUE ANALYSIS")
print("=" * 72)
missing = df_raw.isnull().sum()
print(f"  eng   {int(missing['eng'])}")
print(f"  urdu  {int(missing['urdu'])}")
print(f"  Total missing: {int(missing.sum())}")

print("\n  DUPLICATE ROW ANALYSIS")
print("=" * 72)
n_dup = df_raw.duplicated().sum()
print(f"  Total duplicate rows   : {n_dup}")
print(f"  Percentage duplicates  : {n_dup/len(df_raw)*100:.2f}%")

print("\n  SENTENCE LENGTH STATISTICS (raw)")
print("=" * 72)
df_raw["eng_len"]  = df_raw["eng"].fillna("").apply(lambda x: len(str(x).split()))
df_raw["urdu_len"] = df_raw["urdu"].fillna("").apply(lambda x: len(str(x).split()))
display(df_raw[["eng_len","urdu_len"]].describe().round(2))

percs = [50, 75, 90, 95, 97, 99]
print(f"\n  {'Percentile':>12}  {'ENG tokens':>12}  {'URDU tokens':>12}")
for p in percs:
    print(f"  {p:>11}th  {int(np.percentile(df_raw['eng_len'],p)):>12}  {int(np.percentile(df_raw['urdu_len'],p)):>12}")

# ── 6-panel visualisation ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
fig.suptitle("Corpus Exploration — Raw Dataset (9,103 pairs)", fontsize=15, fontweight="bold")

ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

ax[0].hist(df_raw["eng_len"], bins=60, color=PALETTE["primary"], alpha=0.82, edgecolor="white")
ax[0].axvline(df_raw["eng_len"].mean(), color=PALETTE["warn"], ls="--", lw=1.8, label=f"Mean={df_raw['eng_len'].mean():.1f}")
ax[0].axvline(df_raw["eng_len"].median(), color=PALETTE["accent"], ls="-.", lw=1.5, label=f"Median={df_raw['eng_len'].median():.0f}")
ax[0].set_title("English Token Length Distribution"); ax[0].set_xlabel("# Tokens"); ax[0].legend(fontsize=8)

ax[1].hist(df_raw["urdu_len"], bins=60, color=PALETTE["secondary"], alpha=0.82, edgecolor="white")
ax[1].axvline(df_raw["urdu_len"].mean(), color=PALETTE["warn"], ls="--", lw=1.8, label=f"Mean={df_raw['urdu_len'].mean():.1f}")
ax[1].axvline(df_raw["urdu_len"].median(), color=PALETTE["accent"], ls="-.", lw=1.5, label=f"Median={df_raw['urdu_len'].median():.0f}")
ax[1].set_title("Urdu Token Length Distribution"); ax[1].set_xlabel("# Tokens"); ax[1].legend(fontsize=8)

ratio = df_raw["urdu_len"] / df_raw["eng_len"].replace(0, np.nan)
ax[2].hist(ratio.dropna(), bins=50, color=PALETTE["accent"], alpha=0.82, edgecolor="white")
ax[2].axvline(ratio.mean(), color=PALETTE["warn"], ls="--", lw=1.8, label=f"Mean={ratio.mean():.2f}")
ax[2].axvline(1.0, color="grey", ls=":", lw=1.2, label="1:1")
ax[2].set_title("Urdu/English Length Ratio"); ax[2].set_xlabel("Ratio"); ax[2].legend(fontsize=8)

sc = ax[3].scatter(df_raw["eng_len"], df_raw["urdu_len"], c=ratio.fillna(1), cmap="coolwarm", alpha=0.22, s=5)
lim = max(df_raw["eng_len"].max(), df_raw["urdu_len"].max()) + 5
ax[3].plot([0, lim], [0, lim], "k--", lw=1, alpha=0.35, label="y=x")
plt.colorbar(sc, ax=ax[3]).set_label("Len ratio", fontsize=8)
ax[3].set_title("Eng vs Urdu Length"); ax[3].set_xlabel("English"); ax[3].set_ylabel("Urdu"); ax[3].legend(fontsize=8)

ax[4].boxplot([df_raw["eng_len"].values, df_raw["urdu_len"].values], labels=["English","Urdu"],
              patch_artist=True, notch=True, boxprops=dict(facecolor=PALETTE["light"]),
              medianprops=dict(color=PALETTE["warn"], lw=2))
ax[4].set_title("Length Box-Plot"); ax[4].set_ylabel("# Tokens")

all_eng = " ".join(df_raw["eng"].dropna()).split()
top20   = Counter(all_eng).most_common(20)
w, c    = zip(*top20)
ax[5].bar(w, c, color=[PALETTE["primary"] if i<10 else PALETTE["secondary"] for i in range(20)], edgecolor="white")
ax[5].set_yscale("log"); ax[5].tick_params(axis="x", rotation=45)
ax[5].set_title("Top-20 English Tokens (log scale)"); ax[5].set_ylabel("Frequency")

plt.savefig(PLOT_DIR / "01_corpus_exploration.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾  Saved: 01_corpus_exploration.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — ADVANCED DATA PREPROCESSING & QUALITY FILTERING               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

URDU_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

def preprocess_english(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip(): return None
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = text.replace("\u2018","'").replace("\u2019","'").replace("\u201c",'"').replace("\u201d",'"')
    text = re.sub(r"[^a-z0-9\s.,!?;:\'\"\-]", " ", text)
    text = re.sub(r"([!?.,;:]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text.split()) >= 2 else None

def preprocess_urdu(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip(): return None
    text = text.strip()
    for u, a in [("۔"," . "),("،"," , "),("؟"," ? "),("؛"," ; "),("٪"," % ")]:
        text = text.replace(u, a)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\uFEFF]","",text)
    text = re.sub(r"[\[\(].*?[\]\)]","",text)
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s.,!?;:\'\"\-]", " ", text)
    text = re.sub(r"\s+"," ",text).strip()
    return text if len(text.split()) >= 2 else None

def urdu_char_ratio(text: str) -> float:
    s = text.replace(" ","")
    return len(URDU_RE.findall(text)) / max(len(s),1)

print("  Preprocessing in progress...")
df = df_raw[["eng","urdu"]].copy()
df["eng_clean"]  = df["eng"].apply(preprocess_english)
df["urdu_clean"] = df["urdu"].apply(preprocess_urdu)
n_null_eng  = df["eng_clean"].isnull().sum()
n_null_urdu = df["urdu_clean"].isnull().sum()
print(f"  Nulls after English preprocessing  : {n_null_eng}")
print(f"  Nulls after Urdu    preprocessing  : {n_null_urdu}")
df.dropna(subset=["eng_clean","urdu_clean"], inplace=True)

before_dup = len(df)
df.drop_duplicates(subset=["eng_clean","urdu_clean"], inplace=True)
print(f"  Rows removed as duplicates         : {before_dup - len(df)}")

df["_ur"] = df["urdu_clean"].apply(urdu_char_ratio)
LOW_THRESH = 0.40
df = df[df["_ur"] >= LOW_THRESH].copy(); df.drop("_ur", axis=1, inplace=True)

df["eng_len"]  = df["eng_clean"].apply(lambda x: len(x.split()))
df["urdu_len"] = df["urdu_clean"].apply(lambda x: len(x.split()))
df = df[(df["eng_len"] >= 2) & (df["urdu_len"] >= 2)].copy()

ENG_MAX  = int(np.percentile(df["eng_len"],  97))
URDU_MAX = int(np.percentile(df["urdu_len"], 97))
MAX_SRC_LEN = ENG_MAX
MAX_TGT_LEN = URDU_MAX + 2

print(f"  97th pct English length (MAX_SRC_LEN) : {ENG_MAX}")
print(f"  97th pct Urdu    length (MAX_TGT_LEN) : {URDU_MAX}")

before_len = len(df)
df = df[(df["eng_len"] <= ENG_MAX) & (df["urdu_len"] <= URDU_MAX)].copy()
df["len_ratio"] = df["urdu_len"] / df["eng_len"]
lo, hi = df["len_ratio"].quantile(0.01), df["len_ratio"].quantile(0.99)
before_ratio = len(df)
df = df[(df["len_ratio"] >= lo) & (df["len_ratio"] <= hi)].copy().reset_index(drop=True)
print(f"  Final cleaned dataset size             : {len(df):,} pairs")

df[["eng_clean","urdu_clean"]].to_csv(RES_DIR/"cleaned_dataset.csv", index=False)
print(f"  Saved: outputs/results/cleaned_dataset.csv")

print("\n  5 CLEANED SAMPLES:")
for i, (_, row) in enumerate(df.sample(5, random_state=SEED).iterrows(), 1):
    print(f"  [{i}] ENG : {row['eng_clean'][:90]}")
    print(f"      URD : {row['urdu_clean'][:90]}\n")

# Preprocessing visualisation
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Post-Preprocessing Length & Quality Analysis", fontsize=14, fontweight="bold")

axes[0,0].hist(df_raw["eng_len"], bins=50, color=PALETTE["primary"], alpha=0.4, label="Before", density=True)
axes[0,0].hist(df["eng_len"],     bins=50, color=PALETTE["secondary"],alpha=0.7, label="After",  density=True)
axes[0,0].axvline(ENG_MAX, color=PALETTE["warn"],ls="--",lw=1.8,label=f"Cap={ENG_MAX}")
axes[0,0].set_title("English Length: Before vs After"); axes[0,0].legend(); axes[0,0].set_xlabel("# tokens")

axes[0,1].hist(df_raw["urdu_len"], bins=50, color=PALETTE["primary"], alpha=0.4, label="Before", density=True)
axes[0,1].hist(df["urdu_len"],     bins=50, color=PALETTE["secondary"],alpha=0.7, label="After",  density=True)
axes[0,1].axvline(URDU_MAX,color=PALETTE["warn"],ls="--",lw=1.8,label=f"Cap={URDU_MAX}")
axes[0,1].set_title("Urdu Length: Before vs After"); axes[0,1].legend(); axes[0,1].set_xlabel("# tokens")

axes[0,2].scatter(df["eng_len"], df["urdu_len"], c=df["len_ratio"], cmap="viridis", alpha=0.2, s=5)
axes[0,2].set_title("Eng vs Urdu Length — Cleaned"); axes[0,2].set_xlabel("English"); axes[0,2].set_ylabel("Urdu")

axes[1,0].hist(df["len_ratio"], bins=50, color=PALETTE["accent"], alpha=0.82, edgecolor="white")
axes[1,0].axvline(1.0, color="grey",ls=":",lw=1.2,label="1:1")
axes[1,0].axvline(df["len_ratio"].mean(),color=PALETTE["warn"],ls="--",lw=1.8,label=f"Mean={df['len_ratio'].mean():.2f}")
axes[1,0].set_title("Length Ratio (Urdu/Eng)"); axes[1,0].set_xlabel("Ratio"); axes[1,0].legend()

s0,s1,s2,s3 = len(df_raw), len(df_raw)-n_null_eng-n_null_urdu, len(df_raw)-before_dup+before_dup-before_dup, len(df)
sizes = [len(df_raw), len(df_raw)-n_null_eng-n_null_urdu, before_ratio+( before_len-len(df)), before_len, len(df)]
stages = ["Raw","Null drop","Dedup+Script","Length cap","Final"]
colors_ = [PALETTE["primary"]]*3 + [PALETTE["accent"]] + [PALETTE["success"]]
axes[1,1].bar(stages, sizes, color=colors_, edgecolor="white")
for rect, val in zip(axes[1,1].patches, sizes):
    axes[1,1].text(rect.get_x()+rect.get_width()/2, rect.get_height()+20, f"{val:,}", ha="center", fontsize=8, fontweight="bold")
axes[1,1].set_title("Dataset Funnel"); axes[1,1].set_ylabel("# pairs"); axes[1,1].tick_params(axis="x",labelsize=8)

axes[1,2].hist(df["eng_len"], bins=40, color=PALETTE["primary"], alpha=0.7, label="English", density=True)
axes[1,2].hist(df["urdu_len"],bins=40, color=PALETTE["secondary"],alpha=0.7, label="Urdu",    density=True)
axes[1,2].set_title("Clean Length Distributions (overlay)"); axes[1,2].set_xlabel("# tokens"); axes[1,2].legend()

plt.tight_layout()
plt.savefig(PLOT_DIR/"02_preprocessing_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾  Saved: 02_preprocessing_analysis.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — STRATIFIED TRAIN / VALIDATION / TEST SPLIT                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

df["len_bin"] = pd.qcut(df["eng_len"], q=5, labels=False, duplicates="drop")
unique_idx = df.drop_duplicates(subset=["eng_clean"]).index.tolist()
train_idx, temp_idx = train_test_split(unique_idx, test_size=0.20, random_state=SEED,
                                        stratify=df.loc[unique_idx,"len_bin"])
val_idx, test_idx   = train_test_split(temp_idx,  test_size=0.50, random_state=SEED,
                                        stratify=df.loc[temp_idx, "len_bin"])

train_set = set(df.loc[train_idx,"eng_clean"])
val_set   = set(df.loc[val_idx,  "eng_clean"])
test_set  = set(df.loc[test_idx, "eng_clean"])

train_df = df[df["eng_clean"].isin(train_set)].copy().reset_index(drop=True)
val_df   = df[df["eng_clean"].isin(val_set)  ].copy().reset_index(drop=True)
test_df  = df[df["eng_clean"].isin(test_set) ].copy().reset_index(drop=True)

total = len(train_df)+len(val_df)+len(test_df)
print("=" * 64)
print("  DATASET SPLIT SUMMARY")
print("=" * 64)
print(f"  Training   : {len(train_df):,} pairs  ({len(train_df)/total*100:.1f}%)")
print(f"  Validation : {len(val_df):,}   pairs  ({len(val_df)/total*100:.1f}%)")
print(f"  Test       : {len(test_df):,}   pairs  ({len(test_df)/total*100:.1f}%)")
print(f"  Total      : {total:,}")
print(f"\n  Overlap checks (0 = no overlap):")
tv = len(set(train_df["eng_clean"]) & set(val_df["eng_clean"]))
tt = len(set(train_df["eng_clean"]) & set(test_df["eng_clean"]))
vt = len(set(val_df["eng_clean"])   & set(test_df["eng_clean"]))
print(f"  Train ∩ Val  : {tv}")
print(f"  Train ∩ Test : {tt}")
print(f"  Val   ∩ Test : {vt}")
assert tv == tt == vt == 0
print("  No overlap between any splits.  ✅")

train_df[["eng_clean","urdu_clean"]].to_csv(RES_DIR/"train_split.csv", index=False)
val_df[["eng_clean","urdu_clean"]].to_csv(RES_DIR/"val_split.csv",   index=False)
test_df[["eng_clean","urdu_clean"]].to_csv(RES_DIR/"test_split.csv",  index=False)
print("  Saved: train_split.csv, val_split.csv, test_split.csv")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Dataset Split Analysis", fontsize=14, fontweight="bold")
sizes   = [len(train_df), len(val_df), len(test_df)]
lbls    = [f"Train\n{len(train_df):,}\n({len(train_df)/total*100:.1f}%)", f"Val\n{len(val_df):,}\n({len(val_df)/total*100:.1f}%)", f"Test\n{len(test_df):,}\n({len(test_df)/total*100:.1f}%)"]
colors_ = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
axes[0].pie(sizes, labels=lbls, colors=colors_, autopct="%1.1f%%", startangle=140, wedgeprops={"edgecolor":"white","linewidth":2})
axes[0].set_title("Split Proportions")
for sp, lbl, col in [(train_df,"Train",PALETTE["primary"]),(val_df,"Val",PALETTE["secondary"]),(test_df,"Test",PALETTE["accent"])]:
    axes[1].hist(sp["eng_len"],  bins=40, density=True, alpha=0.45, label=lbl, color=col)
    axes[2].hist(sp["urdu_len"], bins=40, density=True, alpha=0.45, label=lbl, color=col)
axes[1].set_title("English Length per Split"); axes[1].set_xlabel("# tokens"); axes[1].legend()
axes[2].set_title("Urdu Length per Split");    axes[2].set_xlabel("# tokens"); axes[2].legend()
plt.tight_layout()
plt.savefig(PLOT_DIR/"03_dataset_split.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾  Saved: 03_dataset_split.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — TOKENISATION & VOCABULARY CONSTRUCTION                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class VocabConfig:
    min_freq: int = 2; max_size: int = 50_000; lower: bool = False

class Vocabulary:
    PAD_TOKEN="<pad>"; PAD_IDX=0
    BOS_TOKEN="<bos>"; BOS_IDX=1
    EOS_TOKEN="<eos>"; EOS_IDX=2
    UNK_TOKEN="<unk>"; UNK_IDX=3
    SPECIAL=["<pad>","<bos>","<eos>","<unk>"]

    def __init__(self, name="vocab", cfg=None):
        self.name=name; self.cfg=cfg or VocabConfig()
        self.freq=Counter(); self.token2idx={}; self.idx2token={}
        for i,t in enumerate(self.SPECIAL):
            self.token2idx[t]=i; self.idx2token[i]=t

    def tokenize(self, text):
        return (text.lower() if self.cfg.lower else text).strip().split()

    def build(self, sentences):
        for s in sentences: self.freq.update(self.tokenize(s))
        eligible=sorted([(t,c) for t,c in self.freq.items() if c>=self.cfg.min_freq],key=lambda x:(-x[1],x[0]))
        if self.cfg.max_size: eligible=eligible[:self.cfg.max_size-4]
        idx=4
        for token,_ in eligible:
            if token not in self.token2idx:
                self.token2idx[token]=idx; self.idx2token[idx]=token; idx+=1
        return self

    def encode(self, text, add_bos=False, add_eos=False):
        ids=[self.token2idx.get(t, self.UNK_IDX) for t in self.tokenize(text)]
        if add_bos: ids=[self.BOS_IDX]+ids
        if add_eos: ids=ids+[self.EOS_IDX]
        return ids

    def decode(self, ids, skip_special=True):
        skip={self.PAD_IDX, self.BOS_IDX}; out=[]
        for i in ids:
            if skip_special and i in skip: continue
            tok=self.idx2token.get(i, self.UNK_TOKEN)
            if tok==self.EOS_TOKEN: break
            out.append(tok)
        return " ".join(out)

    def oov_rate(self, sentences):
        total=oov=0
        oov_toks=Counter()
        for s in sentences:
            for t in self.tokenize(s):
                total+=1
                if t not in self.token2idx: oov+=1; oov_toks[t]+=1
        return {"total_tokens":total,"oov_count":oov,"oov_rate_pct":round(oov/max(total,1)*100,4),"top_oov":oov_toks.most_common(10)}

    def __len__(self): return len(self.token2idx)
    def __contains__(self, t): return t in self.token2idx
    def save(self, path):
        with open(path,"wb") as f: pickle.dump(self,f)

print("  Building English (source) vocabulary...")
src_vocab = Vocabulary("English").build(train_df["eng_clean"].tolist())
print("  Building Urdu    (target) vocabulary...")
tgt_vocab = Vocabulary("Urdu").build(train_df["urdu_clean"].tolist())

print("\n" + "=" * 64)
print("  VOCABULARY SUMMARY")
print("=" * 64)
print(f"  English (source) vocab size : {len(src_vocab):,}")
print(f"  Urdu    (target) vocab size : {len(tgt_vocab):,}")
print(f"  Min frequency threshold     : 2")
print(f"  Special tokens              : <pad>=0, <bos>=1, <eos>=2, <unk>=3")

for tag, voc, sents in [("val ENG",  src_vocab, val_df["eng_clean"].tolist()),
                         ("val URDU", tgt_vocab, val_df["urdu_clean"].tolist()),
                         ("test ENG", src_vocab, test_df["eng_clean"].tolist()),
                         ("test URDU",tgt_vocab, test_df["urdu_clean"].tolist())]:
    r = voc.oov_rate(sents)
    print(f"  OOV rate on {tag:8s}       : {r['oov_rate_pct']:.2f}%  ({r['oov_count']:,}/{r['total_tokens']:,})")
print("  Vocabularies saved.")

src_vocab.save(RES_DIR/"src_vocab.pkl")
tgt_vocab.save(RES_DIR/"tgt_vocab.pkl")

# ── Vocabulary visualisation ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Vocabulary Analysis", fontsize=14, fontweight="bold")

eng_c = sorted(src_vocab.freq.values(), reverse=True)
urd_c = sorted(tgt_vocab.freq.values(), reverse=True)

axes[0,0].plot(range(1,len(eng_c)+1), eng_c, color=PALETTE["primary"], lw=1.5)
axes[0,0].set_xscale("log"); axes[0,0].set_yscale("log")
axes[0,0].set_title("English Vocab — Zipf Distribution"); axes[0,0].set_xlabel("Rank (log)"); axes[0,0].set_ylabel("Freq (log)")

axes[0,1].plot(range(1,len(urd_c)+1), urd_c, color=PALETTE["secondary"], lw=1.5)
axes[0,1].set_xscale("log"); axes[0,1].set_yscale("log")
axes[0,1].set_title("Urdu Vocab — Zipf Distribution"); axes[0,1].set_xlabel("Rank (log)"); axes[0,1].set_ylabel("Freq (log)")

top30_e = src_vocab.freq.most_common(30); te, ce = zip(*top30_e)
axes[0,2].barh(list(te)[::-1], list(ce)[::-1], color=PALETTE["primary"], alpha=0.8, edgecolor="white")
axes[0,2].set_title("English Top-30 Tokens"); axes[0,2].set_xlabel("Frequency")

cum_e = np.cumsum(eng_c)/sum(eng_c); cum_u = np.cumsum(urd_c)/sum(urd_c)
axes[1,0].plot(range(1,len(cum_e)+1), cum_e, color=PALETTE["primary"], lw=2, label="English")
axes[1,0].plot(range(1,len(cum_u)+1), cum_u, color=PALETTE["secondary"], lw=2, label="Urdu")
for p in [0.80,0.90,0.95]:
    axes[1,0].axhline(p, color="grey", ls=":", lw=1, alpha=0.6)
    axes[1,0].text(50, p+0.005, f"{int(p*100)}%", fontsize=7, color="grey")
axes[1,0].set_title("Cumulative Vocab Coverage"); axes[1,0].set_xlabel("Vocab size"); axes[1,0].legend()

top30_u = tgt_vocab.freq.most_common(30); tu, cu = zip(*top30_u)
axes[1,1].barh(list(tu)[::-1], list(cu)[::-1], color=PALETTE["secondary"], alpha=0.8, edgecolor="white")
axes[1,1].set_title("Urdu Top-30 Tokens"); axes[1,1].set_xlabel("Frequency")

fb = [1,2,3,5,10,20,50,100,500,10000]
he = [sum(1 for c in src_vocab.freq.values() if fb[i]<=c<fb[i+1]) for i in range(len(fb)-1)]
bl = [f"[{fb[i]},{fb[i+1]})" for i in range(len(fb)-1)]
axes[1,2].bar(bl, he, color=PALETTE["accent"], alpha=0.8, edgecolor="white")
axes[1,2].set_title("English Tokens by Freq Bin"); axes[1,2].set_xlabel("Frequency range"); axes[1,2].set_ylabel("# unique tokens")
axes[1,2].tick_params(axis="x", rotation=40)

plt.tight_layout()
plt.savefig(PLOT_DIR/"04_vocabulary_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾  Saved: 04_vocabulary_analysis.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — SEQUENCE ENCODING, PADDING & BATCHING                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BATCH_SIZE = 64

class NMTDataset(Dataset):
    def __init__(self, dataframe, sv, tv, max_src=None, max_tgt=None):
        self.sv=sv; self.tv=tv; self.data=dataframe.reset_index(drop=True)
        self.src_ids, self.tgt_in_ids, self.tgt_out_ids = [], [], []
        for _, row in self.data.iterrows():
            s = sv.encode(row["eng_clean"],  add_bos=False, add_eos=True)
            t = tv.encode(row["urdu_clean"], add_bos=True,  add_eos=True)
            if max_src: s=s[:max_src]
            if max_tgt: t=t[:max_tgt+2]
            self.src_ids.append(s); self.tgt_in_ids.append(t[:-1]); self.tgt_out_ids.append(t[1:])

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return {"src":torch.tensor(self.src_ids[idx],     dtype=torch.long),
                "tgt_in":torch.tensor(self.tgt_in_ids[idx], dtype=torch.long),
                "tgt_out":torch.tensor(self.tgt_out_ids[idx],dtype=torch.long),
                "src_text":self.data.loc[idx,"eng_clean"],
                "tgt_text":self.data.loc[idx,"urdu_clean"]}

def collate_fn(batch):
    src_  =[b["src"]     for b in batch]; ti=[b["tgt_in"]  for b in batch]; to=[b["tgt_out"] for b in batch]
    sl=torch.tensor([len(s) for s in src_]); tl=torch.tensor([len(t) for t in ti])
    return {"src":nn.utils.rnn.pad_sequence(src_,  batch_first=True, padding_value=0),
            "tgt_in": nn.utils.rnn.pad_sequence(ti, batch_first=True, padding_value=0),
            "tgt_out":nn.utils.rnn.pad_sequence(to, batch_first=True, padding_value=0),
            "src_lengths":sl,"tgt_lengths":tl,
            "src_texts":[b["src_text"] for b in batch],"tgt_texts":[b["tgt_text"] for b in batch]}

print("  Building datasets...")
train_dataset = NMTDataset(train_df, src_vocab, tgt_vocab, MAX_SRC_LEN, MAX_TGT_LEN)
val_dataset   = NMTDataset(val_df,   src_vocab, tgt_vocab, MAX_SRC_LEN, MAX_TGT_LEN)
test_dataset  = NMTDataset(test_df,  src_vocab, tgt_vocab, MAX_SRC_LEN, MAX_TGT_LEN)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

sb = next(iter(train_loader))
print(f"  Train batches : {len(train_loader)}")
print(f"  Val   batches : {len(val_loader)}")
print(f"  Test  batches : {len(test_loader)}")
print(f"  Sample batch shapes:")
print(f"    src     : {tuple(sb['src'].shape)}    (batch, src_len)")
print(f"    tgt_in  : {tuple(sb['tgt_in'].shape)}  (batch, tgt_len)")
print(f"    tgt_out : {tuple(sb['tgt_out'].shape)}  (batch, tgt_len)")
print(f"    src_lengths : {tuple(sb['src_lengths'].shape)}")
pad_pos = (sb['src'] == 0).sum().item()
print(f"  Padding positions in src (sample): {pad_pos} / {sb['src'].numel()}")
print("\n  Teacher-forcing alignment:")
print(f"    tgt_in  first 5 ids: {sb['tgt_in'][0][:5].tolist()}  (should start with BOS=1)")
print(f"    tgt_out last  5 ids: {sb['tgt_out'][0][-5:].tolist()} (should end with EOS=2)")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Batch Encoding & Padding", fontsize=14, fontweight="bold")
N = min(16, BATCH_SIZE)
axes[0].imshow(sb["src"][:N].numpy(), aspect="auto", cmap="Blues", interpolation="nearest")
axes[0].set_title(f"src token IDs (first {N} samples)"); axes[0].set_xlabel("Token position"); axes[0].set_ylabel("Sample")
axes[1].imshow((sb["src"][:N]==0).float().numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
axes[1].set_title("Padding mask (red=PAD)"); axes[1].set_xlabel("Token position")
axes[2].hist(sb["src_lengths"].numpy(), bins=20, color=PALETTE["primary"], alpha=0.8, edgecolor="white")
axes[2].set_title("Source lengths in batch"); axes[2].set_xlabel("# tokens"); axes[2].set_ylabel("Count")
plt.tight_layout()
plt.savefig(PLOT_DIR/"05_batch_structure.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾  Saved: 05_batch_structure.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — VANILLA RNN ENCODER-DECODER ARCHITECTURE                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class ModelConfig:
    src_vocab_size:int=0; tgt_vocab_size:int=0; embed_dim:int=256; hidden_dim:int=512
    n_layers:int=2; dropout:float=0.3; pad_idx:int=0; lr:float=1e-3; clip:float=1.0
    epochs:int=30; batch_size:int=64; label_smoothing:float=0.1

class RNNEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg=cfg
        self.embedding=nn.Embedding(cfg.src_vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx)
        self.dropout=nn.Dropout(cfg.dropout)
        self.rnn=nn.RNN(cfg.embed_dim, cfg.hidden_dim, cfg.n_layers, batch_first=True,
                        nonlinearity="tanh", dropout=cfg.dropout if cfg.n_layers>1 else 0.0)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad(): self.embedding.weight[self.cfg.pad_idx].fill_(0)
        for n,p in self.rnn.named_parameters():
            if "weight_ih" in n: nn.init.xavier_uniform_(p)
            elif "weight_hh" in n: nn.init.orthogonal_(p)
            elif "bias" in n: nn.init.zeros_(p)
    def forward(self, src): return self.rnn(self.dropout(self.embedding(src)))

class RNNDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg=cfg
        self.embedding=nn.Embedding(cfg.tgt_vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx)
        self.dropout=nn.Dropout(cfg.dropout)
        self.rnn=nn.RNN(cfg.embed_dim, cfg.hidden_dim, cfg.n_layers, batch_first=True,
                        nonlinearity="tanh", dropout=cfg.dropout if cfg.n_layers>1 else 0.0)
        self.fc_out=nn.Linear(cfg.hidden_dim, cfg.tgt_vocab_size)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad(): self.embedding.weight[self.cfg.pad_idx].fill_(0)
        for n,p in self.rnn.named_parameters():
            if "weight_ih" in n: nn.init.xavier_uniform_(p)
            elif "weight_hh" in n: nn.init.orthogonal_(p)
            elif "bias" in n: nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)
    def forward(self, tgt_in, hidden):
        out, h = self.rnn(self.dropout(self.embedding(tgt_in)), hidden)
        return self.fc_out(out), h

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, cfg): super().__init__(); self.encoder=enc; self.decoder=dec; self.cfg=cfg
    def forward(self, src, tgt_in): _, h=self.encoder(src); logits,_=self.decoder(tgt_in, h); return logits
    def encode(self, src): return self.encoder(src)

def build_model(cfg):
    return Seq2Seq(RNNEncoder(cfg), RNNDecoder(cfg), cfg).to(DEVICE)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

DEFAULT_CFG = ModelConfig(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
    embed_dim=256, hidden_dim=512, n_layers=2, dropout=0.3, lr=1e-3, clip=1.0,
    epochs=30, batch_size=BATCH_SIZE, label_smoothing=0.1)

model = build_model(DEFAULT_CFG)

print("=" * 64)
print("  MODEL ARCHITECTURE SUMMARY")
print("=" * 64)
print(model)
print("\n" + "=" * 64)
print("  PARAMETER COUNT")
print("=" * 64)
enc_p=count_params(model.encoder); dec_p=count_params(model.decoder); tot=count_params(model)
print(f"  Encoder parameters  : {enc_p:,}")
print(f"  Decoder parameters  : {dec_p:,}")
print(f"  Total parameters    : {tot:,}")
print(f"  Model size (float32): {tot*4/1e6:.2f} MB")
print(f"\n  Configuration:")
for k,v in vars(DEFAULT_CFG).items(): print(f"    {k:<20}: {v}")

model.eval()
with torch.no_grad():
    dummy_src=torch.randint(4,len(src_vocab),(4,10)).to(DEVICE)
    dummy_tgt=torch.randint(4,len(tgt_vocab),(4,8)).to(DEVICE)
    out=model(dummy_src, dummy_tgt)
print(f"\n  Forward pass sanity: input ({dummy_src.shape}, {dummy_tgt.shape}) → output {tuple(out.shape)}  ✅")

param_bd={n:sum(x.numel() for x in m.parameters(recurse=False) if x.requires_grad) for n,m in model.named_modules() if sum(x.numel() for x in m.parameters(recurse=False) if x.requires_grad)>0}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Parameter Breakdown", fontsize=14, fontweight="bold")
axes[0].pie(list(param_bd.values()), labels=list(param_bd.keys()), autopct="%1.1f%%", startangle=140,
            colors=plt.cm.Set3(np.linspace(0,1,len(param_bd))), wedgeprops={"edgecolor":"white"})
axes[0].set_title("Parameter Share by Module")
axes[1].barh(list(param_bd.keys())[::-1], list(param_bd.values())[::-1], color=PALETTE["primary"], alpha=0.8, edgecolor="white")
axes[1].set_xlabel("# Parameters"); axes[1].set_title("Parameter Count by Module")
plt.tight_layout()
plt.savefig(PLOT_DIR/"06_model_architecture.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾  Saved: 06_model_architecture.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — TRAINING LOOP, EARLY STOPPING & EXPERIMENT TRACKING           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__(); self.vocab_size=vocab_size; self.pad_idx=pad_idx
        self.smoothing=smoothing; self.confidence=1.0-smoothing
        self.criterion=nn.KLDivLoss(reduction="sum")
    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            sd=torch.full_like(log_p, self.smoothing/(self.vocab_size-2))
            sd.scatter_(1, target.unsqueeze(1), self.confidence)
            sd[:,self.pad_idx]=0.0
        mask=(target!=self.pad_idx)
        return self.criterion(log_p[mask], sd[mask])/mask.sum()

def train_epoch(model, loader, optimizer, criterion, clip):
    model.train(); tl=tt=0
    for b in loader:
        src=b["src"].to(DEVICE); ti=b["tgt_in"].to(DEVICE); to=b["tgt_out"].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        lg=model(src, ti); B,T,V=lg.shape
        loss=criterion(lg.reshape(B*T,V), to.reshape(B*T))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
        n=(to!=0).sum().item(); tl+=loss.item()*n; tt+=n
    avg=tl/max(tt,1); return avg, math.exp(min(avg,10))

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval(); tl=tt=0
    for b in loader:
        src=b["src"].to(DEVICE); ti=b["tgt_in"].to(DEVICE); to=b["tgt_out"].to(DEVICE)
        lg=model(src, ti); B,T,V=lg.shape
        loss=criterion(lg.reshape(B*T,V), to.reshape(B*T))
        n=(to!=0).sum().item(); tl+=loss.item()*n; tt+=n
    avg=tl/max(tt,1); return avg, math.exp(min(avg,10))

def grad_norm(model):
    return math.sqrt(sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None))

def train_model(model, cfg, trn, val, ckpt_name="best_model.pt"):
    criterion=LabelSmoothingLoss(cfg.tgt_vocab_size, cfg.pad_idx, cfg.label_smoothing)
    optimizer=optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9,0.98), eps=1e-8)
    scheduler=ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    PATIENCE=7; hist={"epoch":[],"train_loss":[],"val_loss":[],"train_ppl":[],"val_ppl":[],"lr":[],"grad_norm":[],"epoch_time":[]}
    best_val=float("inf"); wait=0; ckpt_path=CKPT_DIR/ckpt_name
    print(f"  Training on {DEVICE} | max {cfg.epochs} epochs | early-stop patience={PATIENCE}")
    print(f"  {'─'*80}")
    print(f"  {'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>10} │ {'Train PPL':>9} │ {'Val PPL':>9} │ {'LR':>8} │ Time")
    print(f"  {'─'*80}")
    for ep in range(1, cfg.epochs+1):
        t0=time.time()
        tl,tp=train_epoch(model, trn, optimizer, criterion, cfg.clip)
        vl,vp=eval_epoch(model, val, criterion)
        gn=grad_norm(model); lr_=optimizer.param_groups[0]["lr"]; elapsed=time.time()-t0
        scheduler.step(vl)
        for k,v_ in [("epoch",ep),("train_loss",tl),("val_loss",vl),("train_ppl",tp),("val_ppl",vp),("lr",lr_),("grad_norm",gn),("epoch_time",elapsed)]:
            hist[k].append(v_)
        tag=""
        if vl<best_val:
            best_val=vl; wait=0
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),"optimizer_state":optimizer.state_dict(),"val_loss":vl,"val_ppl":vp,"cfg":cfg}, ckpt_path)
            tag="  ← BEST ✓"
        else: wait+=1
        print(f"  {ep:5d} │ {tl:10.4f} │ {vl:10.4f} │ {tp:9.2f} │ {vp:9.2f} │ {lr_:8.2e} │ {elapsed:.1f}s{tag}")
        if wait>=PATIENCE: print(f"\n  ⏹  Early stopping at epoch {ep}."); break
    print(f"  {'─'*80}")
    best_ep=hist["epoch"][hist["val_loss"].index(min(hist["val_loss"]))]
    print(f"  ✅  Best epoch    : {best_ep}")
    print(f"  ✅  Best val loss : {min(hist['val_loss']):.4f}")
    print(f"  ✅  Best val PPL  : {min(hist['val_ppl']):.2f}")
    pd.DataFrame(hist).to_csv(RES_DIR/"training_history.csv", index=False)
    return hist, best_val

history, best_val_loss = train_model(model, DEFAULT_CFG, train_loader, val_loader)

ep=history["epoch"]; best_ep=ep[history["val_loss"].index(min(history["val_loss"]))]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Training Dynamics", fontsize=15, fontweight="bold")
axes[0,0].plot(ep, history["train_loss"],"o-",color=PALETTE["primary"],lw=2,ms=4,label="Train")
axes[0,0].plot(ep, history["val_loss"],"s-",color=PALETTE["secondary"],lw=2,ms=4,label="Val")
axes[0,0].axvline(best_ep,color=PALETTE["success"],ls="--",lw=1.5,label=f"Best ep {best_ep}")
axes[0,0].fill_between(ep, history["train_loss"], history["val_loss"], alpha=0.08, color=PALETTE["warn"], label="Gen. gap")
axes[0,0].set_title("Loss"); axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss"); axes[0,0].legend()
axes[0,1].plot(ep, history["train_ppl"],"o-",color=PALETTE["primary"],lw=2,ms=4,label="Train PPL")
axes[0,1].plot(ep, history["val_ppl"],"s-",color=PALETTE["secondary"],lw=2,ms=4,label="Val PPL")
axes[0,1].axvline(best_ep,color=PALETTE["success"],ls="--",lw=1.5)
axes[0,1].set_title("Perplexity"); axes[0,1].set_xlabel("Epoch"); axes[0,1].legend()
axes[1,0].plot(ep, history["lr"],"D-",color=PALETTE["accent"],lw=1.8,ms=5)
axes[1,0].set_yscale("log"); axes[1,0].set_title("Learning Rate"); axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("LR (log)")
axes[1,1].plot(ep, history["grad_norm"],"^-",color=PALETTE["warn"],lw=1.8,ms=4,alpha=0.8)
axes[1,1].axhline(DEFAULT_CFG.clip,color="grey",ls=":",lw=1.5,label=f"clip={DEFAULT_CFG.clip}")
axes[1,1].set_title("Gradient Norm"); axes[1,1].set_xlabel("Epoch"); axes[1,1].legend()
plt.tight_layout()
plt.savefig(PLOT_DIR/"07_training_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾  Saved: 07_training_dynamics.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — HYPERPARAMETER TUNING (GRID SEARCH)                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

GRID_EPOCHS = 8
GRID = [
    (128, 256, 1, 1e-3, 0.2, 64),
    (128, 256, 1, 5e-4, 0.2, 64),
    (128, 256, 2, 1e-3, 0.2, 64),
    (128, 256, 2, 5e-4, 0.3, 64),
    (256, 512, 1, 1e-3, 0.2, 64),
    (256, 512, 1, 5e-4, 0.3, 64),
    (256, 512, 2, 1e-3, 0.3, 32),
    (256, 512, 2, 5e-4, 0.4, 32)
]
grid_results = []
print(f"  {len(GRID)} configurations × {GRID_EPOCHS} epochs\n")
print(f"  {'─'*88}")
print(f"  {'Cfg':>3} │ {'emb':>4} │ {'hid':>4} │ {'L':>2} │ {'lr':>6} │ {'dp':>4} │ {'bs':>3} │ {'BestValLoss':>12} │ {'PPL':>7} │ {'Params':>9}")
print(f"  {'─'*88}")

for i,(ed,hd,nl,lr_,dp,bs) in enumerate(GRID, 1):
    gc=ModelConfig(src_vocab_size=len(src_vocab),tgt_vocab_size=len(tgt_vocab),embed_dim=ed,
                   hidden_dim=hd,n_layers=nl,dropout=dp,lr=lr_,clip=1.0,epochs=GRID_EPOCHS,
                   batch_size=bs,label_smoothing=0.1)
    if bs!=BATCH_SIZE:
        gtr=DataLoader(train_dataset,bs,shuffle=True,collate_fn=collate_fn,num_workers=0)
        gva=DataLoader(val_dataset,  bs,shuffle=False,collate_fn=collate_fn,num_workers=0)
    else: gtr,gva=train_loader,val_loader
    gm=build_model(gc); gcr=LabelSmoothingLoss(len(tgt_vocab),0,0.1)
    go=optim.Adam(gm.parameters(),lr=lr_,betas=(0.9,0.98),eps=1e-8)
    gs=ReduceLROnPlateau(go,mode="min",factor=0.5,patience=2)
    vls,tls,etimes=[],[],[]
    for ep in range(1,GRID_EPOCHS+1):
        t0=time.time(); tl,_=train_epoch(gm,gtr,go,gcr,1.0); vl,_=eval_epoch(gm,gva,gcr)
        gs.step(vl); tls.append(tl); vls.append(vl); etimes.append(time.time()-t0)
    bvl=min(vls); bppl=math.exp(min(bvl,10)); np_=count_params(gm)
    print(f"  {i:3d} │ {ed:4d} │ {hd:4d} │ {nl:2d} │ {lr_:6.4f} │ {dp:4.1f} │ {bs:3d} │ {bvl:12.4f} │ {bppl:7.2f} │ {np_:9,}")
    grid_results.append({"config":i,"embed_dim":ed,"hidden_dim":hd,"n_layers":nl,"lr":lr_,"dropout":dp,
                          "batch_size":bs,"best_val_loss":round(bvl,5),"best_val_ppl":round(bppl,3),
                          "n_params":np_,"avg_epoch_time":round(sum(etimes)/len(etimes),2),
                          "train_losses":tls,"val_losses":vls})
    del gm,go,gcr
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print(f"  {'─'*88}")
grid_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ("train_losses","val_losses")} for r in grid_results]).sort_values("best_val_loss").reset_index(drop=True)
grid_df["rank"]=range(1,len(grid_df)+1)
print("\n  GRID SEARCH LEADERBOARD:")
display(grid_df)
grid_df.to_csv(RES_DIR/"grid_search_results.csv",index=False)
print("  Saved: grid_search_results.csv")

best_row=grid_df.iloc[0]
print("\n  ★  BEST CONFIG:")
for c in ["embed_dim","hidden_dim","n_layers","lr","dropout","batch_size","best_val_loss","best_val_ppl"]:
    print(f"    {c:<20}: {best_row[c]}")

BEST_CFG=ModelConfig(src_vocab_size=len(src_vocab),tgt_vocab_size=len(tgt_vocab),
    embed_dim=int(best_row["embed_dim"]),hidden_dim=int(best_row["hidden_dim"]),
    n_layers=int(best_row["n_layers"]),dropout=float(best_row["dropout"]),
    lr=float(best_row["lr"]),clip=1.0,epochs=30,batch_size=int(best_row["batch_size"]),label_smoothing=0.1)
if int(best_row["batch_size"])!=BATCH_SIZE:
    best_trn=DataLoader(train_dataset,int(best_row["batch_size"]),shuffle=True,collate_fn=collate_fn,num_workers=0)
    best_val=DataLoader(val_dataset,  int(best_row["batch_size"]),shuffle=False,collate_fn=collate_fn,num_workers=0)
else: best_trn,best_val=train_loader,val_loader
model=build_model(BEST_CFG)
print(f"\n  Retraining best config for full {BEST_CFG.epochs} epochs...")
best_history,_=train_model(model, BEST_CFG, best_trn, best_val, ckpt_name="best_model.pt")

fig, axes=plt.subplots(2,3,figsize=(18,10))
fig.suptitle("Hyperparameter Grid Search Analysis",fontsize=14,fontweight="bold")
bar_c=[PALETTE["success"] if i==0 else PALETTE["primary"] for i in range(len(grid_df))]
cfg_l=[f"C{int(r['config'])}\nE{int(r['embed_dim'])} H{int(r['hidden_dim'])}" for _,r in grid_df.iterrows()]
axes[0,0].barh(cfg_l[::-1],grid_df["best_val_loss"][::-1].values,color=bar_c[::-1],edgecolor="white")
axes[0,0].set_title("Validation Loss by Config (ranked)"); axes[0,0].set_xlabel("Best Val Loss")
for hd in sorted(grid_df["hidden_dim"].unique()):
    sub=grid_df[grid_df["hidden_dim"]==hd]
    axes[0,1].scatter(sub["n_params"],sub["best_val_loss"],label=f"hidden={int(hd)}",s=80,alpha=0.8)
axes[0,1].set_title("# Params vs Val Loss"); axes[0,1].set_xlabel("# Parameters"); axes[0,1].legend(fontsize=8)
for r in grid_results:
    axes[0,2].plot(range(1,len(r["val_losses"])+1),r["val_losses"],alpha=0.6,lw=1.5,label=f"C{r['config']}")
axes[0,2].set_title("Val Loss Curves (all configs)"); axes[0,2].set_xlabel("Epoch"); axes[0,2].legend(fontsize=6,ncol=2)
em_g=[grid_df[grid_df["embed_dim"]==ed]["best_val_loss"].values for ed in sorted(grid_df["embed_dim"].unique())]
axes[1,0].boxplot(em_g,labels=[str(int(e)) for e in sorted(grid_df["embed_dim"].unique())],patch_artist=True,
                  boxprops=dict(facecolor=PALETTE["light"]),medianprops=dict(color=PALETTE["warn"],lw=2))
axes[1,0].set_title("Effect of Embedding Dim"); axes[1,0].set_ylabel("Best Val Loss")
nl_g=[grid_df[grid_df["n_layers"]==nl]["best_val_loss"].values for nl in sorted(grid_df["n_layers"].unique())]
axes[1,1].boxplot(nl_g,labels=[str(int(n)) for n in sorted(grid_df["n_layers"].unique())],patch_artist=True,
                  boxprops=dict(facecolor=PALETTE["light"]),medianprops=dict(color=PALETTE["warn"],lw=2))
axes[1,1].set_title("Effect of # RNN Layers"); axes[1,1].set_ylabel("Best Val Loss")
sc=axes[1,2].scatter(grid_df["avg_epoch_time"],grid_df["best_val_loss"],c=grid_df["n_params"],cmap="plasma",s=100,alpha=0.85)
plt.colorbar(sc,ax=axes[1,2]).set_label("# Params",fontsize=8)
axes[1,2].set_title("Time vs Performance"); axes[1,2].set_xlabel("Avg epoch time (s)")
plt.tight_layout()
plt.savefig(PLOT_DIR/"08_hyperparameter_search.png",dpi=150,bbox_inches="tight")
plt.show()
print("💾  Saved: 08_hyperparameter_search.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — INFERENCE, DECODING & BLEU EVALUATION                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

ckpt=torch.load(CKPT_DIR/"best_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
print(f"  ✅  Checkpoint loaded (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, PPL={ckpt['val_ppl']:.2f})")

@torch.no_grad()
def greedy_decode(model, src_text, sv, tv, max_len=60):
    model.eval()
    src_ids=sv.encode(src_text, add_bos=False, add_eos=True)[:MAX_SRC_LEN]
    src_t=torch.tensor([src_ids],dtype=torch.long,device=DEVICE)
    _,hidden=model.encoder(src_t)
    tgt=[Vocabulary.BOS_IDX]; tok=torch.tensor([[Vocabulary.BOS_IDX]],dtype=torch.long,device=DEVICE)
    for _ in range(max_len):
        lg,hidden=model.decoder(tok,hidden)
        nxt=lg[0,0,:].argmax().item()
        if nxt==Vocabulary.EOS_IDX: break
        tgt.append(nxt); tok=torch.tensor([[nxt]],dtype=torch.long,device=DEVICE)
    return tv.decode(tgt[1:],skip_special=True)

@torch.no_grad()
def beam_search_decode(model, src_text, sv, tv, beam=4, max_len=60, lp=0.7):
    model.eval()
    src_ids=sv.encode(src_text, add_bos=False, add_eos=True)[:MAX_SRC_LEN]
    src_t=torch.tensor([src_ids],dtype=torch.long,device=DEVICE)
    _,hidden=model.encoder(src_t)
    beams=[(0.0,[Vocabulary.BOS_IDX],hidden)]; done=[]
    for _ in range(max_len):
        cands=[]
        for lp_,toks,h in beams:
            if toks[-1]==Vocabulary.EOS_IDX: done.append((lp_,toks)); continue
            lt=torch.tensor([[toks[-1]]],dtype=torch.long,device=DEVICE)
            lg,nh=model.decoder(lt,h)
            lps=F.log_softmax(lg[0,0,:],dim=-1)
            tk_lp,tk_id=lps.topk(beam)
            for l,t in zip(tk_lp.tolist(),tk_id.tolist()): cands.append((lp_+l,toks+[t],nh))
        if not cands: break
        cands.sort(key=lambda x:x[0],reverse=True); beams=cands[:beam]
        if all(t[-1]==Vocabulary.EOS_IDX for _,t,_ in beams):
            for l,t,_ in beams: done.append((l,t)); break
    if not done:
        for l,t,_ in beams: done.append((l,t))
    scored=[(s/((5+len(t))/6)**lp,t) for s,t in done]
    scored.sort(key=lambda x:x[0],reverse=True)
    out=[x for x in scored[0][1] if x not in (Vocabulary.BOS_IDX,Vocabulary.EOS_IDX)]
    return tv.decode(out,skip_special=True)

print("\n  DECODING DEMO (5 test samples)")
print("  " + "─"*80)
for i in range(5):
    row=test_df.iloc[i]; src,ref=row["eng_clean"],row["urdu_clean"]
    g=greedy_decode(model,src,src_vocab,tgt_vocab)
    b=beam_search_decode(model,src,src_vocab,tgt_vocab,beam=4)
    print(f"  Source   : {src[:80]}")
    print(f"  Reference: {ref[:80]}")
    print(f"  Greedy   : {g[:80]}")
    print(f"  Beam (4) : {b[:80]}\n")

N_EVAL=len(test_df)
refs,g_preds,b_preds,srcs=[],[],[],[]
g_times,b_times=[],[]
print(f"  Decoding {N_EVAL} test pairs...")
for i in tqdm(range(N_EVAL), desc="Decoding"):
    row=test_df.iloc[i]; src,ref=row["eng_clean"],row["urdu_clean"]
    t0=time.time(); gp=greedy_decode(model,src,src_vocab,tgt_vocab); g_times.append(time.time()-t0)
    t0=time.time(); bp=beam_search_decode(model,src,src_vocab,tgt_vocab); b_times.append(time.time()-t0)
    srcs.append(src); refs.append(ref); g_preds.append(gp); b_preds.append(bp)

smoothie=SmoothingFunction().method4
def bleu_scores(preds,refs):
    rt=[[r.split()] for r in refs]; pt=[p.split() for p in preds]
    return {f"BLEU-{n}":round(corpus_bleu(rt,pt,weights=tuple([1/n]*n+[0]*(4-n)),smoothing_function=smoothie)*100,3) for n in [1,2,3,4]}

g_bleu=bleu_scores(g_preds,refs); b_bleu=bleu_scores(b_preds,refs)
sent_g=[sentence_bleu([r.split()],p.split(),smoothing_function=smoothie)*100 for r,p in zip(refs,g_preds)]
sent_b=[sentence_bleu([r.split()],p.split(),smoothing_function=smoothie)*100 for r,p in zip(refs,b_preds)]
bleu_df=pd.DataFrame({"Decoding Method":["Greedy","Beam (k=4)"],**{k:[g_bleu[k],b_bleu[k]] for k in ["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]}})
print("\n  BLEU SCORES ON TEST SET:")
display(bleu_df)
bleu_df.to_csv(RES_DIR/"bleu_scores.csv",index=False)

trans_df=pd.DataFrame({"source":srcs,"reference":refs,"greedy":g_preds,"beam4":b_preds,"bleu_greedy":sent_g,"bleu_beam4":sent_b})
trans_df.to_csv(RES_DIR/"translation_examples.csv",index=False,encoding="utf-8-sig")
print(f"  Saved: bleu_scores.csv, translation_examples.csv")

b_labels=["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]
fig,axes=plt.subplots(2,3,figsize=(18,10))
fig.suptitle("Test-Set Evaluation: Greedy vs Beam Search",fontsize=14,fontweight="bold")
x=np.arange(4); w=0.35
b1=axes[0,0].bar(x-w/2,[g_bleu[b] for b in b_labels],w,label="Greedy",color=PALETTE["primary"],alpha=0.85,edgecolor="white")
b2=axes[0,0].bar(x+w/2,[b_bleu[b] for b in b_labels],w,label="Beam (k=4)",color=PALETTE["secondary"],alpha=0.85,edgecolor="white")
for bar in list(b1)+list(b2): h=bar.get_height(); axes[0,0].text(bar.get_x()+bar.get_width()/2,h+0.04,f"{h:.2f}",ha="center",fontsize=8,fontweight="bold")
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(b_labels); axes[0,0].set_title("Corpus BLEU"); axes[0,0].legend()
axes[0,1].hist(sent_g,bins=40,alpha=0.6,label="Greedy",color=PALETTE["primary"],density=True)
axes[0,1].hist(sent_b,bins=40,alpha=0.6,label="Beam-4",color=PALETTE["secondary"],density=True)
axes[0,1].set_title("Sentence BLEU Distribution"); axes[0,1].set_xlabel("Sentence BLEU"); axes[0,1].legend()
axes[0,2].scatter(test_df["eng_len"],sent_b,alpha=0.25,s=6,color=PALETTE["accent"])
z=np.polyfit(test_df["eng_len"],sent_b,1); p=np.poly1d(z)
xl=np.linspace(test_df["eng_len"].min(),test_df["eng_len"].max(),100)
axes[0,2].plot(xl,p(xl),color=PALETTE["warn"],lw=2,label="Trend")
axes[0,2].set_title("Source Length vs BLEU (Beam-4)"); axes[0,2].set_xlabel("Source length"); axes[0,2].set_ylabel("Sent BLEU"); axes[0,2].legend()
r_lens=[len(r.split()) for r in refs]; h_lens=[len(h.split()) for h in b_preds]
axes[1,0].scatter(r_lens,h_lens,alpha=0.2,s=6,c=sent_b,cmap="RdYlGn")
lm=max(max(r_lens),max(h_lens))+2; axes[1,0].plot([0,lm],[0,lm],"k--",lw=1,alpha=0.4)
axes[1,0].set_title("Reference vs Hypothesis Length"); axes[1,0].set_xlabel("Ref length"); axes[1,0].set_ylabel("Hyp length")
axes[1,1].boxplot([np.array(g_times)*1000,np.array(b_times)*1000],labels=["Greedy","Beam-4"],patch_artist=True,
                  boxprops=dict(facecolor=PALETTE["light"]),medianprops=dict(color=PALETTE["warn"],lw=2),notch=True)
axes[1,1].set_title("Decode Time / Sentence"); axes[1,1].set_ylabel("Time (ms)")
sg_=np.sort(sent_g); sb_=np.sort(sent_b); cdf=np.arange(1,len(sg_)+1)/len(sg_)
axes[1,2].plot(sg_,cdf,color=PALETTE["primary"],lw=2,label="Greedy")
axes[1,2].plot(sb_,cdf,color=PALETTE["secondary"],lw=2,label="Beam-4")
axes[1,2].set_title("CDF: Sentence BLEU"); axes[1,2].set_xlabel("Sentence BLEU"); axes[1,2].legend()
plt.tight_layout()
plt.savefig(PLOT_DIR/"09_bleu_evaluation.png",dpi=150,bbox_inches="tight")
plt.show()
print("💾  Saved: 09_bleu_evaluation.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 11 — ERROR ANALYSIS, OOD STATISTICS & RESEARCH DISCUSSION         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

ERROR_TYPES={
    "Repetition Loop":"Same token appears ≥3 times",
    "Complete Hallucination":"Zero lexical overlap with reference",
    "Severe Under-generation":"Hyp < 30% ref length",
    "Severe Over-generation":"Hyp > 200% ref length",
    "OOV Cascade":"≥3 <unk> in hypothesis",
    "Poor Reordering":"20–40% overlap, wrong order",
    "Partial Match":"20–40% word overlap",
    "Near Miss":"Sentence BLEU ≥ 10",
    "Acceptable":"Sentence BLEU ≥ 20",
}

def classify(src, ref, hyp, bleu):
    ht=hyp.split(); rt=ref.split()
    if len(ht)>4 and any(ht.count(t)>=3 for t in ht): return "Repetition Loop"
    ov=len(set(rt)&set(ht))/max(len(set(rt)),1)
    if ov==0.0 and len(ht)>0: return "Complete Hallucination"
    r=len(ht)/max(len(rt),1)
    if r<0.30: return "Severe Under-generation"
    if r>2.00: return "Severe Over-generation"
    if ht.count("<unk>")>=3: return "OOV Cascade"
    if bleu>=20.0: return "Acceptable"
    if bleu>=10.0: return "Near Miss"
    if ov>=0.4: return "Poor Reordering"
    if ov>=0.2: return "Partial Match"
    return "Complete Hallucination"

trans_df["error_type"]=[classify(s,r,h,bl) for s,r,h,bl in
                         zip(trans_df["source"],trans_df["reference"],trans_df["beam4"],trans_df["bleu_beam4"])]
err_counts=trans_df["error_type"].value_counts()
print("  ERROR DISTRIBUTION (Beam-4, full test set):")
print(f"  {'─'*56}")
print(f"  {'Error Type':<35} {'Count':>6}  {'%':>6}")
print(f"  {'─'*56}")
for err,cnt in err_counts.items():
    print(f"  {err:<35} {cnt:>6}  {cnt/len(trans_df)*100:>5.1f}%")

trans_df.to_csv(RES_DIR/"error_analysis.csv",index=False,encoding="utf-8-sig")
print("\n  Saved: error_analysis.csv")

print("\n  TOP-5 BEST TRANSLATIONS:")
for _,r in trans_df.nlargest(5,"bleu_beam4").iterrows():
    print(f"  [BLEU={r['bleu_beam4']:.2f} | {r['error_type']}]")
    print(f"    SRC : {r['source'][:80]}")
    print(f"    REF : {r['reference'][:80]}")
    print(f"    HYP : {r['beam4'][:80]}\n")

print("  5 WORST TRANSLATIONS:")
for _,r in trans_df.nsmallest(5,"bleu_beam4").iterrows():
    print(f"  [BLEU={r['bleu_beam4']:.2f} | {r['error_type']}]")
    print(f"    SRC : {r['source'][:80]}")
    print(f"    REF : {r['reference'][:80]}")
    print(f"    HYP : {r['beam4'][:80]}\n")

# OOD analysis
test_df2=test_df.copy()
test_df2["oov_count"]=test_df2["eng_clean"].apply(lambda x:sum(1 for w in x.split() if w not in src_vocab))
len_thresh=float(np.percentile(test_df2["eng_len"],90))
ood_mask=(test_df2["eng_len"]>len_thresh)|(test_df2["oov_count"]>=2)
ood_df=test_df2[ood_mask].reset_index(drop=True)
id_df=test_df2[~ood_mask].reset_index(drop=True)
print(f"  OOD threshold: length > {len_thresh:.0f} OR OOV >= 2")
print(f"  OOD samples: {len(ood_df):,}  |  In-dist: {len(id_df):,}")

ood_preds,ood_refs=[],[]
for _,row in tqdm(ood_df.iterrows(),total=len(ood_df),desc="OOD"):
    pred=beam_search_decode(model,row["eng_clean"],src_vocab,tgt_vocab)
    ood_preds.append(pred); ood_refs.append(row["urdu_clean"])
id_sample=id_df.sample(min(200,len(id_df)),random_state=SEED)
id_preds,id_refs=[],[]
for _,row in tqdm(id_sample.iterrows(),total=len(id_sample),desc="ID"):
    pred=beam_search_decode(model,row["eng_clean"],src_vocab,tgt_vocab)
    id_preds.append(pred); id_refs.append(row["urdu_clean"])

ood_b=bleu_scores(ood_preds,ood_refs); id_b=bleu_scores(id_preds,id_refs)
ood_table=pd.DataFrame({"Distribution":["In-Distribution","OOD"],**{k:[id_b[k],ood_b[k]] for k in ["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]}})
print("\n  OOD vs In-Distribution BLEU:")
display(ood_table)
for k in ["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]:
    print(f"  {k} performance drop (ID→OOD): {id_b[k]-ood_b[k]:+.3f}")

fig,axes=plt.subplots(2,3,figsize=(18,10))
fig.suptitle("Error Analysis & OOD Evaluation",fontsize=14,fontweight="bold")
en_=[err_counts.get(et,0) for et in ERROR_TYPES]; ec=plt.cm.Set2(np.linspace(0,1,len(ERROR_TYPES)))
axes[0,0].barh(list(ERROR_TYPES.keys())[::-1],[err_counts.get(et,0) for et in list(ERROR_TYPES.keys())][::-1],color=ec,edgecolor="white")
axes[0,0].set_title("Error Distribution (Beam-4)"); axes[0,0].set_xlabel("Count")
err_bleu=[trans_df[trans_df["error_type"]==et]["bleu_beam4"].values for et in ERROR_TYPES]
axes[0,1].boxplot([x for x in err_bleu if len(x)>0],labels=[et for et,x in zip(ERROR_TYPES,err_bleu) if len(x)>0],
                  patch_artist=True,boxprops=dict(facecolor=PALETTE["light"]),medianprops=dict(color=PALETTE["warn"],lw=2))
axes[0,1].tick_params(axis="x",rotation=45,labelsize=7); axes[0,1].set_title("BLEU by Error Type")
oov_c=[sum(1 for w in s.split() if w not in src_vocab) for s in trans_df["source"]]
axes[0,2].scatter(oov_c,trans_df["bleu_beam4"],alpha=0.3,s=8,color=PALETTE["primary"])
axes[0,2].set_title("OOV Count vs BLEU"); axes[0,2].set_xlabel("# OOV in source"); axes[0,2].set_ylabel("Sentence BLEU")
x_=np.arange(4); w=0.35
axes[1,0].bar(x_-w/2,[id_b[b] for b in b_labels],w,label="In-Dist",color=PALETTE["primary"],alpha=0.85,edgecolor="white")
axes[1,0].bar(x_+w/2,[ood_b[b] for b in b_labels],w,label="OOD",color=PALETTE["warn"],alpha=0.85,edgecolor="white")
axes[1,0].set_xticks(x_); axes[1,0].set_xticklabels(b_labels)
axes[1,0].set_title("In-Dist vs OOD BLEU"); axes[1,0].legend()
err_lens={et:[] for et in ERROR_TYPES}
for _,r in trans_df.iterrows(): err_lens[r["error_type"]].append(len(r["source"].split()))
vd=[err_lens[et] for et in ERROR_TYPES if err_lens[et]]
vl=[et for et in ERROR_TYPES if err_lens[et]]
vp=axes[1,1].violinplot(vd,showmedians=True)
for pc in vp["bodies"]: pc.set_facecolor(PALETTE["primary"]); pc.set_alpha(0.5)
axes[1,1].set_xticks(range(1,len(vl)+1)); axes[1,1].set_xticklabels(vl,rotation=45,fontsize=7)
axes[1,1].set_title("Source Length by Error Type"); axes[1,1].set_ylabel("# tokens")
axes[1,2].hist([len(h.split()) for h in trans_df["beam4"]],bins=40,color=PALETTE["primary"],alpha=0.8,edgecolor="white")
axes[1,2].set_title("Hypothesis Length Distribution"); axes[1,2].set_xlabel("# tokens"); axes[1,2].set_ylabel("Count")
plt.tight_layout()
plt.savefig(PLOT_DIR/"10_error_analysis.png",dpi=150,bbox_inches="tight")
plt.show()
print("💾  Saved: 10_error_analysis.png")

print("  LIMITATIONS OF VANILLA RNN FOR NMT")
print("  1. VANISHING GRADIENTS: tanh activations cause gradients to shrink over long sequences.")
print("  2. INFORMATION BOTTLENECK: The entire source compressed into h_T.")
print("  3. POOR REORDERING: English (SVO) → Urdu (SOV) requires long-range reordering.")
print("  4. REPETITION: Decoder gets trapped in probability loops.")
print("  5. OOV HANDLING: Word-level vocab maps rare tokens to <unk>.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 12 — FINAL EXPERIMENT SUMMARY & DASHBOARD                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

best_ep_idx=best_history["val_loss"].index(min(best_history["val_loss"]))
best_ep=best_history["epoch"][best_ep_idx]

print("=" * 76)
summary_rows = [
    ("── DATASET", ""),("Raw sentence pairs", f"{len(df_raw):,}"),
    ("After all cleaning", f"{len(df):,}"),
    ("Train / Val / Test", f"{len(train_df):,} / {len(val_df):,} / {len(test_df):,}"),
    ("── VOCABULARY", ""),("English vocab size", f"{len(src_vocab):,}"),
    ("Urdu vocab size", f"{len(tgt_vocab):,}"),("Min token frequency", "2"),
    ("── MODEL", ""),("Architecture","Vanilla RNN Encoder-Decoder (tanh)"),
    ("Embedding dim", BEST_CFG.embed_dim),("Hidden dim", BEST_CFG.hidden_dim),
    ("RNN layers", BEST_CFG.n_layers),("Dropout", BEST_CFG.dropout),
    ("Label smoothing ε", BEST_CFG.label_smoothing),("Total parameters", f"{count_params(model):,}"),
    ("── TRAINING",""),("Epochs trained", len(best_history["epoch"])),
    ("Best epoch", best_ep), ("Best val loss", f"{min(best_history['val_loss']):.4f}"),
    ("Best val PPL", f"{min(best_history['val_ppl']):.2f}"),
    ("── EVALUATION",""),
    ("Greedy BLEU-1", f"{g_bleu['BLEU-1']:.3f}"),("Greedy BLEU-4", f"{g_bleu['BLEU-4']:.3f}"),
    ("Beam-4 BLEU-1", f"{b_bleu['BLEU-1']:.3f}"),("Beam-4 BLEU-4", f"{b_bleu['BLEU-4']:.3f}"),
    ("── OOD",""),("OOD BLEU-4", f"{ood_b['BLEU-4']:.3f}"),
    ("ID  BLEU-4", f"{id_b['BLEU-4']:.3f}"),
    ("BLEU-4 drop (ID→OOD)", f"{id_b['BLEU-4']-ood_b['BLEU-4']:+.3f}"),
    ("── ERRORS",""),("Most common error", err_counts.index[0]),
    ("Acceptable translations", f"{err_counts.get('Acceptable',0)/len(trans_df)*100:.1f}%"),
]
for k,v in summary_rows:
    if "──" in str(k): print(f"\n  {k}")
    else: print(f"  {k:<35}: {v}")
print("\n" + "=" * 76)
pd.DataFrame({"Metric":[k for k,v in summary_rows if "──" not in str(k)],
              "Value":[v for k,v in summary_rows if "──" not in str(k)]}).to_csv(RES_DIR/"final_summary.csv",index=False)
print("  Saved: final_summary.csv")

print("\n  OUTPUT FILES:")
for p in sorted(PLOT_DIR.glob("*.png")): print(f"    outputs/plots/{p.name}")
for p in sorted(CKPT_DIR.glob("*.pt")):  print(f"    outputs/checkpoints/{p.name}")
for p in sorted(RES_DIR.glob("*")):      print(f"    outputs/results/{p.name}")

# Final dark dashboard
DARK={"bg":"#1C1C2E","ax":"#1E1E2F","grid":"#444466","t":"white","b1":"#82B4FF","b2":"#FF9F7B"}
fig=plt.figure(figsize=(18,12)); fig.patch.set_facecolor(DARK["bg"])
import matplotlib.gridspec as gridspec
gs=gridspec.GridSpec(2,4,figure=fig,hspace=0.45,wspace=0.38)
ax=[fig.add_subplot(gs[r,c]) for r in range(2) for c in range(4)]
for a in ax:
    a.set_facecolor(DARK["ax"]); a.tick_params(colors=DARK["t"])
    a.xaxis.label.set_color(DARK["t"]); a.yaxis.label.set_color(DARK["t"]); a.title.set_color(DARK["t"])
    for sp in a.spines.values(): sp.set_edgecolor(DARK["grid"])

ep_=best_history["epoch"]
ax[0].plot(ep_,best_history["train_loss"],lw=2,color=DARK["b1"],label="Train Loss")
ax[0].plot(ep_,best_history["val_loss"],lw=2,color=DARK["b2"],label="Val Loss")
ax[0].axvline(best_ep,color="#A8FFD0",ls="--",lw=1.5,label=f"Best ep {best_ep}")
ax[0].set_title("Training Loss"); ax[0].legend(labelcolor=DARK["t"],fontsize=8)

x_=np.arange(4); w=0.35
ax[1].bar(x_-w/2,[g_bleu[b] for b in b_labels],w,color=DARK["b1"],alpha=0.9,label="Greedy",edgecolor="none")
ax[1].bar(x_+w/2,[b_bleu[b] for b in b_labels],w,color=DARK["b2"],alpha=0.9,label="Beam-4",edgecolor="none")
ax[1].set_xticks(x_); ax[1].set_xticklabels(b_labels); ax[1].set_title("Corpus BLEU"); ax[1].legend(labelcolor=DARK["t"],fontsize=8)

ax[2].hist(sent_g,bins=40,density=True,alpha=0.65,color=DARK["b1"],label="Greedy")
ax[2].hist(sent_b,bins=40,density=True,alpha=0.65,color=DARK["b2"],label="Beam-4")
ax[2].set_title("Sentence BLEU Dist."); ax[2].legend(labelcolor=DARK["t"],fontsize=8)

ec_=[err_counts.get(et,0) for et in ERROR_TYPES]
wedges,_,atext=ax[3].pie(ec_,labels=None,colors=plt.cm.Set2(np.linspace(0,1,len(ERROR_TYPES))),
                          autopct="%1.0f%%",startangle=140,wedgeprops={"edgecolor":"none"},
                          textprops={"color":DARK["t"],"fontsize":7})
ax[3].legend(wedges,list(ERROR_TYPES.keys()),loc="center left",bbox_to_anchor=(1,0,0.5,1),
             fontsize=6,labelcolor=DARK["t"],facecolor=DARK["ax"])
ax[3].set_title("Error Distribution")

ax[4].bar(x_-w/2,[id_b[b] for b in b_labels],w,color=DARK["b1"],alpha=0.9,label="ID",edgecolor="none")
ax[4].bar(x_+w/2,[ood_b[b] for b in b_labels],w,color="#FF6B6B",alpha=0.9,label="OOD",edgecolor="none")
ax[4].set_xticks(x_); ax[4].set_xticklabels(b_labels); ax[4].set_title("ID vs OOD BLEU"); ax[4].legend(labelcolor=DARK["t"],fontsize=8)

for a in ax[5:]: a.set_visible(False)

fig.suptitle("English→Urdu NMT | Vanilla RNN Encoder-Decoder | Final Dashboard",fontsize=13,fontweight="bold",color=DARK["t"],y=1.01)
plt.savefig(PLOT_DIR/"11_final_dashboard.png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.show()
print("💾  Saved: 11_final_dashboard.png")
print("\n  ✅  Pipeline complete. All 11 plots and all CSV/pkl files generated.")

