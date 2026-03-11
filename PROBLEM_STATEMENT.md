# Problem Statement — English → Urdu Neural Machine Translation

## Overview

Build a **Neural Machine Translation (NMT)** system that translates English sentences into Urdu using a **Vanilla RNN Encoder-Decoder** architecture. The goal is to understand the mechanics of sequence-to-sequence models, vocabulary construction, and evaluation metrics by implementing everything from scratch with pure PyTorch.

---

## Objective

Implement a fully functional `English → Urdu` NMT pipeline that covers:
1. Data preprocessing and exploration
2. Vocabulary construction
3. Sequence encoding and batching
4. Vanilla RNN model design
5. Training with teacher forcing
6. Hyperparameter tuning (grid search)
7. Inference with greedy and beam search decoding
8. BLEU score evaluation
9. Error analysis

---

## Dataset

**English-Urdu Parallel Corpus**
- Source: [Kaggle — English to Urdu Translation Dataset](https://www.kaggle.com/datasets/muhammadmoinansari/english-to-urdu-translation-dataset)
- Format: `.xlsx` with two columns: `eng` (English), `urdu` (Urdu)
- Size: ~9,103 sentence pairs

Place at: `data/english_to_urdu_dataset.xlsx`

---

## Required Tasks

| # | Task | Marks |
|---|------|-------|
| 1 | Data Preprocessing | 10 |
| 2 | Vocabulary Construction | 10 |
| 3 | Sequence Encoding & Batching | 10 |
| 4 | Model Implementation (Vanilla RNN) | 20 |
| 5 | Training Loop | 15 |
| 6 | Hyperparameter Tuning | 10 |
| 7 | Greedy Decoding | 5 |
| 8 | Beam Search Decoding | 10 |
| 9 | BLEU Evaluation | 5 |
| 10 | Error Analysis | 5 |
| **Total** | | **100** |

---

## Technical Constraints

- ✅ Must use **Vanilla RNN** (`nn.RNN` with tanh nonlinearity)
- ❌ No LSTM, GRU, or Transformer layers allowed
- ❌ No pre-trained embeddings or models
- ✅ Must implement teacher forcing during training
- ✅ Must implement both greedy and beam search decoding
- ✅ Must report BLEU-1 through BLEU-4 on the test set
- ✅ Fixed random seed = 42 for reproducibility

---

## Deliverables

1. **Jupyter Notebook** (`notebooks/english_to_urdu_nmt.ipynb`) — main pipeline
2. **EDA Notebook** (`notebooks/dataset_statistics.ipynb`) — dataset statistics
3. **Standalone Script** (`src/english_to_urdu_nmt.py`) — runnable Python script
4. **Prompts File** (`prompts.txt`) — AI prompts used during development
5. **README.md** — project overview and walkthrough
6. **Outputs** — plots, CSVs, model checkpoint

---

## Evaluation Criteria

- Code quality and documentation
- Correct implementation of vanilla RNN (no forbidden architectures)
- Quality of preprocessing pipeline
- Training stability and convergence
- BLEU score performance
- Depth of error analysis

---

## Reference

Assignment: Generative AI — Spring 2026
