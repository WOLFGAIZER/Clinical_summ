# Automated Clinical CRF Filling — CL4Health 2026

A fully local, privacy-preserving pipeline for automated Clinical Report Form (CRF) extraction from emergency department notes. Submitted as part of the **CL4Health 2026 Shared Task**.

---

## Overview

This system extracts structured values for **134 CRF items** from unstructured clinical notes using a **Three-Tier Fusion Architecture** combining:

- 🔍 **FAISS-based per-CRF evidence retrieval**
- 📝 **Longitudinal narrative generation** from EHR timelines
- 🤖 **Local LLM inference** (Llama 3 8B via Ollama) with schema micro-batching
- 🛡️ **Post-hoc Waterfall Calibration** to suppress hallucinated predictions

> **Best Result: Macro F1 (y+n) = 0.3196** on the 80-patient development set — a +232% improvement over the uncalibrated baseline.

---

## Project Structure

```
├── src/
│   ├── pipeline/
│   │   ├── main.py           # Orchestrator — full pipeline execution
│   │   ├── calibrate.py      # Post-hoc Waterfall Gating Calibrator
│   │   └── evaluate.py       # Evaluation metrics (Macro F1, Precision, Recall)
│   ├── generation/
│   │   ├── generator.py      # LLM prompt + JSON repair (Llama 3 8B)
│   │   └── query_generator.py
│   ├── retrieval/
│   │   └── vector_store.py   # FAISS index + re-ranking
│   ├── embeddings/
│   │   └── embedder.py       # Dense event embeddings
│   ├── events/
│   │   └── parser.py         # WTTS event parser
│   └── preprocess/
│       ├── wtts_builder.py   # Weighted Temporal Token Sequence builder
│       └── data_loader.py    # Dataset ingestion
├── data/
│   ├── raw/                  # Clinical notes + ground truth (not included)
│   └── processed/            # Cached WTTS, narratives, predictions
├── outputs/                  # Per-patient checkpoint JSONs
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and run Ollama
```bash
# Install Ollama from https://ollama.com
ollama pull llama3:8b
```

### 3. Add your data
Place the dataset files in:
```
data/raw/dyspnea-clinical-notes/
data/raw/dyspnea-crf-development/
data/raw/dev_gt.jsonl
```

---

## Running the Pipeline

### Step 1 — Run extraction (all patients)
```bash
python -m src.pipeline.main
```
Checkpoints are saved to `outputs/{pid}.json` after each patient. Restart safely — completed patients are skipped automatically.

### Step 2 — Calibrate predictions
```bash
python -m src.pipeline.calibrate
```
Applies the Waterfall Gating Filter and saves to `data/processed/calibrated_predictions.json`.

### Step 3 — Evaluate
```bash
python -m src.pipeline.evaluate --preds data/processed/calibrated_predictions.json
```

---

## Key Results

| Configuration | Macro F1 (y+n) | Precision (y) | Recall (y) |
|---|---|---|---|
| Uncalibrated baseline | 0.0962 | 0.09 | 0.67 |
| **Calibrated (final)** | **0.3196** | **0.53** | **0.39** |

---

## Architecture Highlights

| Component | Description |
|---|---|
| **WTTS Builder** | Builds a timestamped, weighted clinical event sequence per patient |
| **Per-CRF Retrieval** | FAISS search (k=2) per item — eliminates evidence starvation |
| **Temporal Expansion** | Pulls n-1, n, n+1 neighbors for causal context |
| **Schema Batching** | 134 items split into 40-item micro-batches per LLM call |
| **JSON Armor** | Two-layer repair: truncation recovery + regex cleanup |
| **Waterfall Calibrator** | Three sequential gates: P_item / C_model / E_str |

---

## Hardware

Tested on a **local laptop CPU** with no GPU acceleration.  
Model: `Llama 3 (8B)` via Ollama.  
Average runtime: ~5–6 hours for 80 patients (with narrative + WTTS caching on subsequent runs).

---

## License

For academic use only. Dataset not included — available from the CL4Health 2026 task organizers.
