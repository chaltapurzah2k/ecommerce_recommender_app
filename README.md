# Ecommerce Recommender App

A production-style recommendation project for fashion e-commerce that combines:

- Hybrid candidate retrieval (embedding + behavior signals)
- Learning-to-rank with LightGBM/XGBoost
- Neural Two-Tower retrieval (PyTorch + FAISS)
- App and dashboard layers for experimentation

This repo is designed so you can run quickly with mock data, then switch to real datasets and embeddings.

---

## Highlights

- End-to-end ranking pipeline with temporal split and ranking metrics
- Two-Tower dual encoder with sequence-aware user modeling
- Multi-modal item encoding (image + text embeddings)
- FAISS retrieval support with numpy fallback
- Cold-start fallback and practical ranking features

---

## Project Layout

```text
.
├── app.py
├── streamlit_app.py
├── dashboard_app.py
├── data/
├── embeddings/
├── ranking_pipeline/
│   ├── run_pipeline.py
│   ├── candidate_generator.py
│   ├── feature_engineering.py
│   ├── train_ranker.py
│   └── inference.py
├── two-tower_recommendation_system/
│   ├── run_two_tower.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── retrieval.py
│   └── integration.py
└── requirements.txt
```

---

## Data Expectations

### Products file
Expected columns:

- `item_id`
- `product_name`
- `category`
- `brand`
- `price`
- Optional: `popularity_score`, `ctr_score`, `in_stock`

### User events file
Expected columns:

- `user_id`
- `item_id`
- `event_type` (`view`, `click`, `add_to_cart`, `purchase`)
- `timestamp`

### Embeddings
Pickle files that can be loaded as:

- `item_id -> vector` dict, or
- array-like payloads supported by the runner converters

---

## Setup

### 1. Create and activate environment

```bash
conda create -n ecommerce_env python=3.10 -y
conda activate ecommerce_env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you use ranking models and they are missing:

```bash
pip install lightgbm xgboost
```

---

## Quick Start

### A) Ranking pipeline (candidate retrieval + ranker)

Runs with mock data if real file paths are not provided.

```bash
python ranking_pipeline/run_pipeline.py --model both
```

With real files:

```bash
python ranking_pipeline/run_pipeline.py \
  --products data/products_export.csv \
  --events data/user_events.csv \
  --image_emb path/to/image_embeddings.pkl \
  --text_emb path/to/text_embeddings.pkl \
  --model both
```

What it does:

1. Builds top-N candidates from embeddings
2. Engineers user/item/match features
3. Trains LightGBM/XGBoost rankers with temporal split
4. Prints ranking metrics and top recommendations

---

### B) Two-Tower neural retrieval pipeline

Quick smoke test:

```bash
python two-tower_recommendation_system/run_two_tower.py --quick --epochs 1 --batch-size 64 --top-k 5 --top-n 50
```

With real files:

```bash
python two-tower_recommendation_system/run_two_tower.py \
  --products data/products_export.csv \
  --events data/user_events.csv \
  --image-emb path/to/image_embeddings.pkl \
  --text-emb path/to/text_embeddings.pkl \
  --epochs 5 --batch-size 128 --top-k 10 --top-n 200
```

What it does:

1. Trains a Two-Tower model (Transformer user tower + multimodal item tower)
2. Evaluates retrieval with `Recall@K` and `NDCG@K`
3. Builds retrieval index and returns top recommendations

---

## Model Architecture Overview

### Stage 1: Retrieval

- Baseline: embedding similarity retrieval in `ranking_pipeline/candidate_generator.py`
- Neural: learned user/item embeddings in `two-tower_recommendation_system/model.py`

### Stage 2: Ranking

- Feature engineering in `ranking_pipeline/feature_engineering.py`
- Rankers in `ranking_pipeline/train_ranker.py`
- Inference in `ranking_pipeline/inference.py`

### Stage 3: Integration

- Neural similarity can be added as a ranking feature through
  `two-tower_recommendation_system/integration.py`

---

## Metrics

Ranking pipeline reports:

- `Precision@10`
- `Recall@10`
- `MAP@10`
- `NDCG@10`
- `Hit Rate@10`

Two-Tower training reports:

- `Recall@K`
- `NDCG@K`

---

## Apps

Depending on your workflow, you can run app layers such as:

```bash
streamlit run streamlit_app.py
```

To automatically use any free port without manually specifying one:

```bash
python run_streamlit.py
```

To launch a different Streamlit app the same way:

```bash
python run_streamlit.py embedding_recommender_app.py
```

and/or other Flask dashboard entry points in the root.

---

## Troubleshooting

- FAISS import issues:
  - Use `faiss-cpu` from `requirements.txt`
  - Disable FAISS in Two-Tower runner with `--no-faiss`

- Missing LightGBM/XGBoost:
  - Install with `pip install lightgbm xgboost`

- Empty train/validation split:
  - Verify timestamp parsing and event volume

- Sparse embedding coverage:
  - Ensure embedding keys match `item_id` values

---

## Next Improvements

- Add automatic handoff from Two-Tower retrieval to trained ranker in a single command
- Add model checkpoint registry and experiment tracking
- Add online feature store and real-time serving hooks

---

Built for iterative experimentation: start simple, measure, and progressively improve retrieval + ranking quality.
