# Online Retrainig Module

This folder contains a standalone, optional real-time online update system for recommendation boosting.

It does **not** modify any existing project code.

## What it does

- Ingests user events in real time (for example, `add_to_cart`).
- Updates item-item co-cart strength online (`A + B -> stronger pair`).
- Produces a ranking boost score for candidate items.
- Supports:
  - in-memory mode (no infrastructure)
  - Redis mode (recommended for production)

## Why this helps

You can improve ranking freshness instantly, without full model retraining:

`final_score = model_score + alpha * online_boost`

Where `online_boost` is derived from recently updated co-cart links.

## Files

- `config.py`: runtime settings
- `online_update_engine.py`: online co-cart updater and scorer
- `simulate_events.py`: local demo runner
- `bra_realtime_demo.py`: bra-only ranking before/after add-to-cart from exported recommendation CSV
- `requirements.txt`: optional dependency list

## Quick start (in-memory)

Run from this folder:

```powershell
python simulate_events.py --mode memory
```

## Bra-only ranking with add-to-cart impact

This uses your exported file from `ranking_pipeline/recommendation_exports` and keeps only bra items.

Run:

```powershell
python bra_realtime_demo.py --user_id USER_0063
```

What you get:

- Console table for bra ranking before add-to-cart
- Console table for bra ranking after add-to-cart on top recommended bra
- Output CSVs in `online retrainig/outputs`:
  - `bra_ranking_before_USER_XXXX.csv`
  - `bra_ranking_after_USER_XXXX.csv`

## Quick start (Redis)

1. Start Redis.
2. Set environment variables if needed:

```powershell
$env:ONLINE_RT_REDIS_URL = "redis://localhost:6379/0"
```

3. Run:

```powershell
python simulate_events.py --mode redis
```

## Integration idea (later)

Without changing your training pipeline, you can combine scores at serving time:

1. Get model scores from current ranker.
2. Get online boosts from this module.
3. Blend them using `alpha`.

Example:

```python
blended = model_score + 0.15 * online_boost
```

## Notes

- Co-cart counts support time decay to emphasize fresh behavior.
- Use key TTL in Redis to control memory growth.
- This is designed as an add-on service and can run as a separate process.
