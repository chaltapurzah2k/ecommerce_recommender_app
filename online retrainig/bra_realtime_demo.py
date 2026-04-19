from __future__ import annotations

import argparse
import csv
import os
from datetime import timedelta
from typing import Dict, List, Sequence, Tuple

from config import OnlineUpdateConfig
from online_update_engine import OnlineCoCartUpdater, UserEvent, _utc_now


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_user_recommendation_csv(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _is_bra_item(row: Dict[str, str]) -> bool:
    name = (row.get("product_name") or "").lower()
    category = (row.get("category") or "").lower()
    return ("bra" in name) or ("bra" in category)


def _extract_bra_candidates(rows: Sequence[Dict[str, str]]) -> List[Tuple[str, float, str]]:
    out: List[Tuple[str, float, str]] = []
    for row in rows:
        if (row.get("row_type") or "recommendation") != "recommendation":
            continue
        if not _is_bra_item(row):
            continue

        item_id = row.get("item_id") or ""
        if not item_id:
            continue

        base_score = _to_float((row.get("ranking_score") or "").strip(), default=0.0)
        name = row.get("product_name") or item_id
        out.append((item_id, base_score, name))

    return out


def _extract_cart_items(rows: Sequence[Dict[str, str]]) -> List[str]:
    cart = []
    for row in rows:
        if (row.get("row_type") or "") == "cart_item":
            item_id = row.get("item_id") or ""
            if item_id:
                cart.append(item_id)
    return cart


def _seed_global_patterns(updater: OnlineCoCartUpdater, bra_item_ids: Sequence[str]) -> None:
    now = _utc_now()
    if len(bra_item_ids) < 2:
        return

    # Create synthetic stream patterns so online boosts become visible immediately.
    for idx in range(25):
        u = f"SEED_U_{idx:03d}"
        a = bra_item_ids[idx % len(bra_item_ids)]
        b = bra_item_ids[(idx + 1) % len(bra_item_ids)]
        updater.ingest_event(UserEvent(user_id=u, event_type="add_to_cart", item_id=a, timestamp=now - timedelta(minutes=3)))
        updater.ingest_event(UserEvent(user_id=u, event_type="add_to_cart", item_id=b, timestamp=now - timedelta(minutes=2)))


def _print_table(title: str, rows: Sequence[Tuple[str, float, float, float]], names: Dict[str, str]) -> None:
    print("\n" + title)
    print("-" * len(title))
    print(f"{'rank':<6}{'item_id':<16}{'product_name':<45}{'model':>10}{'boost':>10}{'blended':>12}")
    for i, (item_id, model_score, boost, blended) in enumerate(rows, start=1):
        name = names.get(item_id, item_id)
        print(f"{i:<6}{item_id:<16}{name[:44]:<45}{model_score:>10.4f}{boost:>10.4f}{blended:>12.4f}")


def _write_output_csv(path: str, rows: Sequence[Tuple[str, float, float, float]], names: Dict[str, str], phase: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "rank", "item_id", "product_name", "model_score", "online_boost", "blended_score"])
        for i, (item_id, model_score, boost, blended) in enumerate(rows, start=1):
            writer.writerow([phase, i, item_id, names.get(item_id, item_id), model_score, boost, blended])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bra-only real-time re-ranking demo")
    p.add_argument("--user_id", required=True, help="User id, e.g. USER_0063")
    p.add_argument(
        "--csv_dir",
        default=os.path.join("..", "ranking_pipeline", "recommendation_exports"),
        help="Directory containing recommendation_for_USER_XXXX.csv",
    )
    p.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to save before/after ranking CSV snapshots",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = os.path.join(args.csv_dir, f"recommendation_for_{args.user_id}.csv")
    rows = _load_user_recommendation_csv(csv_path)

    bra_candidates = _extract_bra_candidates(rows)
    if not bra_candidates:
        print("No bra recommendation rows found in the source CSV.")
        return

    candidate_pairs = [(item_id, score) for item_id, score, _ in bra_candidates]
    names = {item_id: name for item_id, _, name in bra_candidates}
    bra_item_ids = [item_id for item_id, _, _ in bra_candidates]

    cart_items = _extract_cart_items(rows)
    if not cart_items:
        # If no explicit cart rows exist, emulate one from the first bra item.
        cart_items = [bra_item_ids[0]]

    cfg = OnlineUpdateConfig(mode="memory")
    updater = OnlineCoCartUpdater(cfg)

    # Seed synthetic global co-cart updates so boosts are visible in demo output.
    _seed_global_patterns(updater, bra_item_ids)

    now = _utc_now()

    before = updater.blend_scores(user_cart_items=cart_items, candidates=candidate_pairs, now=now)
    _print_table("BRA RANKING BEFORE ADD-TO-CART", before, names)

    clicked_item = before[0][0]
    print(f"\nSimulating add-to-cart click on recommended bra item: {clicked_item}")

    # Reconstruct user short history then ingest the click event.
    for i, item in enumerate(cart_items):
        updater.ingest_event(
            UserEvent(
                user_id=args.user_id,
                event_type="add_to_cart",
                item_id=item,
                timestamp=now - timedelta(minutes=(len(cart_items) - i + 1)),
            )
        )

    updater.ingest_event(
        UserEvent(
            user_id=args.user_id,
            event_type="add_to_cart",
            item_id=clicked_item,
            timestamp=now,
        )
    )

    cart_after = list(cart_items) + [clicked_item]
    after = updater.blend_scores(user_cart_items=cart_after, candidates=candidate_pairs, now=now)
    _print_table("BRA RANKING AFTER ADD-TO-CART", after, names)

    before_out = os.path.join(args.output_dir, f"bra_ranking_before_{args.user_id}.csv")
    after_out = os.path.join(args.output_dir, f"bra_ranking_after_{args.user_id}.csv")
    _write_output_csv(before_out, before, names, phase="before_add_to_cart")
    _write_output_csv(after_out, after, names, phase="after_add_to_cart")

    print(f"\nSaved: {before_out}")
    print(f"Saved: {after_out}")


if __name__ == "__main__":
    main()
