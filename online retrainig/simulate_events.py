from datetime import timedelta

from config import OnlineUpdateConfig
from online_update_engine import OnlineCoCartUpdater, UserEvent, _utc_now


def main() -> None:
    cfg = OnlineUpdateConfig()
    updater = OnlineCoCartUpdater(cfg)

    now = _utc_now()

    # Simulate stream events: user adds A then B, which strengthens A<->B.
    events = [
        UserEvent(user_id="U1", event_type="add_to_cart", item_id="ITEM_A", timestamp=now - timedelta(minutes=3)),
        UserEvent(user_id="U1", event_type="add_to_cart", item_id="ITEM_B", timestamp=now - timedelta(minutes=2)),
        UserEvent(user_id="U2", event_type="add_to_cart", item_id="ITEM_A", timestamp=now - timedelta(minutes=2)),
        UserEvent(user_id="U2", event_type="add_to_cart", item_id="ITEM_B", timestamp=now - timedelta(minutes=1)),
        UserEvent(user_id="U3", event_type="add_to_cart", item_id="ITEM_A", timestamp=now - timedelta(minutes=1)),
        UserEvent(user_id="U3", event_type="add_to_cart", item_id="ITEM_C", timestamp=now),
    ]

    for e in events:
        updater.ingest_event(e)

    # Pretend these are model scores from your existing ranker.
    user_cart_items = ["ITEM_A"]
    base_candidates = [
        ("ITEM_B", 0.62),
        ("ITEM_C", 0.63),
        ("ITEM_D", 0.65),
    ]

    blended = updater.blend_scores(user_cart_items=user_cart_items, candidates=base_candidates, now=now)

    print("\nBlended ranking (item_id, model_score, online_boost, blended_score):")
    for row in blended:
        print(row)


if __name__ == "__main__":
    main()
