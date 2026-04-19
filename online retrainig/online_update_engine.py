from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config import OnlineUpdateConfig

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


@dataclass
class UserEvent:
    user_id: str
    event_type: str
    item_id: str
    timestamp: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hours_between(a: datetime, b: datetime) -> float:
    return abs((a - b).total_seconds()) / 3600.0


class InMemoryOnlineStore:
    def __init__(self, max_history_per_user: int = 50):
        self.user_recent_items: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_per_user))
        self.co_cart_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        self.co_cart_last_updated: Dict[Tuple[str, str], datetime] = {}

    @staticmethod
    def _pair(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def add_user_item(self, user_id: str, item_id: str) -> None:
        self.user_recent_items[user_id].append(item_id)

    def get_user_recent_items(self, user_id: str) -> List[str]:
        return list(self.user_recent_items[user_id])

    def add_pair_weight(self, item_a: str, item_b: str, delta: float, now: datetime) -> None:
        if item_a == item_b:
            return
        key = self._pair(item_a, item_b)
        self.co_cart_weights[key] += delta
        self.co_cart_last_updated[key] = now

    def get_pair_weight_and_time(self, item_a: str, item_b: str) -> Tuple[float, Optional[datetime]]:
        key = self._pair(item_a, item_b)
        return self.co_cart_weights.get(key, 0.0), self.co_cart_last_updated.get(key)


class RedisOnlineStore:
    def __init__(self, redis_url: str, max_history_per_user: int, ttl_seconds: int):
        if redis is None:
            raise RuntimeError("redis package is not installed. Install with: pip install redis")

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.max_history_per_user = max_history_per_user
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _pair(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        p = RedisOnlineStore._pair(a, b)
        return f"co_cart:{p[0]}::{p[1]}"

    @staticmethod
    def _user_history_key(user_id: str) -> str:
        return f"user_recent:{user_id}"

    def add_user_item(self, user_id: str, item_id: str) -> None:
        key = self._user_history_key(user_id)
        p = self.client.pipeline()
        p.rpush(key, item_id)
        p.ltrim(key, -self.max_history_per_user, -1)
        p.expire(key, self.ttl_seconds)
        p.execute()

    def get_user_recent_items(self, user_id: str) -> List[str]:
        key = self._user_history_key(user_id)
        return self.client.lrange(key, 0, -1)

    def add_pair_weight(self, item_a: str, item_b: str, delta: float, now: datetime) -> None:
        if item_a == item_b:
            return

        key = self._pair_key(item_a, item_b)
        now_ts = now.timestamp()

        p = self.client.pipeline()
        p.hincrbyfloat(key, "weight", float(delta))
        p.hset(key, mapping={"last_updated_ts": now_ts})
        p.expire(key, self.ttl_seconds)
        p.execute()

    def get_pair_weight_and_time(self, item_a: str, item_b: str) -> Tuple[float, Optional[datetime]]:
        key = self._pair_key(item_a, item_b)
        out = self.client.hgetall(key)
        if not out:
            return 0.0, None

        weight = float(out.get("weight", 0.0))
        ts = out.get("last_updated_ts")
        if ts is None:
            return weight, None

        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return weight, dt


class OnlineCoCartUpdater:
    def __init__(self, config: OnlineUpdateConfig):
        self.config = config
        if config.mode == "redis":
            self.store = RedisOnlineStore(
                redis_url=config.redis_url,
                max_history_per_user=config.max_history_per_user,
                ttl_seconds=config.co_cart_ttl_seconds,
            )
        else:
            self.store = InMemoryOnlineStore(max_history_per_user=config.max_history_per_user)

    def ingest_event(self, event: UserEvent) -> None:
        # For ranking freshness, only cart actions update co-cart links.
        if event.event_type != "add_to_cart":
            return

        recent_items = self.store.get_user_recent_items(event.user_id)
        for prev_item in recent_items:
            if prev_item == event.item_id:
                continue
            self.store.add_pair_weight(prev_item, event.item_id, delta=1.0, now=event.timestamp)

        self.store.add_user_item(event.user_id, event.item_id)

    def online_boost(self, user_cart_items: Sequence[str], candidate_item: str, now: Optional[datetime] = None) -> float:
        if now is None:
            now = _utc_now()

        boost = 0.0
        for cart_item in user_cart_items:
            weight, updated_at = self.store.get_pair_weight_and_time(cart_item, candidate_item)
            if weight <= 0:
                continue

            if updated_at is not None:
                age_h = _hours_between(now, updated_at)
                decay = self.config.decay_per_hour ** age_h
            else:
                decay = 1.0

            boost += weight * decay

        return float(boost)

    def blend_scores(
        self,
        user_cart_items: Sequence[str],
        candidates: Iterable[Tuple[str, float]],
        now: Optional[datetime] = None,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Returns rows of:
        (item_id, model_score, online_boost, blended_score)
        """
        out: List[Tuple[str, float, float, float]] = []
        for item_id, model_score in candidates:
            boost = self.online_boost(user_cart_items, item_id, now=now)
            blended = float(model_score) + self.config.alpha * boost
            out.append((item_id, float(model_score), boost, blended))

        out.sort(key=lambda x: x[3], reverse=True)
        return out
