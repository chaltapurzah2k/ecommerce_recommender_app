import os
from dataclasses import dataclass


@dataclass
class OnlineUpdateConfig:
    mode: str = os.getenv("ONLINE_RT_MODE", "memory")  # memory or redis
    redis_url: str = os.getenv("ONLINE_RT_REDIS_URL", "redis://localhost:6379/0")
    co_cart_ttl_seconds: int = int(os.getenv("ONLINE_RT_CO_CART_TTL", "604800"))  # 7 days
    decay_per_hour: float = float(os.getenv("ONLINE_RT_DECAY_PER_HOUR", "0.995"))
    max_history_per_user: int = int(os.getenv("ONLINE_RT_MAX_HISTORY_PER_USER", "50"))
    alpha: float = float(os.getenv("ONLINE_RT_ALPHA", "0.15"))
