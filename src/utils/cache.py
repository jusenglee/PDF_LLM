import time
import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

class ResponseCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = OrderedDict()  # LRU 캐시로 사용
        self.timestamps = {}        # 타임스탬프 저장
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            # monotonic 시계 사용 (시스템 시간 변경에 영향 받지 않음)
            if time.monotonic() - self.timestamps[key] < self.ttl:
                # LRU 업데이트 - 항목 재배치
                value = self.cache.pop(key)
                self.cache[key] = value  # 맨 뒤로 이동
                self.hits += 1
                return value
            else:
                # TTL 만료된 항목 제거
                self.cache.pop(key, None)
                self.timestamps.pop(key, None)

        self.misses += 1
        return None

    async def put(self, key: str, value: str):
        # LRU 캐시 크기 제한 (삽입 전 확인)
        if len(self.cache) >= self.max_size and key not in self.cache:
            # 가장 오래 사용되지 않은 항목 제거 (OrderedDict의 첫 항목)
            oldest_key, _ = self.cache.popitem(last=False)
            self.timestamps.pop(oldest_key, None)

        # 항목 추가/갱신
        self.cache[key] = value
        self.timestamps[key] = time.monotonic()  # monotonic 시계 사용

    async def close(self):
        self.cache.clear()
        self.timestamps.clear()

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.3f}",
            "cache_size": len(self.cache)
        }
