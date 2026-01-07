from collections import deque
from cache_config import NUM_SETS, ASSOCIATIVITY
from cache_utils import get_set_index, get_tag

class CacheLRU:
    def __init__(self):
        self.cache = [
            deque(maxlen=ASSOCIATIVITY) for _ in range(NUM_SETS)
        ]
        self.hits = 0
        self.misses = 0

    def access(self, addr):
        set_idx = get_set_index(addr)
        tag = get_tag(addr)
        cache_set = self.cache[set_idx]

        if tag in cache_set:
            # HIT
            self.hits += 1
            cache_set.remove(tag)
            cache_set.appendleft(tag)
        else:
            # MISS
            self.misses += 1
            if len(cache_set) == ASSOCIATIVITY:
                cache_set.pop()
            cache_set.appendleft(tag)

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0
        }
