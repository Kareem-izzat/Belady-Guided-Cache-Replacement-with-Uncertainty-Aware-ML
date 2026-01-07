from collections import defaultdict, deque
from cache_utils import get_block_address
import math
from cache_config import NUM_SETS, ASSOCIATIVITY, BLOCK_SIZE

def build_future_positions(trace_addrs):
    """
    For each block address, store a queue of future access positions.
    """
    future_pos = defaultdict(deque)
    for i, addr in enumerate(trace_addrs):
        block = get_block_address(addr)
        future_pos[block].append(i)
    return future_pos


BLOCK_OFFSET_BITS = int(math.log2(BLOCK_SIZE))

def get_set_index_from_block(block_addr):
    return block_addr & (NUM_SETS - 1)

def belady_dataset(trace_addrs, max_accesses=200_000):
    trace_addrs = trace_addrs[:max_accesses]
    print(f"Building future positions for {len(trace_addrs)} addresses...")
    future_pos = build_future_positions(trace_addrs)
    print(f"Starting cache simulation...")

    cache_sets = [set() for _ in range(NUM_SETS)]
    last_access = {}
    frequency = defaultdict(int)
    inserted_at = {}

    X = []
    y = []
    
    evictions = 0
    hits = 0
    misses = 0

    for t, addr in enumerate(trace_addrs):
        if t % 1000 == 0 and t > 0:
            print(f"Processed {t}/{len(trace_addrs)} accesses... hits={hits}, misses={misses}, evictions={evictions}, samples={len(X)}")
        block = get_block_address(addr)
        set_idx = get_set_index_from_block(block)

        frequency[block] += 1

        if future_pos[block] and future_pos[block][0] == t:
            future_pos[block].popleft()

        # HIT
        if block in cache_sets[set_idx]:
            hits += 1
            last_access[block] = t
            continue

        # MISS
        misses += 1
        
        # MISS, free space
        if len(cache_sets[set_idx]) < ASSOCIATIVITY:
            cache_sets[set_idx].add(block)
            inserted_at[block] = t
            last_access[block] = t
            continue

        # MISS, eviction needed
        evictions += 1
        candidates = list(cache_sets[set_idx])

        def next_use(b):
            q = future_pos[b]
            return q[0] if q else float("inf")

        victim = max(candidates, key=next_use)

        # Generate training samples: for each candidate, predict if it will be evicted
        for c in candidates:
            recency = t - last_access.get(c, t)
            freq = frequency.get(c, 0)
            age = t - inserted_at.get(c, t)

            X.append([recency, freq, age])
            label = 0 if c == victim else 1
            y.append(label)

        cache_sets[set_idx].remove(victim)
        cache_sets[set_idx].add(block)
        inserted_at[block] = t
        last_access[block] = t
    
    print(f"Final stats: hits={hits}, misses={misses}, evictions={evictions}, samples={len(X)}")

    return X, y
