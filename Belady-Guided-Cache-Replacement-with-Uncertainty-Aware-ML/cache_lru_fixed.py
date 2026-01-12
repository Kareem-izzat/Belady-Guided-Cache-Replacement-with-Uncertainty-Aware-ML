"""
Standard LRU Cache Implementation - Baseline for comparison
"""
from collections import deque
from cache_config import NUM_SETS, ASSOCIATIVITY, BLOCK_SIZE
from cache_utils import get_block_address, get_set_index, get_tag

class CacheLRUFixed:
    """Standard LRU cache with fixed configuration"""
    
    def __init__(self):
        # Cache state: each set is a list of (tag, data) tuples
        self.cache_sets = [[] for _ in range(NUM_SETS)]
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.accesses = 0
    
    def access(self, address):
        """Access cache at given address"""
        self.accesses += 1
        
        block_addr = get_block_address(address)
        set_idx = get_set_index(address)
        tag = get_tag(address)
        
        cache_set = self.cache_sets[set_idx]
        
        # Check for hit
        for i, (cached_tag, _) in enumerate(cache_set):
            if cached_tag == tag:
                # Hit: move to MRU position (end of list)
                cache_set.append(cache_set.pop(i))
                self.hits += 1
                return True
        
        # Miss
        self.misses += 1
        
        # Add new block
        if len(cache_set) >= ASSOCIATIVITY:
            # Evict LRU (first element)
            cache_set.pop(0)
        
        # Add to MRU position (end)
        cache_set.append((tag, block_addr))
        return False
    
    def stats(self):
        """Return cache statistics"""
        if self.accesses == 0:
            return {
                'hits': 0,
                'misses': 0,
                'accesses': 0,
                'hit_rate': 0.0
            }
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'accesses': self.accesses,
            'hit_rate': (self.hits / self.accesses) * 100
        }
