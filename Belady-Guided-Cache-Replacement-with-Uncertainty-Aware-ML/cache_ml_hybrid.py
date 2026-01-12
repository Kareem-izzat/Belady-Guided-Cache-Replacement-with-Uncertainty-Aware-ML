"""
HYBRID CACHE IMPLEMENTATION
- Uses single Decision Tree (not Random Forest)  
- Only 2 features: Recency + Frequency
- Confidence gating with LRU fallback
- Optimized for hardware implementation
"""

import pickle
import numpy as np
from collections import deque, defaultdict
from cache_config import NUM_SETS, ASSOCIATIVITY, BLOCK_SIZE
from cache_utils import get_block_address, get_set_index, get_tag

class CacheMLHybrid:
    """
    Hybrid ML Cache with hardware-friendly design:
    - Single decision tree (not random forest)
    - 2 features instead of 3 (Recency, Frequency) 
    - Confidence gating
    - Simple integer arithmetic only
    """
    
    def __init__(self, model_path='cache_model_hybrid.pkl', confidence_threshold=0.88):
        # Load the decision tree model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Loaded hybrid model: depth={self.model.get_depth()}, leaves={self.model.get_n_leaves()}")
        except FileNotFoundError:
            print(f"❌ Model file {model_path} not found. Train model first!")
            raise
        
        self.confidence_threshold = confidence_threshold
        
        # Cache state
        self.cache_sets = [[] for _ in range(NUM_SETS)]
        
        # Feature tracking (simplified - only 2 features)
        self.last_access_time = {}
        self.access_frequency = defaultdict(int)
        self.current_time = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.accesses = 0
        self.ml_evictions = 0
        self.lru_evictions = 0
    
    def _extract_features(self, tag):
        """Extract 2 features for ML prediction"""
        # Feature 1: Recency (time since last access)
        recency = self.current_time - self.last_access_time.get(tag, 0)
        
        # Feature 2: Frequency (access count)
        frequency = self.access_frequency.get(tag, 0)
        
        return [recency, frequency]
    
    def _get_eviction_victim_ml(self, cache_set):
        """Use ML model with confidence gating to select eviction victim"""
        tags = [tag for tag, _ in cache_set]
        
        # Extract features for all blocks
        features = np.array([self._extract_features(tag) for tag in tags])
        
        # Get ML predictions and probabilities
        probabilities = self.model.predict_proba(features)
        
        # Get confidence (max probability for each prediction)
        confidences = np.max(probabilities, axis=1)
        max_confidence = np.max(confidences)
        
        # Check if ML is confident enough
        if max_confidence >= self.confidence_threshold:
            # Use ML prediction - evict block with highest eviction probability
            evict_probs = probabilities[:, 1]  # Probability of class 1 (evict)
            victim_idx = np.argmax(evict_probs)
            self.ml_evictions += 1
            return victim_idx
        else:
            # Fall back to LRU (evict first = oldest)
            self.lru_evictions += 1
            return 0
    
    def access(self, address):
        """Access cache at given address"""
        self.accesses += 1
        self.current_time += 1
        
        block_addr = get_block_address(address)
        set_idx = get_set_index(address)
        tag = get_tag(address)
        
        cache_set = self.cache_sets[set_idx]
        
        # Update access tracking
        self.last_access_time[tag] = self.current_time
        self.access_frequency[tag] += 1
        
        # Check for hit
        for i, (cached_tag, _) in enumerate(cache_set):
            if cached_tag == tag:
                # Hit: move to end (MRU position)
                cache_set.append(cache_set.pop(i))
                self.hits += 1
                return True
        
        # Miss
        self.misses += 1
        
        # Add new block
        if len(cache_set) >= ASSOCIATIVITY:
            # Use ML model with confidence gating
            victim_idx = self._get_eviction_victim_ml(cache_set)
            cache_set.pop(victim_idx)
        
        # Add to MRU position
        cache_set.append((tag, block_addr))
        return False
    
    def stats(self):
        """Return cache statistics including ML usage"""
        if self.accesses == 0:
            return {
                'hits': 0,
                'misses': 0,
                'accesses': 0,
                'hit_rate': 0.0,
                'ml_usage': 0.0
            }
        
        total_evictions = self.ml_evictions + self.lru_evictions
        ml_usage = (self.ml_evictions / total_evictions * 100) if total_evictions > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'accesses': self.accesses,
            'hit_rate': (self.hits / self.accesses) * 100,
            'ml_usage': ml_usage,
            'ml_evictions': self.ml_evictions,
            'lru_evictions': self.lru_evictions
        }
