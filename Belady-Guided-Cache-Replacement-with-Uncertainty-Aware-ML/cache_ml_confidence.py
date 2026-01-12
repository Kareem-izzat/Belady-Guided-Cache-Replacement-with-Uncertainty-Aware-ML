"""
ML-based cache with confidence gating (uncertainty-aware replacement).
This is the key contribution: when model confidence is low, fall back to LRU.
"""

import pickle
import numpy as np
from collections import deque, defaultdict
from cache_config import NUM_SETS, ASSOCIATIVITY, BLOCK_SIZE

def get_block_address(addr):
    """Get cache block address by masking off offset bits"""
    return addr & ~(BLOCK_SIZE - 1)

def get_set_index(addr):
    """Extract set index from address"""
    block_addr = get_block_address(addr)
    return (block_addr // BLOCK_SIZE) % NUM_SETS

def get_tag(addr):
    """Extract tag from address"""
    block_addr = get_block_address(addr)
    return block_addr // (BLOCK_SIZE * NUM_SETS)

class CacheMLConfidence:
    """
    Cache using ML predictions with confidence gating.
    
    Key innovation: Uses model's prediction confidence (probability) as a trust metric.
    - High confidence (>= threshold): Use ML prediction
    - Low confidence (< threshold): Fall back to LRU
    
    This handles uncertainty in the model's predictions.
    """
    
    def __init__(self, num_sets=NUM_SETS, associativity=ASSOCIATIVITY, 
                 model_path='cache_model_improved.pkl', confidence_threshold=0.70, use_ml=True):
        """
        Initialize cache with ML model and confidence gating.
        
        Args:
            num_sets: Number of cache sets
            associativity: Ways per set
            model_path: Path to trained RandomForest model
            confidence_threshold: Minimum confidence to trust ML (0.0-1.0)
            use_ml: If False, use pure LRU (for baseline comparison)
        """
        self.num_sets = num_sets
        self.associativity = associativity
        self.use_ml = use_ml
        
        # Load trained model only if using ML
        if use_ml:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = None
        
        self.confidence_threshold = confidence_threshold
        
        # Cache structure: list of sets, each set is a list of blocks
        self.cache_sets = [[] for _ in range(num_sets)]
        
        # LRU tracking: deque per set for recency ordering
        self.lru_queues = [deque() for _ in range(num_sets)]
        
        # Feature tracking for ML predictions
        self.recency_counter = 0  # Global counter for recency
        self.last_access_time = {}  # block_addr -> last access time
        self.access_frequency = defaultdict(int)  # block_addr -> access count
        self.first_access_time = {}  # block_addr -> first access time
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.ml_decisions = 0  # Decisions made by ML
        self.lru_fallbacks = 0  # Decisions made by LRU fallback
        self.confidence_scores = []  # Track all confidence scores
        self.last_used_ml = False  # Track last eviction decision
    
    def _compute_features(self, block_addr):
        """Compute features for ML prediction."""
        # Recency: time since last access (0 if never accessed)
        if block_addr in self.last_access_time:
            recency = self.recency_counter - self.last_access_time[block_addr]
        else:
            recency = 0
        
        # Frequency: total access count
        frequency = self.access_frequency[block_addr]
        
        # Age: time since first access (0 if never accessed)
        if block_addr in self.first_access_time:
            age = self.recency_counter - self.first_access_time[block_addr]
        else:
            age = 0
        
        return [recency, frequency, age]
    
    def _update_lru(self, set_idx, block_addr):
        """Update LRU queue for a set."""
        lru_queue = self.lru_queues[set_idx]
        
        # Remove block if already in queue
        if block_addr in lru_queue:
            lru_queue.remove(block_addr)
        
        # Add to front (most recently used)
        lru_queue.append(block_addr)
    
    def _get_lru_victim(self, set_idx):
        """Get LRU victim from a set."""
        return self.lru_queues[set_idx][0]  # Front of deque = LRU
    
    def _ml_select_victim(self, set_idx, cache_set):
        """
        Use ML model with confidence gating to select victim.
        
        Returns:
            (victim_block_addr, used_ml, confidence)
        """
        # If ML disabled, always use LRU
        if not self.use_ml or self.model is None:
            victim = self._get_lru_victim(set_idx)
            self.lru_fallbacks += 1
            self.last_used_ml = False
            return victim, False, 0.0
        
        # Compute features for all blocks in set
        features = []
        block_addrs = []
        for block_addr, tag in cache_set:
            feat = self._compute_features(block_addr)
            features.append(feat)
            block_addrs.append(block_addr)
        
        features_array = np.array(features)
        
        # Get prediction probabilities (confidence scores)
        # predict_proba returns [prob_evict, prob_keep] for each block
        probabilities = self.model.predict_proba(features_array)
        
        # prob_evict = probabilities[:, 0]  # Probability of evicting
        prob_keep = probabilities[:, 1]  # Probability of keeping
        
        # Find block with lowest "keep" probability = highest eviction priority
        evict_idx = np.argmin(prob_keep)
        confidence = 1.0 - prob_keep[evict_idx]  # Confidence in evicting this block
        
        # CONFIDENCE GATE: Check if we trust this prediction
        if confidence >= self.confidence_threshold:
            # High confidence: use ML prediction
            self.ml_decisions += 1
            victim = block_addrs[evict_idx]
            used_ml = True
            self.last_used_ml = True
        else:
            # Low confidence: fall back to LRU
            self.lru_fallbacks += 1
            victim = self._get_lru_victim(set_idx)
            used_ml = False
            self.last_used_ml = False
        
        self.confidence_scores.append(confidence)
        
        return victim, used_ml, confidence
    
    def access(self, address):
        """
        Access cache with ML + confidence gating.
        
        Returns:
            (hit, decision_info) where decision_info contains eviction details
        """
        self.recency_counter += 1
        
        block_addr = get_block_address(address)
        set_idx = get_set_index(address)
        tag = get_tag(address)
        
        cache_set = self.cache_sets[set_idx]
        
        # Update access tracking
        self.last_access_time[block_addr] = self.recency_counter
        self.access_frequency[block_addr] += 1
        if block_addr not in self.first_access_time:
            self.first_access_time[block_addr] = self.recency_counter
        
        # Update LRU queue
        self._update_lru(set_idx, block_addr)
        
        # Check if hit
        for i, (cached_block, cached_tag) in enumerate(cache_set):
            if cached_block == block_addr:
                self.hits += 1
                return True, {'hit': True}
        
        # Miss
        self.misses += 1
        decision_info = {'hit': False, 'eviction': False}
        
        # If set not full, just add
        if len(cache_set) < ASSOCIATIVITY:
            cache_set.append((block_addr, tag))
        else:
            # Set full: need to evict using ML + confidence gate
            victim, used_ml, confidence = self._ml_select_victim(set_idx, cache_set)
            
            # Remove victim
            cache_set[:] = [(b, t) for b, t in cache_set if b != victim]
            
            # Remove from LRU queue
            self.lru_queues[set_idx].remove(victim)
            
            # Add new block
            cache_set.append((block_addr, tag))
            
            self.evictions += 1
            decision_info = {
                'hit': False,
                'eviction': True,
                'victim': victim,
                'used_ml': used_ml,
                'confidence': confidence
            }
        
        return False, decision_info
    
    def stats(self):
        """Return cache statistics."""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
        
        avg_confidence = np.mean(self.confidence_scores) if self.confidence_scores else 0
        min_confidence = np.min(self.confidence_scores) if self.confidence_scores else 0
        max_confidence = np.max(self.confidence_scores) if self.confidence_scores else 0
        
        ml_ratio = self.ml_decisions / (self.ml_decisions + self.lru_fallbacks) if self.evictions > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'ml_decisions': self.ml_decisions,
            'lru_fallbacks': self.lru_fallbacks,
            'ml_ratio': ml_ratio,
            'confidence_threshold': self.confidence_threshold,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
        }
