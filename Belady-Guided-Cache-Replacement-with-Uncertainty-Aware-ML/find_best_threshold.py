"""
FIND BEST CONFIDENCE THRESHOLD
Test multiple thresholds to maximize performance while minimizing degradations
"""

import os
import sys
from cache_lru_fixed import CacheLRUFixed
from cache_ml_hybrid import CacheMLHybrid
from trace_reader import read_champsim_trace

# Test different confidence thresholds
THRESHOLDS_TO_TEST = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88]

def evaluate_threshold(threshold, trace_files, max_accesses=1_000_000):
    """Evaluate model with specific confidence threshold across all traces"""
    print(f"\n{'='*80}")
    print(f"TESTING THRESHOLD: {threshold:.2f}")
    print(f"{'='*80}")
    
    results = []
    total_lru = 0
    total_hybrid = 0
    degradations = 0
    
    for trace_file in trace_files:
        trace_path = os.path.join("traces", trace_file)
        addresses = read_champsim_trace(trace_path, max_accesses=max_accesses)
        
        # LRU baseline
        lru_cache = CacheLRUFixed()
        for addr in addresses:
            lru_cache.access(addr)
        lru_rate = lru_cache.stats()['hit_rate']
        
        # Hybrid ML with this threshold
        hybrid_cache = CacheMLHybrid(confidence_threshold=threshold)
        for addr in addresses:
            hybrid_cache.access(addr)
        hybrid_rate = hybrid_cache.stats()['hit_rate']
        
        improvement = hybrid_rate - lru_rate
        if improvement < 0:
            degradations += 1
        
        total_lru += lru_rate
        total_hybrid += hybrid_rate
        
        results.append({
            'trace': trace_file,
            'lru': lru_rate,
            'hybrid': hybrid_rate,
            'improvement': improvement
        })
        
        print(f"{trace_file:30s} LRU: {lru_rate:6.2f}%  Hybrid: {hybrid_rate:6.2f}%  Δ: {improvement:+7.2f}%")
    
    avg_improvement = (total_hybrid - total_lru) / len(trace_files)
    
    print(f"\n{'='*80}")
    print(f"THRESHOLD {threshold:.2f} SUMMARY:")
    print(f"  Average improvement: {avg_improvement:+.2f}%")
    print(f"  Degradations: {degradations}/{len(trace_files)}")
    print(f"{'='*80}")
    
    return {
        'threshold': threshold,
        'avg_improvement': avg_improvement,
        'degradations': degradations,
        'results': results
    }

def main():
    print("="*80)
    print("FINDING OPTIMAL CONFIDENCE THRESHOLD")
    print("="*80)
    
    # All trace files
    trace_files = [
        "scenario1_mixed.trace",
        "scenario2_hostile.trace",
        "scenario3_friendly.trace",
        "scenario4_scan_heavy.trace",
        "scenario5_phases.trace",
        "scenario6_alternating.trace",
        "scenario7_mixed_diff.trace",
        "scenario8_gradual.trace",
        "scenario9_bursts.trace",
        "scenario10_noisy.trace"
    ]
    
    # Test all thresholds
    all_results = []
    for threshold in THRESHOLDS_TO_TEST:
        result = evaluate_threshold(threshold, trace_files)
        all_results.append(result)
    
    # Find best threshold
    print("\n" + "="*80)
    print("COMPARISON OF ALL THRESHOLDS")
    print("="*80)
    print(f"{'Threshold':<12} {'Avg Improvement':<18} {'Degradations':<15} {'Status'}")
    print("-"*80)
    
    best_result = None
    best_score = float('-inf')
    
    for result in all_results:
        # Score: favor high improvement, penalize degradations heavily
        score = result['avg_improvement'] - (result['degradations'] * 5.0)
        
        status = "✅ Good" if result['degradations'] == 0 else f"⚠️  {result['degradations']} failures"
        print(f"{result['threshold']:<12.2f} {result['avg_improvement']:>+7.2f}%         {result['degradations']:>2d}/10           {status}")
        
        if score > best_score:
            best_score = score
            best_result = result
    
    print("-"*80)
    print(f"\n🎯 BEST THRESHOLD: {best_result['threshold']:.2f}")
    print(f"   Average improvement: {best_result['avg_improvement']:+.2f}%")
    print(f"   Degradations: {best_result['degradations']}/10")
    print(f"   Score: {best_score:.2f}")
    print("\n💡 RECOMMENDATION:")
    if best_result['degradations'] == 0:
        print(f"   Use threshold {best_result['threshold']:.2f} - no degradations!")
    else:
        # Find highest threshold with no degradations
        safe_results = [r for r in all_results if r['degradations'] == 0]
        if safe_results:
            safest = max(safe_results, key=lambda x: x['avg_improvement'])
            print(f"   For production: Use {safest['threshold']:.2f} (safe, {safest['avg_improvement']:+.2f}% avg)")
            print(f"   For research: Use {best_result['threshold']:.2f} (risky, {best_result['avg_improvement']:+.2f}% avg)")
        else:
            print(f"   All thresholds have degradations. Use {best_result['threshold']:.2f} for best overall performance")

if __name__ == "__main__":
    main()
