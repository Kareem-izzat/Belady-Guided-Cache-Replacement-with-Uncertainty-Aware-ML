"""
COMPREHENSIVE EVALUATION ON ALL TRACES
Tests LRU vs Hybrid ML with different access counts
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from trace_reader import read_champsim_trace
from cache_lru_fixed import CacheLRUFixed
from cache_ml_hybrid import CacheMLHybrid

def test_cache(CacheClass, addresses, **kwargs):
    cache = CacheClass(**kwargs) if kwargs else CacheClass()
    for addr in addresses:
        cache.access(addr)
    return cache.stats()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    traces = [
        ("Scenario 1: Mixed", "scenario1_mixed.trace", 1_000_000),
        ("Scenario 2: Hostile", "scenario2_hostile.trace", 1_000_000),
        ("Scenario 3: Friendly", "scenario3_friendly.trace", 1_000_000),
        ("Scenario 4: Scan Heavy", "scenario4_scan_heavy.trace", 1_000_000),
        ("Scenario 5: Phases", "scenario5_phases.trace", 1_000_000),
        ("Scenario 6: Alternating", "scenario6_alternating.trace", 1_000_000),
        ("Scenario 7: Mixed Difficulty", "scenario7_mixed_diff.trace", 1_000_000),
        ("Scenario 8: Gradual Shift", "scenario8_gradual.trace", 1_000_000),
        ("Scenario 9: Random Bursts", "scenario9_bursts.trace", 1_000_000),
        ("Scenario 10: Noisy Patterns", "scenario10_noisy.trace", 1_000_000)
    ]
    
    # Use 10 scenario traces for comprehensive evaluation
    test_sizes = None  # Will use the max_accesses specified per trace
    
    all_results = []
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION: LRU vs HYBRID ML")
    print("="*80)
    
    for trace_name, trace_file, max_accesses in traces:
        trace_path = os.path.join(script_dir, "traces", trace_file)
        
        if not os.path.exists(trace_path):
            print(f"\nSkipping {trace_name} - not found")
            continue
        
        print(f"\n{'='*30} {trace_name} {'='*30}")
        print(f"Loading {max_accesses:,} accesses from {trace_file}...")
        
        trace_results = {'name': trace_name, 'sizes': [], 'lru': [], 'hybrid': []}
        
        # Load trace
        addresses = read_champsim_trace(trace_path, max_accesses=max_accesses)
        print(f"Loaded {len(addresses):,} addresses")
        
        actual_size = len(addresses)
        
        # Test LRU
        print("Running LRU...")
        lru_stats = test_cache(CacheLRUFixed, addresses)
        lru_hit = lru_stats['hit_rate']
        
        # Test Hybrid ML
        print("Running Hybrid ML...")
        hybrid_stats = test_cache(CacheMLHybrid, addresses)
        hybrid_hit = hybrid_stats['hit_rate']
        ml_usage = hybrid_stats['ml_usage']
        
        improvement = hybrid_hit - lru_hit
        
        print(f"\n📊 Results for {len(addresses):,} accesses:")
        print(f"   LRU:    {lru_hit:6.2f}%")
        print(f"   Hybrid: {hybrid_hit:6.2f}% (ML active: {ml_usage:.0f}%)")
        print(f"   Δ:      {improvement:+6.2f}%")
        
        trace_results['sizes'].append(actual_size)
        trace_results['lru'].append(lru_hit)
        trace_results['hybrid'].append(hybrid_hit)
        
        all_results.append(trace_results)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Best Results per Trace")
    print("="*80)
    print(f"{'Trace':<20} {'LRU':>10} {'Hybrid':>10} {'Improvement':>12} {'Status'}")
    print("-"*80)
    
    all_lru = []
    all_hybrid = []
    
    for result in all_results:
        name = result['name']
        lru_hit = result['lru'][-1]
        hybrid_hit = result['hybrid'][-1]
        improvement = hybrid_hit - lru_hit
        status = "✅ Better" if improvement >= 0 else "❌ Worse"
        
        all_lru.append(lru_hit)
        all_hybrid.append(hybrid_hit)
        
        print(f"{name:<20} {lru_hit:>9.2f}% {hybrid_hit:>9.2f}% {improvement:>11.2f}% {status}")
    
    # Average
    avg_lru = np.mean(all_lru)
    avg_hybrid = np.mean(all_hybrid)
    avg_improvement = avg_hybrid - avg_lru
    
    print("-"*80)
    print(f"{'AVERAGE':<20} {avg_lru:>9.2f}% {avg_hybrid:>9.2f}% {avg_improvement:>11.2f}%")
    print("="*80)
    
    # Analysis
    print(f"\n📊 ANALYSIS:")
    if avg_improvement > 2:
        print(f"✅ Hybrid ML shows significant improvement!")
        print(f"   Recommendation: Use hybrid approach")
    elif avg_improvement > 0:
        print(f"⚠️  Hybrid ML shows modest improvement")
        print(f"   Recommendation: Consider workload characteristics")
    else:
        print(f"❌ Hybrid ML shows degradation")
        print(f"   Recommendation: Use standard LRU")
    
    print(f"\n💾 HARDWARE CONSIDERATION:")
    print(f"   LRU baseline hit rate: {avg_lru:.1f}%")
    if avg_lru > 90:
        print(f"   For traces with such high hit rates, complex ML may not justify")
        print(f"   the additional hardware cost (area, power, latency)")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_results))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, all_lru, width, label='LRU', color='steelblue')
    bars2 = ax.bar(x + width/2, all_hybrid, width, label='Hybrid ML', color='coral')
    
    ax.set_xlabel('Trace Scenario', fontsize=12)
    ax.set_ylabel('Hit Rate (%)', fontsize=12)
    ax.set_title('LRU vs Hybrid ML Cache Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in all_results], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(script_dir, 'all_traces_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Visualization saved: all_traces_comparison.png")
    
if __name__ == "__main__":
    main()
