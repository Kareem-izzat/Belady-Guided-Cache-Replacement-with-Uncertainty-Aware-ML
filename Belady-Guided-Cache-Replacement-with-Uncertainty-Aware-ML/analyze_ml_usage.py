"""
DETAILED ANALYSIS: ML Usage and Confidence Correlation
Analyzes:
1. % of decisions using ML vs LRU per trace
2. Correlation between confidence and improvement
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
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
    
    # Test different confidence thresholds
    confidence_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88]
    
    print("="*80)
    print("ML USAGE ANALYSIS: % of decisions using ML vs LRU per trace")
    print("="*80)
    
    # Part 1: ML Usage Analysis at default confidence (0.88)
    print("\n📊 Part 1: ML Usage per Trace (Confidence Threshold = 0.88)")
    print("-"*80)
    print(f"{'Trace':<25} {'ML Decisions':>15} {'LRU Decisions':>15} {'ML %':>10}")
    print("-"*80)
    
    ml_usage_data = []
    improvement_data = []
    
    for trace_name, trace_file, max_accesses in traces:
        trace_path = os.path.join(script_dir, "traces", trace_file)
        
        if not os.path.exists(trace_path):
            continue
        
        addresses = read_champsim_trace(trace_path, max_accesses=max_accesses)
        
        # Test LRU
        lru_stats = test_cache(CacheLRUFixed, addresses)
        lru_hit = lru_stats['hit_rate']
        
        # Test Hybrid ML
        hybrid_stats = test_cache(CacheMLHybrid, addresses)
        hybrid_hit = hybrid_stats['hit_rate']
        ml_usage = hybrid_stats['ml_usage']
        ml_evictions = hybrid_stats['ml_evictions']
        lru_evictions = hybrid_stats['lru_evictions']
        
        improvement = hybrid_hit - lru_hit
        
        print(f"{trace_name:<25} {ml_evictions:>15,} {lru_evictions:>15,} {ml_usage:>9.1f}%")
        
        ml_usage_data.append(ml_usage)
        improvement_data.append(improvement)
    
    print("\n" + "="*80)
    print("CONFIDENCE-IMPROVEMENT CORRELATION ANALYSIS")
    print("="*80)
    
    # Part 2: Test different confidence thresholds and measure correlation
    print("\n📊 Part 2: Testing Different Confidence Thresholds")
    print("-"*80)
    
    # Collect data for correlation analysis
    all_ml_usage = []
    all_improvements = []
    all_thresholds = []
    
    for threshold in confidence_thresholds:
        print(f"\n🔍 Testing Confidence Threshold: {threshold:.2f}")
        print(f"{'Trace':<25} {'ML %':>10} {'Improvement':>12}")
        print("-"*60)
        
        threshold_ml_usage = []
        threshold_improvements = []
        
        for trace_name, trace_file, max_accesses in traces:
            trace_path = os.path.join(script_dir, "traces", trace_file)
            
            if not os.path.exists(trace_path):
                continue
            
            addresses = read_champsim_trace(trace_path, max_accesses=max_accesses)
            
            # Test LRU
            lru_stats = test_cache(CacheLRUFixed, addresses)
            lru_hit = lru_stats['hit_rate']
            
            # Test Hybrid ML with specific confidence threshold
            hybrid_stats = test_cache(CacheMLHybrid, addresses, confidence_threshold=threshold)
            hybrid_hit = hybrid_stats['hit_rate']
            ml_usage = hybrid_stats['ml_usage']
            
            improvement = hybrid_hit - lru_hit
            
            print(f"{trace_name:<25} {ml_usage:>9.1f}% {improvement:>11.2f}%")
            
            threshold_ml_usage.append(ml_usage)
            threshold_improvements.append(improvement)
            all_ml_usage.append(ml_usage)
            all_improvements.append(improvement)
            all_thresholds.append(threshold)
        
        avg_ml_usage = np.mean(threshold_ml_usage)
        avg_improvement = np.mean(threshold_improvements)
        print(f"{'AVERAGE':<25} {avg_ml_usage:>9.1f}% {avg_improvement:>11.2f}%")
    
    # Calculate correlation
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Correlation between ML usage and improvement (across all thresholds and traces)
    if len(all_ml_usage) > 2:
        corr_coef, p_value = pearsonr(all_ml_usage, all_improvements)
        print(f"\n📈 Correlation between ML Usage % and Improvement:")
        print(f"   Pearson correlation coefficient: {corr_coef:.4f}")
        print(f"   P-value: {p_value:.4e}")
        
        if abs(corr_coef) > 0.7:
            strength = "strong"
        elif abs(corr_coef) > 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if corr_coef > 0 else "negative"
        
        print(f"   Interpretation: {strength} {direction} correlation")
        
        if p_value < 0.05:
            print(f"   Statistical significance: YES (p < 0.05)")
        else:
            print(f"   Statistical significance: NO (p >= 0.05)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: ML Usage per Trace (at default confidence)
    ax1 = axes[0]
    trace_names = [traces[i][0].replace('Scenario ', 'S') for i in range(len(ml_usage_data))]
    bars = ax1.bar(range(len(ml_usage_data)), ml_usage_data, color='steelblue', edgecolor='black')
    
    # Color bars based on improvement
    for i, (bar, improvement) in enumerate(zip(bars, improvement_data)):
        if improvement > 5:
            bar.set_color('green')
        elif improvement > 0:
            bar.set_color('steelblue')
        else:
            bar.set_color('coral')
    
    ax1.set_xlabel('Trace Scenario', fontsize=12)
    ax1.set_ylabel('ML Usage (%)', fontsize=12)
    ax1.set_title('ML Decision Percentage per Trace\n(Confidence = 0.88)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(trace_names)))
    ax1.set_xticklabels(trace_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% ML Usage')
    ax1.legend()
    
    # Plot 2: ML Usage vs Improvement Scatter
    ax2 = axes[1]
    colors = ['red' if t == 0.60 else 'orange' if t == 0.88 else 'gray' 
              for t in all_thresholds]
    scatter = ax2.scatter(all_ml_usage, all_improvements, c=colors, alpha=0.6, s=50)
    
    # Add trend line
    if len(all_ml_usage) > 2:
        z = np.polyfit(all_ml_usage, all_improvements, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_ml_usage), max(all_ml_usage), 100)
        ax2.plot(x_line, p(x_line), "b--", alpha=0.5, linewidth=2, 
                label=f'Trend: y={z[0]:.3f}x{z[1]:+.2f}')
    
    ax2.set_xlabel('ML Usage (%)', fontsize=12)
    ax2.set_ylabel('Hit Rate Improvement (%)', fontsize=12)
    ax2.set_title(f'Correlation: ML Usage vs Improvement\n(r={corr_coef:.3f}, p={p_value:.2e})', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.legend()
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Conf=0.60'),
                      Patch(facecolor='orange', label='Conf=0.88'),
                      Patch(facecolor='gray', label='Other')]
    ax2.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    output_path = os.path.join(script_dir, 'ml_usage_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Visualization saved: ml_usage_analysis.png")
    
    # Summary insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print(f"\n1️⃣  ML Usage Distribution:")
    print(f"    Average ML usage: {np.mean(ml_usage_data):.1f}%")
    print(f"    Max ML usage: {np.max(ml_usage_data):.1f}% ({traces[np.argmax(ml_usage_data)][0]})")
    print(f"    Min ML usage: {np.min(ml_usage_data):.1f}% ({traces[np.argmin(ml_usage_data)][0]})")
    
    print(f"\n2️⃣  Confidence-Improvement Relationship:")
    if corr_coef > 0:
        print(f"    ✅ Higher ML usage tends to correlate with better improvement")
        print(f"    Implication: ML model is making valuable predictions")
    else:
        print(f"    ⚠️  Higher ML usage does not guarantee better performance")
        print(f"    Implication: Model may need better confidence calibration")
    
    print(f"\n3️⃣  Optimal Configuration:")
    # Find threshold with best average improvement
    threshold_avg_improvements = []
    for threshold in confidence_thresholds:
        threshold_improvements = []
        for trace_name, trace_file, max_accesses in traces:
            trace_path = os.path.join(script_dir, "traces", trace_file)
            if not os.path.exists(trace_path):
                continue
            addresses = read_champsim_trace(trace_path, max_accesses=max_accesses)
            lru_stats = test_cache(CacheLRUFixed, addresses)
            hybrid_stats = test_cache(CacheMLHybrid, addresses, confidence_threshold=threshold)
            threshold_improvements.append(hybrid_stats['hit_rate'] - lru_stats['hit_rate'])
        threshold_avg_improvements.append(np.mean(threshold_improvements))
    
    best_threshold_idx = np.argmax(threshold_avg_improvements)
    best_threshold = confidence_thresholds[best_threshold_idx]
    best_improvement = threshold_avg_improvements[best_threshold_idx]
    
    print(f"    Best confidence threshold: {best_threshold:.2f}")
    print(f"    Average improvement at best threshold: {best_improvement:+.2f}%")
    print(f"    Current threshold (0.88): {threshold_avg_improvements[-1]:+.2f}%")
    
    if best_threshold < 0.88:
        gain = best_improvement - threshold_avg_improvements[-1]
        print(f"    💡 Potential gain by lowering threshold: {gain:+.2f}%")

if __name__ == "__main__":
    main()
