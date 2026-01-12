"""
TRAIN HYBRID ML MODEL
Uses Belady's optimal algorithm to generate training labels
Trains single Decision Tree with 2 features + confidence gating
"""

import os
import pickle
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from trace_reader import read_champsim_trace
from cache_config import NUM_SETS, ASSOCIATIVITY

def create_hybrid_dataset(addresses, cache_size=512):
    """
    Create training dataset using Belady's optimal algorithm
    Returns: (features, labels) where features = [recency, frequency, age]
    """
    from belady_oracle import belady_dataset
    
    # Run Belady's oracle to get eviction decisions
    X, y = belady_dataset(addresses, max_accesses=len(addresses))
    
    # Only use first 2 features (recency, frequency) - drop age for hardware simplicity
    X_simplified = [[sample[0], sample[1]] for sample in X]
    
    return X_simplified, y

def main():
    print("="*60)
    print("HYBRID MODEL TRAINING")
    print("Single Decision Tree + 2 Features + Confidence Gating")
    print("Optimized for Hardware Implementation")
    print("="*60)
    
    # Load training data from 10 diverse scenarios
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trace_files = [
        ("scenario1_mixed.trace", 200_000),
        ("scenario2_hostile.trace", 200_000),
        ("scenario3_friendly.trace", 200_000),
        ("scenario4_scan_heavy.trace", 200_000),
        ("scenario5_phases.trace", 200_000),
        ("scenario6_alternating.trace", 200_000),
        ("scenario7_mixed_diff.trace", 200_000),
        ("scenario8_gradual.trace", 200_000),
        ("scenario9_bursts.trace", 200_000),
        ("scenario10_noisy.trace", 200_000)
    ]
    
    print("Loading 10 DIVERSE SCENARIO traces for training...")
    all_X, all_y = [], []
    
    for trace_file, max_accesses in trace_files:
        trace_path = os.path.join(script_dir, "traces", trace_file)
        if os.path.exists(trace_path):
            print(f"Processing {trace_file} ({max_accesses:,} accesses)...")
            addresses = read_champsim_trace(trace_path, max_accesses=max_accesses)
            print(f"  Loaded {len(addresses):,} addresses")
            X, y = create_hybrid_dataset(addresses)
            all_X.extend(X)
            all_y.extend(y)
            print(f"  ✅ Added {len(X):,} samples from {trace_file}")
        else:
            print(f"  ⚠️  {trace_file} not found, skipping...")
    
    if len(all_X) == 0:
        print("❌ No training data collected. Generate traces first!")
        return
    
    print(f"\n📊 Total training samples: {len(all_X):,}")
    
    # Train/test split
    X = np.array(all_X)
    y = np.array(all_y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Train single decision tree (optimized for hardware)
    print("\nTraining Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=8,              # Match old successful model (229 leaves)
        min_samples_split=20,     # Less conservative for more leaves
        min_samples_leaf=10,      # Allow detailed splits
        random_state=42,
        class_weight='balanced'   # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Training accuracy: {train_acc*100:.2f}%")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print(f"\nModel complexity:")
    print(f"  - Depth: {model.get_depth()}")
    print(f"  - Number of leaves: {model.get_n_leaves()}")
    print(f"  - Features: [Recency, Frequency]")
    
    # Detailed evaluation
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Keep', 'Evict']))
    
    # Save model
    model_path = os.path.join(script_dir, 'cache_model_hybrid.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Model saved to {model_path}")
    print("\nHardware implementation details:")
    print(f"  - Latency: ~{model.get_depth()} cycles (tree traversal)")
    print(f"  - Memory: ~{model.get_n_leaves() * 256 // 1024}KB (node storage)")
    print(f"  - Operations: Integer comparisons only")
    print(f"  - Confidence threshold: 0.88 (88%)")

if __name__ == "__main__":
    main()
