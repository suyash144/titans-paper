import torch
import torch.nn.functional as F
import os, sys
sys.path.insert(0, os.getcwd())
from architecture import MAC
import numpy as np
import time


def to_onehot(seq, num_classes=10):
    """Convert sequence to one-hot encoding"""
    onehot = torch.zeros(len(seq), num_classes)
    for i, val in enumerate(seq):
        onehot[i, val] = 1.0
    return onehot


def generate_unique_patterns(num_patterns, seq_length=8, vocab_size=10):
    """Generate unique random patterns"""
    patterns = []
    seen = set()
    
    attempts = 0
    max_attempts = num_patterns * 100                       # A large number, just to avoid infinite loops
    
    while len(patterns) < num_patterns and attempts < max_attempts:

        pattern = np.random.randint(0, vocab_size, seq_length).tolist()
        pattern_tuple = tuple(pattern)
        
        if pattern_tuple not in seen:
            patterns.append(pattern)
            seen.add(pattern_tuple)
        
        attempts += 1
    
    if len(patterns) < num_patterns:
        print(f"Warning: Could only generate {len(patterns)} unique patterns out of {num_patterns} requested")
    
    return patterns


def train_and_evaluate(model, patterns, num_steps=100, lr=0.001, verbose=False):
    """Train model on patterns and return final accuracy"""
    # One-hot encode all patterns
    batch_input = torch.stack([to_onehot(p) for p in patterns])
    batch_target = batch_input.clone()
    
    optimizer = torch.optim.Adam(model.trainable_params, lr=lr)
    
    losses = []
    accuracies = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        output = model(batch_input)
        loss = F.mse_loss(output, batch_target)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Calculate accuracy
        with torch.no_grad():
            correct_tokens = 0
            total_tokens = 0
            
            for i in range(len(patterns)):
                preds = output[i].argmax(dim=-1).tolist()
                correct_tokens += sum(p == t for p, t in zip(preds, patterns[i]))
                total_tokens += len(patterns[i])
            
            acc = correct_tokens / total_tokens
            accuracies.append(acc)
        
        if verbose and step % 20 == 0:
            print(f"    Step {step:3d} | Loss: {loss.item():.4f} | Accuracy: {acc:.2%}")
    
    # Final evaluation
    with torch.no_grad():
        final_output = model(batch_input)
        
        # Per-sequence accuracy
        seq_accuracies = []
        for i in range(len(patterns)):
            preds = final_output[i].argmax(dim=-1).tolist()
            acc = sum(p == t for p, t in zip(preds, patterns[i])) / len(patterns[i])
            seq_accuracies.append(acc)
        
        avg_accuracy = sum(seq_accuracies) / len(seq_accuracies)
        min_accuracy = min(seq_accuracies)
        
        # Count how many sequences are well-memorized (>95% accuracy)
        well_memorized = sum(1 for acc in seq_accuracies if acc >= 0.95)
    
    return {
        'avg_accuracy': avg_accuracy,
        'min_accuracy': min_accuracy,
        'seq_accuracies': seq_accuracies,
        'well_memorized': well_memorized,
        'final_loss': losses[-1],
        'initial_loss': losses[0],
        'losses': losses,
        'accuracies': accuracies
    }


def run_capacity_test(capacity_levels, seq_length=8, vocab_size=10, model_params=None, num_steps=150, lr=0.001, num_trials=3):
    """
    Test model capacity at different levels
    
    Args:
        capacity_levels: List of sequence counts to test
        seq_length: Length of each sequence
        vocab_size: Size of vocabulary
        model_params: Dict of model hyperparameters
        num_steps: Training steps per test
        lr: Learning rate
        num_trials: Number of trials per capacity level
    """
    if model_params is None:
        model_params = {
            'in_dim': vocab_size,
            'hid_dim': 16,
            'out_dim': vocab_size,
            'ctx_window': 4,
            'persist_mem': 2,
            'num_layers': 1,
            'mem_layers': 1,
            'alpha': 0.99,
            'eta': 0.8,
            'theta': 0.1
        }
        
    results = []
    
    for num_sequences in capacity_levels:

        print(f"Testing Capacity: {num_sequences} sequences")
        
        trial_results = []
        
        for trial in range(num_trials):
            print(f"\n  Trial {trial + 1}/{num_trials}")
            
            # Generate unique patterns
            patterns = generate_unique_patterns(num_sequences, seq_length, vocab_size)
            
            # Instantiate fresh model
            model = MAC(**model_params)
            
            # Train and evaluate
            start_time = time.time()
            result = train_and_evaluate(
                model, patterns, 
                num_steps=num_steps, 
                lr=lr, 
                verbose=False
            )
            elapsed_time = time.time() - start_time
            
            result['num_sequences'] = num_sequences
            result['trial'] = trial
            result['training_time'] = elapsed_time
            result['patterns'] = patterns
            
            trial_results.append(result)
            
            print(f"    Avg Accuracy: {result['avg_accuracy']:.2%}")
            print(f"    Min Accuracy: {result['min_accuracy']:.2%}")
            print(f"    Well Memorized: {result['well_memorized']}/{num_sequences}")
            print(f"    Training Time: {elapsed_time:.2f}s")
        
        # Aggregate trial results
        avg_accuracy = np.mean([r['avg_accuracy'] for r in trial_results])
        std_accuracy = np.std([r['avg_accuracy'] for r in trial_results])
        avg_well_memorized = np.mean([r['well_memorized'] for r in trial_results])
        
        success = avg_accuracy >= 0.95
        
        summary = {
            'num_sequences': num_sequences,
            'avg_accuracy_mean': avg_accuracy,
            'avg_accuracy_std': std_accuracy,
            'well_memorized_mean': avg_well_memorized,
            'success': success,
            'trial_results': trial_results
        }
        
        results.append(summary)
        
        print(f"\n  Summary (across {num_trials} trials):")
        print(f"    Mean Accuracy: {avg_accuracy:.2%} ± {std_accuracy:.2%}")
        print(f"    Mean Well Memorized: {avg_well_memorized:.1f}/{num_sequences}")
        print(f"    Status: {'SUCCESS' if success else 'FAILURE'}")
        
        # Early stopping if model is clearly failing
        if not success and num_sequences >= 1e4:
            print(f"Model failing at {num_sequences} sequences!")
    
    print("CAPACITY TEST RESULTS SUMMARY")
    
    print(f"\n{'Sequences':<12} {'Avg Accuracy':<15} {'Well Memorized':<18} {'Status':<10}")
    print("-" * 70)
    
    max_capacity = 0
    for r in results:
        status = "PASS" if r['success'] else "FAIL"
        print(f"{r['num_sequences']:<12} {r['avg_accuracy_mean']:>6.2%} ± {r['avg_accuracy_std']:<4.2%}  "
              f"{r['well_memorized_mean']:>5.1f}/{r['num_sequences']:<10} {status:<10}")
        
        if r['success']:
            max_capacity = r['num_sequences']
    
    print(f"MAX CAPACITY: {max_capacity} sequences")
    
    return results


if __name__ == "__main__":

    # capacity_levels = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    capacity_levels = [65536, 131072, 262144]
    
    results = run_capacity_test(
        capacity_levels=capacity_levels,
        seq_length=16,
        vocab_size=10,
        num_steps=150,
        lr=0.001,
        num_trials=1
    )
