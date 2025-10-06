import torch
import torch.nn.functional as F
import os, sys
sys.path.insert(0, os.getcwd())
from architecture import MAC
import numpy as np
import time
from tests.capacity import to_onehot, generate_unique_patterns


def pad_sequences(patterns, pad_value=0):
    """Pad sequences to same length for batching"""
    max_len = max(len(p) for p in patterns)
    padded = []
    masks = []
    
    for pattern in patterns:
        pad_len = max_len - len(pattern)
        padded_pattern = pattern + [pad_value] * pad_len
        mask = [1.0] * len(pattern) + [0.0] * pad_len  # 1 for real tokens, 0 for padding
        padded.append(padded_pattern)
        masks.append(mask)
    
    return padded, masks


def train_and_evaluate(model, patterns, num_steps=100, lr=0.001, verbose=False, use_padding=False):
    """Train model on patterns and return accuracy metrics"""
    
    if use_padding:
        # Pad sequences to same length
        padded_patterns, masks = pad_sequences(patterns)
        batch_input = torch.stack([to_onehot(p) for p in padded_patterns])
        batch_target = batch_input.clone()
        mask_tensor = torch.tensor(masks).unsqueeze(-1)  # [batch, seq, 1]
    else:
        # All sequences must be same length
        batch_input = torch.stack([to_onehot(p) for p in patterns])
        batch_target = batch_input.clone()
        mask_tensor = torch.ones(batch_input.shape[0], batch_input.shape[1], 1)
    
    optimizer = torch.optim.Adam(model.trainable_params, lr=lr)
    
    losses = []
    accuracies = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        output = model(batch_input)
        
        # Apply mask to loss calculation
        if use_padding:
            masked_output = output * mask_tensor
            masked_target = batch_target * mask_tensor
            loss = F.mse_loss(masked_output, masked_target)
        else:
            loss = F.mse_loss(output, batch_target)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Calculate accuracy (only on non-padded tokens)
        with torch.no_grad():
            correct_tokens = 0
            total_tokens = 0
            
            for i, pattern in enumerate(patterns):
                preds = output[i, :len(pattern)].argmax(dim=-1).tolist()
                correct_tokens += sum(p == t for p, t in zip(preds, pattern))
                total_tokens += len(pattern)
            
            acc = correct_tokens / total_tokens if total_tokens > 0 else 0
            accuracies.append(acc)
        
        if verbose and step % 25 == 0:
            print(f"    Step {step:3d} | Loss: {loss.item():.4f} | Accuracy: {acc:.2%}")
    
    # Final evaluation
    with torch.no_grad():
        final_output = model(batch_input)
        
        seq_accuracies = []
        seq_details = []
        
        for i, pattern in enumerate(patterns):
            preds = final_output[i, :len(pattern)].argmax(dim=-1).tolist()
            correct = sum(p == t for p, t in zip(preds, pattern))
            acc = correct / len(pattern)
            seq_accuracies.append(acc)
            seq_details.append({
                'length': len(pattern),
                'accuracy': acc,
                'correct_tokens': correct,
                'total_tokens': len(pattern),
                'pattern': pattern,
                'prediction': preds
            })
        
        avg_accuracy = sum(seq_accuracies) / len(seq_accuracies) if seq_accuracies else 0
        min_accuracy = min(seq_accuracies) if seq_accuracies else 0
        well_memorized = sum(1 for acc in seq_accuracies if acc >= 0.95)
    
    return {
        'avg_accuracy': avg_accuracy,
        'min_accuracy': min_accuracy,
        'seq_accuracies': seq_accuracies,
        'seq_details': seq_details,
        'well_memorized': well_memorized,
        'final_loss': losses[-1],
        'losses': losses,
        'accuracies': accuracies
    }


def test_uniform_length(
    length,
    num_sequences=8,
    vocab_size=10,
    model_params=None,
    num_steps=150,
    lr=0.001,
    num_trials=3,
    verbose=False
):
    """Test memorization of sequences all with the same length"""
    
    if model_params is None:
        model_params = {
            'in_dim': vocab_size,
            'hid_dim': 32,
            'out_dim': vocab_size,
            'ctx_window': 4,
            'persist_mem': 2,
            'num_layers': 1,
            'mem_layers': 1,
            'alpha': 0.99,
            'eta': 0.8,
            'theta': 0.1
        }
    
    trial_results = []
    
    for trial in range(num_trials):
        patterns = generate_unique_patterns(num_sequences, length, vocab_size)
        model = MAC(**model_params)
        
        result = train_and_evaluate(
            model, patterns,
            num_steps=num_steps,
            lr=lr,
            verbose=verbose and trial == 0,
            use_padding=False
        )
        
        result['length'] = length
        result['trial'] = trial
        trial_results.append(result)
    
    # Aggregate results
    avg_accuracy = np.mean([r['avg_accuracy'] for r in trial_results])
    std_accuracy = np.std([r['avg_accuracy'] for r in trial_results])
    avg_well_memorized = np.mean([r['well_memorized'] for r in trial_results])
    
    return {
        'length': length,
        'num_sequences': num_sequences,
        'avg_accuracy_mean': avg_accuracy,
        'avg_accuracy_std': std_accuracy,
        'well_memorized_mean': avg_well_memorized,
        'trial_results': trial_results
    }


def test_mixed_lengths(
    lengths,
    num_sequences_per_length=2,
    vocab_size=10,
    model_params=None,
    num_steps=200,
    lr=0.001,
    num_trials=3,
    verbose=False
):
    """Test memorization of sequences with different lengths in the same batch"""
    
    if model_params is None:
        model_params = {
            'in_dim': vocab_size,
            'hid_dim': 32,
            'out_dim': vocab_size,
            'ctx_window': 4,
            'persist_mem': 2,
            'num_layers': 1,
            'mem_layers': 1,
            'alpha': 0.99,
            'eta': 0.8,
            'theta': 0.1
        }
    
    trial_results = []
    
    for trial in range(num_trials):
        # Generate patterns of different lengths
        all_patterns = []
        for length in lengths:
            patterns = generate_unique_patterns(num_sequences_per_length, length, vocab_size)
            all_patterns.extend(patterns)
        
        # Shuffle to mix lengths
        np.random.shuffle(all_patterns)
        
        model = MAC(**model_params)
        
        result = train_and_evaluate(
            model, all_patterns,
            num_steps=num_steps,
            lr=lr,
            verbose=verbose and trial == 0,
            use_padding=True
        )
        
        # Group results by length
        length_stats = {}
        for detail in result['seq_details']:
            length = detail['length']
            if length not in length_stats:
                length_stats[length] = []
            length_stats[length].append(detail['accuracy'])
        
        result['length_stats'] = {
            length: {
                'mean': np.mean(accs),
                'count': len(accs)
            }
            for length, accs in length_stats.items()
        }
        result['trial'] = trial
        trial_results.append(result)
    
    return {
        'lengths': lengths,
        'num_sequences_per_length': num_sequences_per_length,
        'trial_results': trial_results
    }


def run_length_test(
    test_lengths,
    num_sequences=8,
    vocab_size=10,
    model_params=None,
    num_steps=150,
    lr=0.001,
    num_trials=3,
    test_mixed=True
):
    """
    Comprehensive test of sequence length effects on memorization
    
    Args:
        test_lengths: List of sequence lengths to test
        num_sequences: Number of sequences per test
        vocab_size: Size of vocabulary
        model_params: Dict of model hyperparameters
        num_steps: Training steps per test
        lr: Learning rate
        num_trials: Number of trials per length
        test_mixed: Whether to also test mixed-length batches
    """
    
    if model_params is None:
        model_params = {
            'in_dim': vocab_size,
            'hid_dim': 32,
            'out_dim': vocab_size,
            'ctx_window': 4,
            'persist_mem': 2,
            'num_layers': 1,
            'mem_layers': 1,
            'alpha': 0.99,
            'eta': 0.8,
            'theta': 0.1
        }

    for key, val in model_params.items():
        print(f"  {key}: {val}")
    
    # Test 1: Uniform length sequences
    print("PART 1: UNIFORM LENGTH SEQUENCES")
    
    uniform_results = []
    
    for length in test_lengths:
        print(f"Testing Length: {length} tokens")
        
        start_time = time.time()
        result = test_uniform_length(
            length=length,
            num_sequences=num_sequences,
            vocab_size=vocab_size,
            model_params=model_params,
            num_steps=num_steps,
            lr=lr,
            num_trials=num_trials,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        result['total_time'] = elapsed
        uniform_results.append(result)
        
        print(f"\n  Results across {num_trials} trials:")
        print(f"    Mean Accuracy: {result['avg_accuracy_mean']:.2%} ± {result['avg_accuracy_std']:.2%}")
        print(f"    Mean Well Memorized: {result['well_memorized_mean']:.1f}/{num_sequences}")
        print(f"    Total Time: {elapsed:.2f}s")
        print(f"    Status: {'GOOD' if result['avg_accuracy_mean'] >= 0.90 else 'DEGRADED' if result['avg_accuracy_mean'] >= 0.70 else 'POOR'}")
    
    # Summary for uniform lengths test
    print("UNIFORM LENGTH RESULTS SUMMARY")
    print(f"\n{'Length':<10} {'Avg Accuracy':<20} {'Well Memorized':<20} {'Status':<10}")
    
    for r in uniform_results:
        status = "GOOD" if r['avg_accuracy_mean'] >= 0.90 else "WARN" if r['avg_accuracy_mean'] >= 0.70 else "POOR"
        print(f"{r['length']:<10} {r['avg_accuracy_mean']:>6.2%} ± {r['avg_accuracy_std']:<4.2%}      "
              f"{r['well_memorized_mean']:>5.1f}/{num_sequences:<13} {status:<10}")
    
    # Test 2: Mixed length sequences
    mixed_results = None
    if test_mixed and len(test_lengths) > 1:
        print("PART 2: MIXED LENGTH SEQUENCES")
        
        start_time = time.time()
        mixed_results = test_mixed_lengths(
            lengths=test_lengths,
            num_sequences_per_length=max(2, num_sequences // len(test_lengths)),
            vocab_size=vocab_size,
            model_params=model_params,
            num_steps=num_steps + 50,  # Extra steps for harder task (maybe not needed)
            lr=lr,
            num_trials=num_trials,
            verbose=True
        )
        elapsed = time.time() - start_time
        
        # Aggregate mixed results
        print(f"\n  Results across {num_trials} trials:")
        print(f"\n  {'Length':<10} {'Mean Accuracy':<15} {'Count':<10}")
        print("  " + "-" * 40)
        
        # Average across trials
        all_length_stats = {}
        for trial_result in mixed_results['trial_results']:
            for length, stats in trial_result['length_stats'].items():
                if length not in all_length_stats:
                    all_length_stats[length] = []
                all_length_stats[length].append(stats['mean'])
        
        for length in sorted(all_length_stats.keys()):
            accs = all_length_stats[length]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"  {length:<10} {mean_acc:>6.2%} ± {std_acc:<4.2%}  {mixed_results['num_sequences_per_length']:<10}")
        
        # Overall mixed performance
        overall_accs = [r['avg_accuracy'] for r in mixed_results['trial_results']]
        print(f"\n  Overall Mean Accuracy: {np.mean(overall_accs):.2%} ± {np.std(overall_accs):.2%}")
        print(f"  Total Time: {elapsed:.2f}s")
    
    # Final summary
    print("FINAL SUMMARY")
    
    best_length = max(uniform_results, key=lambda r: r['avg_accuracy_mean'])
    worst_length = min(uniform_results, key=lambda r: r['avg_accuracy_mean'])
    
    print(f"\nUniform Length Performance:")
    print(f"  Best Length: {best_length['length']} tokens ({best_length['avg_accuracy_mean']:.2%} accuracy)")
    print(f"  Worst Length: {worst_length['length']} tokens ({worst_length['avg_accuracy_mean']:.2%} accuracy)")
    
    # Find where performance drops
    good_lengths = [r['length'] for r in uniform_results if r['avg_accuracy_mean'] >= 0.95]
    if good_lengths:
        print(f"  Model can reliably handle sequences up to {max(good_lengths)} tokens")
    else:
        print(f"  Model struggles with all tested lengths")
    
    if mixed_results:
        overall_mixed = np.mean([r['avg_accuracy'] for r in mixed_results['trial_results']])
        print(f"\nMixed Length Performance:")
        print(f"  Overall Accuracy: {overall_mixed:.2%}")
        print(f"  Observation: {'Mixed lengths handled well' if overall_mixed >= 0.95 else 'Mixed lengths more challenging than uniform'}")
    
    return {
        'uniform_results': uniform_results,
        'mixed_results': mixed_results
    }


if __name__ == "__main__":

    test_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]
    
    results = run_length_test(
        test_lengths=test_lengths,
        num_sequences=8,
        vocab_size=10,
        num_steps=150,
        lr=0.001,
        num_trials=2,
        test_mixed=True
    )
    