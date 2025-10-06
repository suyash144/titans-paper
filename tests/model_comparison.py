import sys
import os
sys.path.insert(0, os.getcwd())
from capacity import train_and_evaluate, generate_unique_patterns
from architecture import MAC
from model import StandardTransformerModel, TransformerWithStaticMemory
import time
import csv
import numpy as np


def compare_models_on_memorisation(num_sequences=32, seq_length=8, vocab_size=10, 
                                   num_steps=200, lr=0.001, num_trials=3):
    """
    Compare different models on sequence memorisation task
    
    Args:
        num_sequences: Number of unique sequences to memorize
        seq_length: Length of each sequence
        vocab_size: Size of vocabulary
        num_steps: Training steps
        lr: Learning rate
        num_trials: Number of trials to average over
    """
    
    models_config = {
        'MAC': lambda: MAC(
            in_dim=vocab_size,
            hid_dim=32,
            out_dim=vocab_size,
            ctx_window=4,
            persist_mem=4,
            num_layers=1,
            mem_layers=2,
            alpha=0.99,
            eta=0.8,
            theta=0.1
        ),
        'StandardTransformer': lambda: StandardTransformerModel(
            input_features=vocab_size,
            embed_features=32,
            context_len=4,
            num_blocks=2,
            out_dim=vocab_size
        ),
        'TransformerWithStaticMemory': lambda: TransformerWithStaticMemory(
            in_features=vocab_size,
            hidden_features=32,
            seq_length=4,
            static_mem_size=8,
            n_blocks=2,
            out_dim=vocab_size
        )
    }
    
    results = {}
    
    for model_name, model_factory in models_config.items():
        print(f"Testing: {model_name}")
        
        trial_results = []
        
        for trial in range(num_trials):
            print(f"\nTrial {trial + 1}/{num_trials}")
            
            # Generate patterns
            patterns = generate_unique_patterns(num_sequences, seq_length, vocab_size)
            
            # Create model
            model = model_factory()
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            # Train and evaluate
            start_time = time.time()
            result = train_and_evaluate(
                model, patterns,
                num_steps=num_steps,
                lr=lr,
                verbose=False
            )
            elapsed = time.time() - start_time
            
            result['training_time'] = elapsed
            result['num_params'] = num_params
            trial_results.append(result)
            
            print(f"  Avg Accuracy: {result['avg_accuracy']:.2%}")
            print(f"  Min Accuracy: {result['min_accuracy']:.2%}")
            print(f"  Well Memorized: {result['well_memorized']}/{num_sequences}")
            print(f"  Final Loss: {result['final_loss']:.4f}")
            print(f"  Training Time: {elapsed:.2f}s")
        
        # Aggregate results
        avg_accuracy = np.mean([r['avg_accuracy'] for r in trial_results])
        std_accuracy = np.std([r['avg_accuracy'] for r in trial_results])
        avg_well_memorized = np.mean([r['well_memorized'] for r in trial_results])
        avg_final_loss = np.mean([r['final_loss'] for r in trial_results])
        avg_time = np.mean([r['training_time'] for r in trial_results])
        num_params = trial_results[0]['num_params']
        
        results[model_name] = {
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'well_memorized': avg_well_memorized,
            'final_loss': avg_final_loss,
            'training_time': avg_time,
            'num_params': num_params,
            'trial_results': trial_results
        }
        
    return results


if __name__ == "__main__":
    
    print("TASK: 128 sequences, length 16")
    results_hard = compare_models_on_memorisation(
        num_sequences=128,
        seq_length=16,
        vocab_size=10,
        num_steps=400,
        lr=0.001,
        num_trials=2
    )

    csv_filename = f"model_comparison_results.csv"
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Num_Params', 'Avg_Accuracy', 'Std_Accuracy', 
                        'Well_Memorized', 'Total_Sequences', 'Final_Loss', 'Training_Time_s'])
        
        num_seq = 128
        for model_name, result in results_hard.items():
            writer.writerow([
                model_name,
                result['num_params'],
                f"{result['avg_accuracy']:.4f}",
                f"{result['std_accuracy']:.4f}",
                f"{result['well_memorized']:.1f}",
                num_seq,
                f"{result['final_loss']:.4f}",
                f"{result['training_time']:.2f}"
            ])
    
