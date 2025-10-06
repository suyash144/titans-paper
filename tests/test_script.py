"""
Disclaimer: this is an AI-generated script I used for debugging.
"""

import torch
import torch.nn.functional as F
from architecture import MAC


model = MAC(
    in_dim=10,      # Simple one-hot encoding
    hid_dim=32,     # Small hidden size
    out_dim=10,     # Predict next token
    ctx_window=4,   # Small context
    persist_mem=2,  # Minimal persistent memory
    num_layers=1,   # Single layer to start
    mem_layers=1,   # Simplest memory network
    alpha=0.99,
    eta=0.8,
    theta=0.1
)

print("\n" + "="*60)
print("STEP 1: Memorization Test - Single Sequence")
print("="*60)

try:
    # Create a simple pattern: [0,1,2,3] repeats
    pattern = [0, 1, 2, 3, 0, 1, 2, 3]
    
    # One-hot encode
    def to_onehot(seq, num_classes=10):
        onehot = torch.zeros(len(seq), num_classes)
        for i, val in enumerate(seq):
            onehot[i, val] = 1.0
        return onehot
    
    input_seq = to_onehot(pattern).unsqueeze(0)  # [1, 8, 10]
    target_seq = input_seq.clone()  # Autoencoding task
    
    print(f"Pattern to memorize: {pattern}")
    print(f"Input shape: {input_seq.shape}")
    
    optimizer = torch.optim.Adam(model.trainable_params, lr=0.001)
    
    print("\nTraining for 50 steps...")
    losses = []
    
    for step in range(50):
        optimizer.zero_grad()
        
        output = model(input_seq)
        loss = F.mse_loss(output, target_seq)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 10 == 0:
            # Get predictions
            preds = output[0].argmax(dim=-1).tolist()
            accuracy = sum(p == t for p, t in zip(preds, pattern)) / len(pattern)
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2%} | Preds: {preds}")
    
    # Final check
    with torch.no_grad():
        final_output = model(input_seq)
        final_preds = final_output[0].argmax(dim=-1).tolist()
        final_accuracy = sum(p == t for p, t in zip(final_preds, pattern)) / len(pattern)
    
    print(f"\n✓ Training completed")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    print(f"  Final accuracy: {final_accuracy:.2%}")
    print(f"  Target:  {pattern}")
    print(f"  Predicted: {final_preds}")
    
    if final_accuracy > 0.5:
        print("✓ Model shows learning capability")
    else:
        print("⚠ Warning: Model not learning well (accuracy < 50%)")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("STEP 2: Memorization Test - Multiple Sequences")
print("="*60)

try:
    # Create 4 different patterns
    patterns = [
        [0, 1, 0, 1, 0, 1, 0, 1],
        [2, 3, 2, 3, 2, 3, 2, 3],
        [0, 2, 0, 2, 0, 2, 0, 2],
        [1, 3, 1, 3, 1, 3, 1, 3],
    ]
    
    # One-hot encode all patterns
    batch_input = torch.stack([to_onehot(p) for p in patterns])  # [4, 8, 10]
    batch_target = batch_input.clone()
    
    print(f"Patterns to memorize:")
    for i, p in enumerate(patterns):
        print(f"  {i}: {p}")
    
    # Reset model
    model = MAC(
        in_dim=10, hid_dim=32, out_dim=10,
        ctx_window=4, persist_mem=2,
        num_layers=1, mem_layers=1,
        alpha=0.99, eta=0.8, theta=0.1
    )
    
    optimizer = torch.optim.Adam(model.trainable_params, lr=0.001)
    
    print("\nTraining for 50 steps...")
    
    for step in range(50):
        optimizer.zero_grad()
        
        output = model(batch_input)
        loss = F.mse_loss(output, batch_target)
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            # Check accuracy for each sequence
            accuracies = []
            for i in range(len(patterns)):
                preds = output[i].argmax(dim=-1).tolist()
                acc = sum(p == t for p, t in zip(preds, patterns[i])) / len(patterns[i])
                accuracies.append(acc)
            
            avg_acc = sum(accuracies) / len(accuracies)
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | Avg Accuracy: {avg_acc:.2%}")
    
    # Final evaluation
    with torch.no_grad():
        final_output = model(batch_input)
        
        print(f"\n✓ Multi-sequence training completed")
        print(f"\nFinal Results:")
        
        all_correct = True
        for i in range(len(patterns)):
            preds = final_output[i].argmax(dim=-1).tolist()
            acc = sum(p == t for p, t in zip(preds, patterns[i])) / len(patterns[i])
            match = "✓" if acc > 0.95 else "✗"
            print(f"  Seq {i} {match} Acc: {acc:.2%} | Target: {patterns[i]} | Pred: {preds}")
            if acc <= 0.95:
                all_correct = False
        
        if all_correct:
            print("\n✓✓✓ SUCCESS: Model can memorize multiple sequences!")
        else:
            print("\n⚠ Model struggling with some sequences")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("="*60)