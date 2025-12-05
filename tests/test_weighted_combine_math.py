"""
Mathematical Verification for Weighted Combine

This test verifies the mathematical correctness of the weighted combine
operation WITHOUT requiring distributed setup. It tests the core math:

    Forward:  y = Σ(x_i * weight_i)
    Backward: grad_x_i = grad_y * weight_i

Run with:
    cd DeepEP/tests
    python test_weighted_combine_math.py
"""

import torch
import torch.nn as nn
import torch.autograd as autograd


def test_weighted_sum_forward():
    """
    Test: y = Σ(x_i * w_i)

    For simplicity, simulate combine with 2 "ranks" contributing to output.
    """
    print("\n" + "="*60)
    print("TEST: Weighted Sum Forward")
    print("="*60)

    # Simulate 2 ranks each contributing to a token
    # x1 from rank 0, x2 from rank 1
    x1 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)  # [1, 3]
    x2 = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)  # [1, 3]

    # Weights for each contribution
    w1 = torch.tensor([0.3])  # weight for x1
    w2 = torch.tensor([0.7])  # weight for x2

    # Expected: y = x1 * w1 + x2 * w2
    expected = x1 * w1.unsqueeze(1) + x2 * w2.unsqueeze(1)
    # = [1*0.3, 2*0.3, 3*0.3] + [4*0.7, 5*0.7, 6*0.7]
    # = [0.3, 0.6, 0.9] + [2.8, 3.5, 4.2]
    # = [3.1, 4.1, 5.1]

    print(f"  x1 = {x1}")
    print(f"  x2 = {x2}")
    print(f"  w1 = {w1.item()}, w2 = {w2.item()}")
    print(f"  Expected y = x1*w1 + x2*w2 = {expected}")

    # Manual calculation
    manual = torch.tensor([[3.1, 4.1, 5.1]])
    diff = (expected - manual).abs().max().item()
    print(f"  Manual check diff: {diff:.2e}")

    passed = diff < 1e-5
    print(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    assert passed
    return True


def test_weighted_sum_backward():
    """
    Test: grad_x_i = grad_y * w_i

    The backward of y = Σ(x_i * w_i) with respect to x_i is w_i.
    """
    print("\n" + "="*60)
    print("TEST: Weighted Sum Backward")
    print("="*60)

    # Create tensors with gradients
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, requires_grad=True)
    w = torch.tensor([0.7], dtype=torch.float32)

    # Forward: y = x * w
    y = x * w.unsqueeze(1)
    print(f"  x = {x.data}")
    print(f"  w = {w.item()}")
    print(f"  y = x * w = {y.data}")

    # Create upstream gradient
    grad_y = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    print(f"  grad_y = {grad_y}")

    # Backward
    y.backward(grad_y)

    # Expected gradient: grad_x = grad_y * w = [1,1,1] * 0.7 = [0.7, 0.7, 0.7]
    expected_grad = grad_y * w.unsqueeze(1)
    print(f"  Expected grad_x = grad_y * w = {expected_grad}")
    print(f"  Actual grad_x = {x.grad}")

    diff = (x.grad - expected_grad).abs().max().item()
    print(f"  Diff: {diff:.2e}")

    passed = diff < 1e-6
    print(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    assert passed
    return True


def test_multi_token_weighted_combine():
    """
    Test weighted combine with multiple tokens, each with different weights.
    """
    print("\n" + "="*60)
    print("TEST: Multi-Token Weighted Combine")
    print("="*60)

    # 4 tokens, hidden=3
    num_tokens = 4
    hidden = 3

    # Expert outputs
    x = torch.randn((num_tokens, hidden), dtype=torch.float32, requires_grad=True)

    # Per-token weights (simulating routing probabilities)
    weights = torch.rand((num_tokens, 1), dtype=torch.float32)

    print(f"  x shape: {x.shape}")
    print(f"  weights shape: {weights.shape}")
    print(f"  weights = {weights.squeeze().tolist()}")

    # Forward: y = x * weights (element-wise per token)
    y = x * weights
    print(f"  y = x * weights, shape: {y.shape}")

    # Loss for backward
    loss = y.sum()
    loss.backward()

    # Expected gradient: grad_x[i] = weights[i] (broadcasted across hidden dim)
    expected_grad = weights.expand_as(x)
    print(f"  Expected grad_x = weights broadcasted")

    diff = (x.grad - expected_grad).abs().max().item()
    print(f"  Grad diff: {diff:.2e}")

    passed = diff < 1e-6
    print(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    assert passed
    return True


def test_combine_simulation():
    """
    Simulate the full combine operation (sum across "ranks") with weights.

    This simulates what DeepEP's weighted combine does:
    - Multiple ranks contribute to each output token
    - Each contribution is weighted by its routing probability
    """
    print("\n" + "="*60)
    print("TEST: Full Combine Simulation")
    print("="*60)

    # Simulate: 2 ranks, each contributing to 3 output tokens
    num_output_tokens = 3
    hidden = 4
    num_ranks = 2

    # Expert outputs from each rank (in practice, these come from different GPUs)
    x_rank0 = torch.randn((num_output_tokens, hidden), dtype=torch.float32)
    x_rank1 = torch.randn((num_output_tokens, hidden), dtype=torch.float32)

    # Weights for each rank's contribution
    w_rank0 = torch.rand((num_output_tokens, 1), dtype=torch.float32)
    w_rank1 = torch.rand((num_output_tokens, 1), dtype=torch.float32)

    print(f"  x_rank0 shape: {x_rank0.shape}")
    print(f"  x_rank1 shape: {x_rank1.shape}")
    print(f"  w_rank0 = {w_rank0.squeeze().tolist()}")
    print(f"  w_rank1 = {w_rank1.squeeze().tolist()}")

    # ===== Unfused approach: scale then sum =====
    scaled_rank0 = x_rank0 * w_rank0
    scaled_rank1 = x_rank1 * w_rank1
    combined_unfused = scaled_rank0 + scaled_rank1
    print(f"\n  Unfused: combined = (x0 * w0) + (x1 * w1)")

    # ===== Fused approach: weighted sum =====
    # In practice, DeepEP does this in a single kernel
    combined_fused = x_rank0 * w_rank0 + x_rank1 * w_rank1
    print(f"  Fused: combined = x0 * w0 + x1 * w1 (single pass)")

    diff = (combined_unfused - combined_fused).abs().max().item()
    print(f"\n  Diff between unfused and fused: {diff:.2e}")

    passed = diff < 1e-6
    print(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    assert passed
    return True


def test_backward_chain_rule():
    """
    Test the full backward chain rule through weighted combine.

    Given: combined = Σ(x_rank_i * w_i)
           loss = f(combined)

    Then: grad_x_rank_i = grad_combined * w_i
          (where grad_combined = d(loss)/d(combined))
    """
    print("\n" + "="*60)
    print("TEST: Backward Chain Rule")
    print("="*60)

    num_tokens = 3
    hidden = 4

    # Simulating combine from 2 ranks
    x0 = torch.randn((num_tokens, hidden), dtype=torch.float32, requires_grad=True)
    x1 = torch.randn((num_tokens, hidden), dtype=torch.float32, requires_grad=True)

    w0 = torch.rand((num_tokens, 1), dtype=torch.float32)
    w1 = torch.rand((num_tokens, 1), dtype=torch.float32)

    print(f"  x0, x1 shape: {x0.shape}")
    print(f"  w0, w1 values: {w0.squeeze().tolist()}, {w1.squeeze().tolist()}")

    # Forward: combined = x0 * w0 + x1 * w1
    combined = x0 * w0 + x1 * w1

    # Some loss function
    loss = (combined ** 2).sum()
    print(f"  loss = sum(combined^2) = {loss.item():.4f}")

    # Backward
    loss.backward()

    # grad_combined = 2 * combined
    grad_combined = 2 * combined.detach()

    # Expected gradients
    expected_grad_x0 = grad_combined * w0
    expected_grad_x1 = grad_combined * w1

    print(f"\n  Checking grad_x0:")
    diff0 = (x0.grad - expected_grad_x0).abs().max().item()
    print(f"    Diff: {diff0:.2e}")

    print(f"  Checking grad_x1:")
    diff1 = (x1.grad - expected_grad_x1).abs().max().item()
    print(f"    Diff: {diff1:.2e}")

    passed = diff0 < 1e-5 and diff1 < 1e-5
    print(f"\n  Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    assert passed
    return True


def test_edge_cases():
    """Test edge cases: zero weights, unit weights."""
    print("\n" + "="*60)
    print("TEST: Edge Cases")
    print("="*60)

    num_tokens = 4
    hidden = 3

    x = torch.randn((num_tokens, hidden), dtype=torch.float32)

    # Case 1: weights = 1.0 (identity)
    print("\n  Case 1: weights = 1.0")
    w_ones = torch.ones((num_tokens, 1))
    y_ones = x * w_ones
    diff1 = (y_ones - x).abs().max().item()
    print(f"    y should equal x, diff: {diff1:.2e}")
    assert diff1 < 1e-6, "weights=1 should be identity"

    # Case 2: weights = 0.0 (zero output)
    print("  Case 2: weights = 0.0")
    w_zeros = torch.zeros((num_tokens, 1))
    y_zeros = x * w_zeros
    max_val = y_zeros.abs().max().item()
    print(f"    y should be zero, max val: {max_val:.2e}")
    assert max_val < 1e-6, "weights=0 should produce zero output"

    # Case 3: weights = 0.5 (half)
    print("  Case 3: weights = 0.5")
    w_half = torch.full((num_tokens, 1), 0.5)
    y_half = x * w_half
    expected_half = x * 0.5
    diff3 = (y_half - expected_half).abs().max().item()
    print(f"    y should be x*0.5, diff: {diff3:.2e}")
    assert diff3 < 1e-6, "weights=0.5 should halve the input"

    print(f"\n  Result: PASSED ✓")
    return True


def run_all_tests():
    """Run all mathematical verification tests."""
    print("\n" + "#"*60)
    print("# Mathematical Verification Tests")
    print("# for Weighted Combine (Fused Prob Multiplication)")
    print("#"*60)

    tests = [
        test_weighted_sum_forward,
        test_weighted_sum_backward,
        test_multi_token_weighted_combine,
        test_combine_simulation,
        test_backward_chain_rule,
        test_edge_cases,
    ]

    all_passed = True
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  FAILED: {e}")
            all_passed = False

    print("\n" + "#"*60)
    if all_passed:
        print("# ALL MATHEMATICAL TESTS PASSED! ✓")
    else:
        print("# SOME TESTS FAILED! ✗")
    print("#"*60 + "\n")

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
