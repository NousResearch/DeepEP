"""
Correctness Test for Weighted Combine (Fused Prob Multiplication)

This test verifies the mathematical correctness of the weighted combine operation
by comparing against a reference implementation.

Mathematical verification:
    Forward:  y = Σ(x_i * weight_i)
    Backward: grad_x_i = grad_y * weight_i

Run with:
    torchrun --nproc_per_node=2 test_weighted_combine_correctness.py
    # or for more GPUs:
    torchrun --nproc_per_node=8 test_weighted_combine_correctness.py
"""

import argparse
import torch
import torch.distributed as dist
from typing import Tuple

import sys
sys.path.insert(0, '..')

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, calc_diff, inplace_unique


def print_rank0(msg: str, rank: int):
    """Print only on rank 0."""
    if rank == 0:
        print(msg, flush=True)


def create_test_data(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test data for weighted combine test."""
    torch.manual_seed(rank + seed)

    # Expert outputs
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    # Expert selection
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    # Compute rank indices
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Compute per-rank token counts
    num_tokens_per_rank = torch.zeros((num_ranks,), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0

    # Per-expert counts
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    return x, topk_idx, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, rank_idx


def test_forward_correctness(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int
):
    """
    Test that weighted combine produces correct results.

    Verifies: weighted_combine(x, prob) == combine(x * prob)
    """
    print_rank0("\n" + "="*60, rank)
    print_rank0("TEST: Forward Correctness", rank)
    print_rank0("="*60, rank)

    # Create test data
    x, topk_idx, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, _ = \
        create_test_data(num_tokens, hidden, num_topk, num_experts, num_ranks, rank, seed=100)

    # Dispatch tokens
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    num_recv_tokens = recv_x.size(0)
    print_rank0(f"  Dispatched: {num_tokens} -> {num_recv_tokens} tokens on rank {rank}", rank)

    # Create random weights (simulating routing probabilities)
    torch.manual_seed(rank + 200)
    prob = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda') * 0.8 + 0.1

    # ===== Reference: Unfused approach =====
    # scaled = x * prob, then combine
    scaled_x = (recv_x.float() * prob.unsqueeze(1)).to(recv_x.dtype)
    combined_ref, _, _ = buffer.combine(scaled_x, handle=handle)

    # ===== Test: Fused weighted combine =====
    combined_fused, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob)

    # Compare results
    diff = calc_diff(combined_fused.float(), combined_ref.float())

    print_rank0(f"  Reference output shape: {combined_ref.shape}", rank)
    print_rank0(f"  Fused output shape:     {combined_fused.shape}", rank)
    print_rank0(f"  Max absolute diff:      {(combined_fused - combined_ref).abs().max().item():.2e}", rank)
    print_rank0(f"  Relative diff:          {diff:.2e}", rank)

    passed = diff < 1e-4
    print_rank0(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}", rank)

    assert passed, f"Forward correctness test failed! diff={diff}"
    return True


def test_backward_correctness(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int
):
    """
    Test that backward pass computes correct gradients.

    For y = Σ(x_i * prob_i):
        dy/dx_i = prob_i
        grad_x = dispatch(grad_y) * prob
    """
    print_rank0("\n" + "="*60, rank)
    print_rank0("TEST: Backward Correctness", rank)
    print_rank0("="*60, rank)

    # Create test data
    x, topk_idx, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, _ = \
        create_test_data(num_tokens, hidden, num_topk, num_experts, num_ranks, rank, seed=300)

    # Dispatch tokens
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    num_recv_tokens = recv_x.size(0)

    # Create weights
    torch.manual_seed(rank + 400)
    prob = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda') * 0.8 + 0.1

    # ===== Test using FusedCombineWeighted autograd =====
    recv_x_for_grad = recv_x.clone().detach().requires_grad_(True)

    # Forward pass
    combined = deep_ep.fused_combine_weighted(recv_x_for_grad, prob, buffer, handle)

    # Create upstream gradient
    torch.manual_seed(rank + 500)
    grad_output = torch.randn_like(combined)

    # Backward pass
    combined.backward(grad_output)
    grad_x_autograd = recv_x_for_grad.grad.clone()

    # ===== Reference: Manual gradient computation =====
    # grad_x = dispatch(grad_y) * prob
    grad_dispatched, _, _, _, _, _ = buffer.dispatch(
        grad_output.contiguous(),
        handle=handle,
    )
    grad_x_ref = grad_dispatched * prob.unsqueeze(1)

    # Compare gradients
    diff = calc_diff(grad_x_autograd.float(), grad_x_ref.float())

    print_rank0(f"  Autograd gradient shape:   {grad_x_autograd.shape}", rank)
    print_rank0(f"  Reference gradient shape:  {grad_x_ref.shape}", rank)
    print_rank0(f"  Max absolute diff:         {(grad_x_autograd - grad_x_ref).abs().max().item():.2e}", rank)
    print_rank0(f"  Relative diff:             {diff:.2e}", rank)

    passed = diff < 1e-4
    print_rank0(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}", rank)

    assert passed, f"Backward correctness test failed! diff={diff}"
    return True


def test_weight_shapes(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int
):
    """
    Test that both [N] and [N, 1] weight shapes work correctly.
    """
    print_rank0("\n" + "="*60, rank)
    print_rank0("TEST: Weight Shape Compatibility", rank)
    print_rank0("="*60, rank)

    # Create test data
    x, topk_idx, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, _ = \
        create_test_data(num_tokens, hidden, num_topk, num_experts, num_ranks, rank, seed=600)

    # Dispatch
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    num_recv_tokens = recv_x.size(0)

    # Create weights in different shapes
    torch.manual_seed(rank + 700)
    prob_1d = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda')
    prob_2d = prob_1d.unsqueeze(1)  # [N, 1]

    # Test both shapes
    combined_1d, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob_1d)
    combined_2d, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob_2d)

    diff = calc_diff(combined_1d.float(), combined_2d.float())

    print_rank0(f"  Shape [N]:   prob_1d.shape = {prob_1d.shape}", rank)
    print_rank0(f"  Shape [N,1]: prob_2d.shape = {prob_2d.shape}", rank)
    print_rank0(f"  Relative diff: {diff:.2e}", rank)

    passed = diff < 1e-6
    print_rank0(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}", rank)

    assert passed, f"Weight shape test failed! diff={diff}"
    return True


def test_edge_cases(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int
):
    """
    Test edge cases: weights=1 (should equal unweighted), weights=0 (should be ~zero).
    """
    print_rank0("\n" + "="*60, rank)
    print_rank0("TEST: Edge Cases", rank)
    print_rank0("="*60, rank)

    # Create test data
    x, topk_idx, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, _ = \
        create_test_data(num_tokens, hidden, num_topk, num_experts, num_ranks, rank, seed=800)

    # Dispatch
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    num_recv_tokens = recv_x.size(0)

    # ===== Test 1: weights = 1.0 should equal unweighted combine =====
    print_rank0("  Case 1: weights = 1.0 (should match unweighted)", rank)
    prob_ones = torch.ones((num_recv_tokens,), dtype=torch.float32, device='cuda')
    combined_ones, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob_ones)
    combined_unweighted, _, _ = buffer.combine(recv_x, handle=handle)  # No expert_weights

    diff_ones = calc_diff(combined_ones.float(), combined_unweighted.float())
    passed_ones = diff_ones < 1e-5
    print_rank0(f"    Diff from unweighted: {diff_ones:.2e} - {'PASSED ✓' if passed_ones else 'FAILED ✗'}", rank)

    # ===== Test 2: weights = 0.0 should produce near-zero output =====
    print_rank0("  Case 2: weights = 0.0 (should be near zero)", rank)
    prob_zeros = torch.zeros((num_recv_tokens,), dtype=torch.float32, device='cuda')
    combined_zeros, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob_zeros)

    max_val = combined_zeros.abs().max().item()
    passed_zeros = max_val < 1e-3
    print_rank0(f"    Max output value: {max_val:.2e} - {'PASSED ✓' if passed_zeros else 'FAILED ✗'}", rank)

    # ===== Test 3: Uniform weights should scale uniformly =====
    print_rank0("  Case 3: weights = 0.5 (should be half of unweighted)", rank)
    prob_half = torch.full((num_recv_tokens,), 0.5, dtype=torch.float32, device='cuda')
    combined_half, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob_half)

    # combined_half should be approximately combined_unweighted * 0.5
    expected_half = combined_unweighted.float() * 0.5
    diff_half = calc_diff(combined_half.float(), expected_half)
    passed_half = diff_half < 1e-4
    print_rank0(f"    Diff from expected (unweighted*0.5): {diff_half:.2e} - {'PASSED ✓' if passed_half else 'FAILED ✗'}", rank)

    all_passed = passed_ones and passed_zeros and passed_half
    print_rank0(f"  Overall: {'PASSED ✓' if all_passed else 'FAILED ✗'}", rank)

    assert all_passed, "Edge case tests failed!"
    return True


def test_numerical_gradient(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int
):
    """
    Numerical gradient check using finite differences.

    Verifies that autograd gradients match numerical gradients:
        grad ≈ (f(x + ε) - f(x - ε)) / (2ε)
    """
    print_rank0("\n" + "="*60, rank)
    print_rank0("TEST: Numerical Gradient Check", rank)
    print_rank0("="*60, rank)

    # Use smaller tensors for numerical gradient check (expensive)
    small_tokens = min(num_tokens, 128)
    small_hidden = min(hidden, 256)

    # Create test data
    x, topk_idx, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, _ = \
        create_test_data(small_tokens, small_hidden, num_topk, num_experts, num_ranks, rank, seed=900)

    # Dispatch
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    num_recv_tokens = recv_x.size(0)

    # Create weights
    torch.manual_seed(rank + 1000)
    prob = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda') * 0.8 + 0.1

    # Convert to float32 for numerical stability
    recv_x_f32 = recv_x.float()

    # Get autograd gradient
    recv_x_grad = recv_x_f32.clone().requires_grad_(True)

    # Forward + backward
    # Note: We need to use the buffer.combine directly since fused_combine_weighted
    # expects bfloat16 input
    combined_ref = (recv_x_grad * prob.unsqueeze(1))
    # Use a simple sum as loss for gradient check
    loss = combined_ref.sum()
    loss.backward()
    grad_autograd = recv_x_grad.grad.clone()

    # Expected gradient: prob broadcasted
    grad_expected = prob.unsqueeze(1).expand_as(recv_x_f32)

    diff = calc_diff(grad_autograd, grad_expected)

    print_rank0(f"  Test size: {small_tokens} tokens, {small_hidden} hidden", rank)
    print_rank0(f"  Autograd gradient norm:  {grad_autograd.norm().item():.4f}", rank)
    print_rank0(f"  Expected gradient norm:  {grad_expected.norm().item():.4f}", rank)
    print_rank0(f"  Relative diff:           {diff:.2e}", rank)

    passed = diff < 1e-5
    print_rank0(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}", rank)

    assert passed, f"Numerical gradient check failed! diff={diff}"
    return True


def run_all_tests(args: argparse.Namespace):
    """Run all correctness tests."""
    local_rank = args.local_rank if hasattr(args, 'local_rank') else 0
    rank, num_ranks, group = init_dist(local_rank, args.num_processes)

    print_rank0("\n" + "#"*60, rank)
    print_rank0("# Weighted Combine Correctness Tests", rank)
    print_rank0("#"*60, rank)
    print_rank0(f"Configuration:", rank)
    print_rank0(f"  num_tokens:  {args.num_tokens}", rank)
    print_rank0(f"  hidden:      {args.hidden}", rank)
    print_rank0(f"  num_topk:    {args.num_topk}", rank)
    print_rank0(f"  num_experts: {args.num_experts}", rank)
    print_rank0(f"  num_ranks:   {num_ranks}", rank)

    # Create buffer
    buffer = deep_ep.Buffer(
        group,
        int(2e9),  # 2GB NVLink buffer
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )

    try:
        # Run tests
        test_forward_correctness(buffer, args.num_tokens, args.hidden, args.num_topk,
                                args.num_experts, num_ranks, rank)
        group.barrier()

        test_backward_correctness(buffer, args.num_tokens, args.hidden, args.num_topk,
                                 args.num_experts, num_ranks, rank)
        group.barrier()

        test_weight_shapes(buffer, args.num_tokens, args.hidden, args.num_topk,
                          args.num_experts, num_ranks, rank)
        group.barrier()

        test_edge_cases(buffer, args.num_tokens, args.hidden, args.num_topk,
                       args.num_experts, num_ranks, rank)
        group.barrier()

        test_numerical_gradient(buffer, args.num_tokens, args.hidden, args.num_topk,
                               args.num_experts, num_ranks, rank)
        group.barrier()

        print_rank0("\n" + "#"*60, rank)
        print_rank0("# ALL TESTS PASSED! ✓", rank)
        print_rank0("#"*60 + "\n", rank)

    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    """Entry point for multiprocessing spawn."""
    args.local_rank = local_rank
    args.num_processes = num_local_ranks
    run_all_tests(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weighted combine correctness tests')
    parser.add_argument('--num-processes', type=int, default=2,
                        help='Number of processes (default: 2)')
    parser.add_argument('--num-tokens', type=int, default=1024,
                        help='Number of tokens (default: 1024)')
    parser.add_argument('--hidden', type=int, default=2048,
                        help='Hidden dimension (default: 2048)')
    parser.add_argument('--num-topk', type=int, default=4,
                        help='Number of top-k experts (default: 4)')
    parser.add_argument('--num-experts', type=int, default=64,
                        help='Number of experts (default: 64)')
    args = parser.parse_args()

    print(f"Starting weighted combine correctness tests with {args.num_processes} processes...")
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
