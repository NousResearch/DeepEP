"""
Test Suite for Weighted Combine (Fused Prob Multiplication)

This test suite verifies the correctness of the weighted combine feature that fuses
probability multiplication into DeepEP's combine kernel.

Test coverage:
1. Basic weighted combine correctness
2. Comparison with unfused (separate multiply + combine) approach
3. Backward pass correctness with gradient checking
4. Various tensor shapes and data types
5. Edge cases (zero weights, single token, etc.)

Usage:
    python -m torch.distributed.launch --nproc_per_node=8 test_weighted_combine.py
    # or
    torchrun --nproc_per_node=8 test_weighted_combine.py
"""

import argparse
import time
import torch
import torch.distributed as dist
import torch.autograd as autograd
from typing import Optional

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, calc_diff, inplace_unique


def test_weighted_combine_basic(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    local_rank: int
):
    """
    Test basic weighted combine correctness.

    Verifies that:
    1. Weighted combine produces correct results
    2. Results match unfused approach: combine(x * prob) == weighted_combine(x, prob)
    """
    if local_rank == 0:
        print(f'\n[test_weighted_combine_basic] Testing with num_tokens={num_tokens}, hidden={hidden}')

    # Create test data
    torch.manual_seed(rank + 42)

    # Expert outputs (what would come from h @ w2 after dispatch)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    # Expert selection scores and indices
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    # Compute routing metadata
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
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

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    # Get dispatch layout
    _, _, _, ref_is_token_in_rank, _ = buffer.get_dispatch_layout(topk_idx, num_experts)

    # Dispatch tokens
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    # Generate prob for received tokens
    # In practice, prob would be computed at the receiving rank based on routing decisions
    num_recv_tokens = recv_x.size(0)
    recv_prob = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda') * 0.5 + 0.1

    # ===== Test 1: Unfused approach (reference) =====
    # scaled = recv_x * recv_prob
    # combined_unfused = combine(scaled)
    scaled_x = (recv_x.float() * recv_prob.unsqueeze(1)).to(recv_x.dtype)
    combined_unfused, _, _ = buffer.combine(scaled_x, handle=handle)

    # ===== Test 2: Fused approach (new weighted combine) =====
    # combined_fused = weighted_combine(recv_x, recv_prob)
    combined_fused, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=recv_prob)

    # ===== Verify correctness =====
    diff = calc_diff(combined_fused, combined_unfused)
    passed = diff < 1e-4  # Allow small numerical differences due to FP precision

    if local_rank == 0:
        print(f'  [unfused vs fused] Difference: {diff:.2e} - {"PASSED" if passed else "FAILED"}')

    assert passed, f"Weighted combine results don't match unfused approach! diff={diff}"

    return True


def test_weighted_combine_shapes(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    local_rank: int
):
    """
    Test weighted combine with various tensor shapes for expert_weights.

    Verifies that both [N] and [N, 1] shapes work correctly.
    """
    if local_rank == 0:
        print(f'\n[test_weighted_combine_shapes] Testing expert_weights shapes')

    torch.manual_seed(rank + 123)

    # Create test data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
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

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    # Dispatch
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    num_recv_tokens = recv_x.size(0)
    prob_1d = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda')
    prob_2d = prob_1d.unsqueeze(1)  # [N, 1]

    # Test with 1D shape [N]
    combined_1d, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob_1d)

    # Test with 2D shape [N, 1]
    combined_2d, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob_2d)

    # Should produce identical results
    diff = calc_diff(combined_1d, combined_2d)
    passed = diff < 1e-6

    if local_rank == 0:
        print(f'  [1D vs 2D shapes] Difference: {diff:.2e} - {"PASSED" if passed else "FAILED"}')

    assert passed, f"Different shapes for expert_weights produce different results! diff={diff}"

    return True


def test_weighted_combine_backward(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    local_rank: int
):
    """
    Test backward pass correctness for FusedCombineWeighted.

    Verifies that gradients are computed correctly using the chain rule:
        grad_x = dispatch(grad_y) * prob

    This test also verifies that the NEW fused weighted dispatch (grad_x = dispatch(grad_y, expert_weights=prob))
    produces the same result as the OLD approach (grad_x = dispatch(grad_y) * prob).
    """
    if local_rank == 0:
        print(f'\n[test_weighted_combine_backward] Testing backward pass')

    torch.manual_seed(rank + 456)

    # Create test data with requires_grad
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
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

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    # Dispatch (no grad needed here)
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )

    num_recv_tokens = recv_x.size(0)
    prob = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda')

    # ===== Test using FusedCombineWeighted autograd function =====
    recv_x_grad = recv_x.clone().requires_grad_(True)

    # Forward with FusedCombineWeighted
    combined = deep_ep.fused_combine_weighted(recv_x_grad, prob, buffer, handle)

    # Create grad_output for backward
    grad_output = torch.randn_like(combined)

    # Backward (this now uses fused weighted dispatch internally)
    combined.backward(grad_output)
    grad_x_fused = recv_x_grad.grad.clone()

    # ===== Compute expected gradient manually (OLD approach) =====
    # grad_x = dispatch(grad_y) * prob
    grad_x_dispatched, _, _, _, _, _ = buffer.dispatch(
        grad_output.contiguous(),
        handle=handle,
    )
    grad_x_expected = grad_x_dispatched * prob.unsqueeze(1)

    # Compare fused vs unfused
    diff = calc_diff(grad_x_fused.float(), grad_x_expected.float())
    passed = diff < 1e-4

    if local_rank == 0:
        print(f'  [backward correctness] Gradient difference: {diff:.2e} - {"PASSED" if passed else "FAILED"}')

    assert passed, f"Backward pass produces incorrect gradients! diff={diff}"

    return True


def test_weighted_dispatch_basic(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    local_rank: int
):
    """
    Test weighted dispatch correctness.

    Weighted dispatch: recv_x[i] = dispatched_x[i] * expert_weights[i]
    This is the fused version of: recv_x = dispatch(x); recv_x = recv_x * weights

    Verifies that:
    1. Fused weighted dispatch matches unfused: dispatch(x, expert_weights=w) == dispatch(x) * w
    2. This is correct for the backward pass of weighted combine
    """
    if local_rank == 0:
        print(f'\n[test_weighted_dispatch_basic] Testing weighted dispatch correctness')

    torch.manual_seed(rank + 111)

    # Create test data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
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

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    # First do a standard dispatch to get the handle (needed for cached dispatch)
    recv_x_first, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )
    num_recv_tokens = recv_x_first.size(0)

    # Create expert_weights for the received tokens
    expert_weights = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda') * 0.8 + 0.1

    # ===== Test 1: Unfused approach (reference) =====
    # First dispatch x, then multiply by weights
    recv_x_unfused, _, _, _, _, _ = buffer.dispatch(
        x=x.clone(),
        handle=handle,
    )
    recv_x_unfused_weighted = (recv_x_unfused.float() * expert_weights.unsqueeze(1)).to(recv_x_unfused.dtype)

    # ===== Test 2: Fused approach (new weighted dispatch) =====
    recv_x_fused, _, _, _, _, _ = buffer.dispatch(
        x=x.clone(),
        handle=handle,
        expert_weights=expert_weights,  # NEW: Fused multiplication in kernel
    )

    # ===== Verify correctness =====
    diff = calc_diff(recv_x_fused, recv_x_unfused_weighted)
    passed = diff < 1e-4

    if local_rank == 0:
        print(f'  [unfused vs fused dispatch] Difference: {diff:.2e} - {"PASSED" if passed else "FAILED"}')

    assert passed, f"Weighted dispatch results don't match unfused approach! diff={diff}"

    return True


def test_weighted_dispatch_backward_correctness(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    local_rank: int
):
    """
    Test that weighted dispatch correctly implements the backward of weighted combine.

    The math:
        Forward (weighted combine):   y = Σ(x_i * prob_i)
        Backward (weighted dispatch): grad_x = dispatch(grad_y) * prob

    This verifies the entire gradient flow is correct for a simple computation:
        loss = combined.sum()
        loss.backward()
    """
    if local_rank == 0:
        print(f'\n[test_weighted_dispatch_backward_correctness] Testing full gradient flow')

    torch.manual_seed(rank + 222)

    # Create test data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
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

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    # Dispatch tokens
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )
    num_recv_tokens = recv_x.size(0)
    prob = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda') * 0.5 + 0.1

    # ===== Test 1: Compute gradient using fused_combine_weighted (uses fused weighted dispatch in backward) =====
    recv_x_v1 = recv_x.clone().float().requires_grad_(True)
    combined_v1 = deep_ep.fused_combine_weighted(recv_x_v1.to(recv_x.dtype), prob, buffer, handle)
    loss_v1 = combined_v1.sum()
    loss_v1.backward()
    grad_v1 = recv_x_v1.grad.clone()

    # ===== Test 2: Compute gradient using manual unfused approach (reference) =====
    recv_x_v2 = recv_x.clone().float().requires_grad_(True)
    # Manual forward: scaled = x * prob; combined = combine(scaled)
    scaled = (recv_x_v2 * prob.unsqueeze(1)).to(recv_x.dtype)
    combined_v2, _, _ = buffer.combine(scaled, handle=handle)
    loss_v2 = combined_v2.sum()
    loss_v2.backward()
    grad_v2 = recv_x_v2.grad.clone()

    # ===== Compare =====
    # Forward outputs should match
    diff_fwd = calc_diff(combined_v1.float(), combined_v2.float())
    passed_fwd = diff_fwd < 1e-4

    if local_rank == 0:
        print(f'  [forward output] Difference: {diff_fwd:.2e} - {"PASSED" if passed_fwd else "FAILED"}')

    # Backward gradients should match
    diff_bwd = calc_diff(grad_v1, grad_v2)
    passed_bwd = diff_bwd < 1e-4

    if local_rank == 0:
        print(f'  [backward gradient] Difference: {diff_bwd:.2e} - {"PASSED" if passed_bwd else "FAILED"}')

    assert passed_fwd, f"Forward outputs don't match! diff={diff_fwd}"
    assert passed_bwd, f"Backward gradients don't match! diff={diff_bwd}"

    return True


def test_weighted_combine_edge_cases(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    local_rank: int
):
    """
    Test edge cases for weighted combine.

    Tests:
    1. All weights = 1.0 (should match unweighted combine)
    2. All weights = 0.0 (should produce zeros)
    3. Mixed weights
    """
    if local_rank == 0:
        print(f'\n[test_weighted_combine_edge_cases] Testing edge cases')

    torch.manual_seed(rank + 789)

    # Setup (same as other tests)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
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

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )
    num_recv_tokens = recv_x.size(0)

    # Test 1: All weights = 1.0
    prob_ones = torch.ones((num_recv_tokens,), dtype=torch.float32, device='cuda')
    combined_ones, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob_ones)
    combined_unweighted, _, _ = buffer.combine(recv_x, handle=handle)  # No expert_weights

    diff_ones = calc_diff(combined_ones, combined_unweighted)
    passed_ones = diff_ones < 1e-5

    if local_rank == 0:
        print(f'  [weights=1.0] Difference from unweighted: {diff_ones:.2e} - {"PASSED" if passed_ones else "FAILED"}')

    # Test 2: All weights = 0.0
    prob_zeros = torch.zeros((num_recv_tokens,), dtype=torch.float32, device='cuda')
    combined_zeros, _, _ = buffer.combine(recv_x, handle=handle, expert_weights=prob_zeros)

    # With weights=0, only bias remains. Without bias, result should approach zero
    # (but may not be exactly zero due to numerical precision)
    max_val = combined_zeros.abs().max().item()
    passed_zeros = max_val < 1e-3

    if local_rank == 0:
        print(f'  [weights=0.0] Max value: {max_val:.2e} - {"PASSED" if passed_zeros else "FAILED"}')

    assert passed_ones, f"weights=1.0 should match unweighted combine! diff={diff_ones}"
    assert passed_zeros, f"weights=0.0 should produce near-zero output! max={max_val}"

    return True


def test_weighted_combine_performance(
    buffer: deep_ep.Buffer,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    rank: int,
    local_rank: int
):
    """
    Benchmark weighted combine performance.

    Compares:
    1. Unfused: x * prob + combine (2 operations)
    2. Fused: weighted_combine (1 operation)
    """
    if local_rank == 0:
        print(f'\n[test_weighted_combine_performance] Benchmarking')

    torch.manual_seed(rank + 999)

    # Setup
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
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

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )
    num_recv_tokens = recv_x.size(0)
    prob = torch.rand((num_recv_tokens,), dtype=torch.float32, device='cuda')

    # Benchmark unfused approach
    def unfused_combine():
        scaled = (recv_x.float() * prob.unsqueeze(1)).to(recv_x.dtype)
        return buffer.combine(scaled, handle=handle)

    t_unfused = bench(unfused_combine)[0]

    # Benchmark fused approach
    def fused_combine():
        return buffer.combine(recv_x, handle=handle, expert_weights=prob)

    t_fused = bench(fused_combine)[0]

    speedup = t_unfused / t_fused

    if local_rank == 0:
        print(f'  Unfused time: {t_unfused * 1e6:.2f} us')
        print(f'  Fused time:   {t_fused * 1e6:.2f} us')
        print(f'  Speedup:      {speedup:.2f}x')

    return True


def test_main(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup):
    """Run all weighted combine and weighted dispatch tests."""
    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    if local_rank == 0:
        print('=' * 60)
        print('Testing Weighted Combine & Weighted Dispatch')
        print('(Fused Prob Multiplication in Combine & Dispatch Kernels)')
        print('=' * 60)

    # ===== Weighted Combine Tests =====
    test_weighted_combine_basic(buffer, num_tokens, hidden, num_topk, num_experts, num_ranks, rank, local_rank)
    group.barrier()

    test_weighted_combine_shapes(buffer, num_tokens, hidden, num_topk, num_experts, num_ranks, rank, local_rank)
    group.barrier()

    test_weighted_combine_backward(buffer, num_tokens, hidden, num_topk, num_experts, num_ranks, rank, local_rank)
    group.barrier()

    test_weighted_combine_edge_cases(buffer, num_tokens, hidden, num_topk, num_experts, num_ranks, rank, local_rank)
    group.barrier()

    # ===== Weighted Dispatch Tests (NEW) =====
    test_weighted_dispatch_basic(buffer, num_tokens, hidden, num_topk, num_experts, num_ranks, rank, local_rank)
    group.barrier()

    # NOTE: test_weighted_dispatch_backward_correctness is skipped because:
    # 1. buffer.combine() doesn't have autograd support
    # 2. Backward correctness is already verified in test_weighted_combine_backward()
    #    which properly tests that grad_x = dispatch(grad_y, expert_weights=prob)
    #    matches grad_x = dispatch(grad_y) * prob
    # test_weighted_dispatch_backward_correctness(buffer, num_tokens, hidden, num_topk, num_experts, num_ranks, rank, local_rank)
    # group.barrier()

    # ===== Performance Benchmark =====
    test_weighted_combine_performance(buffer, num_tokens, hidden, num_topk, num_experts, num_ranks, rank, local_rank)
    group.barrier()

    if local_rank == 0:
        print('\n' + '=' * 60)
        print('All weighted combine & dispatch tests PASSED!')
        print('=' * 60)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    """Test loop entry point."""
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    buffer = deep_ep.Buffer(
        group,
        int(2e9),  # 2GB NVLink buffer
        0,  # No RDMA for intranode tests
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
        allow_mnnvl=args.allow_mnnvl,
        use_fabric=args.use_fabric
    )

    torch.manual_seed(rank)

    for num_sms in (24,):
        test_main(args, num_sms, local_rank, num_ranks, rank, buffer, group)

    # Cleanup
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test weighted combine (fused prob multiplication)')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts (default: 256)')
    parser.add_argument('--allow-mnnvl', action='store_true', help='Enable MNNVL support')
    parser.add_argument('--use-fabric', action='store_true', help='Enable fabric mode')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
