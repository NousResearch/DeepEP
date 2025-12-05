"""
Fused Combine Operations for DeepEP

This module provides fused combine operations that integrate probability multiplication
into the combine kernel for improved efficiency in MoE (Mixture of Experts) models.

NEW FEATURE: FusedCombineWeighted
---------------------------------
Fuses the probability multiplication into the combine operation:
    BEFORE: expert_out = h @ w2; scaled = expert_out * prob; combined = combine(scaled)
    AFTER:  expert_out = h @ w2; combined = fused_combine_weighted(expert_out, prob, ...)

This saves one memory read/write operation and improves performance.

Backward Correctness:
    Forward:  y = Σ(expert_out_i * prob_i)
    Backward: dy/d(expert_out_i) = prob_i
    So: grad_expert_out = dispatch(grad_y) * prob
"""

import torch
from typing import Tuple, Optional

# Import Buffer class from parent module
from .buffer import Buffer
from .utils import EventOverlap


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the hidden dimension size in bytes."""
    return x.size(1) * x.element_size()


class FusedCombineWeighted(torch.autograd.Function):
    """
    Fused combine operation with probability multiplication.

    This autograd function performs weighted combine in a single CUDA kernel:
        y = Σ(expert_out_i * prob_i)

    Instead of the two-step process:
        scaled = expert_out * prob  # Separate multiply (SLOW)
        y = Σ(scaled_i)             # Combine

    The backward pass correctly scales gradients by the probabilities:
        grad_expert_out = dispatch(grad_y) * prob

    Usage:
        # In MoE forward after expert computation:
        expert_out = h @ w2                                      # [num_tokens, dim]
        combined = fused_combine_weighted(expert_out, prob, buffer, handle)  # Fused!
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        prob: torch.Tensor,
        buffer: Buffer,
        handle: Tuple,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False
    ) -> torch.Tensor:
        """
        Forward: (expert_out * prob) then combine - fused into one CUDA kernel.

        Args:
            x: Expert output [num_local_tokens, dim] with torch.bfloat16
            prob: Routing probabilities [num_local_tokens] or [num_local_tokens, 1] with torch.float32
            buffer: DeepEP Buffer instance for communication
            handle: Communication handle from dispatch
            async_finish: If True, don't wait for kernel completion
            allocate_on_comm_stream: If True, allocate output tensors on comm stream

        Returns:
            combined_x: [num_original_tokens, dim] = Σ(x_i * prob_i)

        Note:
            The second return value (recv_topk_weights) from buffer.combine is ignored
            since we're using expert_weights instead of topk_weights for scaling.
        """
        # Ensure prob is float32 as required by DeepEP
        if prob.dtype != torch.float32:
            prob = prob.float()

        # Flatten prob if it's 2D with shape [N, 1]
        if prob.dim() == 2 and prob.size(1) == 1:
            prob = prob.squeeze(1)

        # Call combine with expert_weights (the new weighted combine feature)
        # This fuses the multiplication into the combine kernel
        combined_x, _, event = buffer.combine(
            x,
            handle=handle,
            expert_weights=prob,  # NEW: Fused multiplication in CUDA kernel
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Wait for completion if async
        if async_finish:
            event.current_stream_wait()

        # Save for backward
        ctx.save_for_backward(prob)
        ctx.handle = handle
        ctx.buffer = buffer

        return combined_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward: dispatch grad_y, then scale by prob.

        Math:
            Forward:  y = Σ(x_i * prob_i)
            Backward: dy/dx_i = prob_i
                     grad_x_i = grad_y * prob_i

        Implementation:
            1. Dispatch sends grad_y back to expert-owning ranks
            2. Scale by prob (chain rule): grad_x = dispatch(grad_y) * prob

        Note:
            We don't compute grad_prob because prob comes from the router
            which typically has its own gradient path through the gating mechanism.
        """
        prob, = ctx.saved_tensors
        handle = ctx.handle
        buffer = ctx.buffer

        # Dispatch sends grad_y back to expert-owning ranks
        # This is the reverse of combine (combine's backward IS dispatch)
        grad_x, _, _, _, _, _ = buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
        )

        # Scale by prob (chain rule)
        # grad_x shape: [num_local_tokens, dim]
        # prob shape: [num_local_tokens] or [num_local_tokens, 1]
        if prob.dim() == 1:
            prob = prob.unsqueeze(1)  # [N] -> [N, 1] for broadcasting
        grad_x = grad_x * prob

        # Return gradients for all forward arguments
        # (x, prob, buffer, handle, async_finish, allocate_on_comm_stream)
        # Only x needs gradient, prob gradient is not computed
        return grad_x, None, None, None, None, None


def fused_combine_weighted(
    x: torch.Tensor,
    prob: torch.Tensor,
    buffer: Buffer,
    handle: Tuple,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False
) -> torch.Tensor:
    """
    Fused: combine(x * prob) in single CUDA kernel.

    This function fuses the probability multiplication into DeepEP's combine kernel,
    eliminating a separate multiply operation for better performance.

    Args:
        x: Expert output [num_local_tokens, dim] with torch.bfloat16
        prob: Routing probabilities [num_local_tokens] or [num_local_tokens, 1] with torch.float32
        buffer: DeepEP Buffer instance for communication
        handle: Communication handle from dispatch
        async_finish: If True, don't wait for kernel completion
        allocate_on_comm_stream: If True, allocate output tensors on comm stream

    Returns:
        combined: [num_original_tokens, dim] = Σ(x_i * prob_i)

    Usage:
        # In MoE forward after expert computation:
        expert_out = h @ w2                                      # [num_tokens, dim]
        combined = fused_combine_weighted(expert_out, prob, buffer, handle)  # Fused!

    Performance:
        This saves one memory read/write operation compared to:
            scaled = expert_out * prob  # Separate multiply (SLOW)
            combined = buffer.combine(scaled, handle)
    """
    return FusedCombineWeighted.apply(x, prob, buffer, handle, async_finish, allocate_on_comm_stream)


# ====================================================================================
# Alternative: FusedCombine class (original unweighted version for reference)
# ====================================================================================

class FusedCombine(torch.autograd.Function):
    """
    Original fused combine operation (unweighted).

    This is the original combine operation that performs pure sum:
        y = Σ(x_i)

    Kept for backwards compatibility and comparison.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        buffer: Buffer,
        handle: Tuple,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False
    ) -> Tuple[torch.Tensor, Optional[EventOverlap]]:
        """
        Forward: combines expert outputs from all ranks (unweighted sum).

        Args:
            x: Expert output [num_local_tokens, dim]
            buffer: DeepEP Buffer instance
            handle: Communication handle from dispatch

        Returns:
            combined_x: [num_original_tokens, dim]
            event: EventOverlap (if async_finish=True)
        """
        combined_x, _, event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        ctx.handle = handle
        ctx.buffer = buffer

        return combined_x, event

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_event=None) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward: dispatches gradients back to expert-owning ranks.

        Uses dispatch() because combine's backward IS dispatch.

        Args:
            grad_output: Gradient from next layer
            grad_event: Gradient for event (unused, always None)
        """
        # grad_event is unused but required by autograd signature
        _ = grad_event

        handle = ctx.handle
        buffer = ctx.buffer

        grad_x, _, _, _, _, _ = buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
        )

        return grad_x, None, None, None, None


def fused_combine(
    x: torch.Tensor,
    buffer: Buffer,
    handle: Tuple,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False
) -> Tuple[torch.Tensor, Optional[EventOverlap]]:
    """
    Original fused combine (unweighted sum).

    This performs: y = Σ(x_i) without any weighting.

    For weighted combine (fused prob multiplication), use fused_combine_weighted().
    """
    return FusedCombine.apply(x, buffer, handle, async_finish, allocate_on_comm_stream)
