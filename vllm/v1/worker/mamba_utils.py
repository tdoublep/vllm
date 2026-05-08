# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import itertools
from collections.abc import Callable
from typing import Any

import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch


@triton.jit
def postprocess_mamba_fused_kernel(
    # Decision inputs (per-request)
    num_accepted_tokens_ptr,
    mamba_state_idx_ptr,
    num_scheduled_tokens_ptr,
    num_computed_tokens_ptr,
    num_draft_tokens_ptr,
    # Block table - shape [num_groups, max_reqs, max_blocks]
    block_table_ptr,
    block_table_stride_group: tl.int64,  # stride between groups (in elements)
    block_table_stride_req: tl.int64,  # stride between requests (in elements)
    # Mamba state metadata (per-layer, per-state-type)
    # These are 1D arrays indexed by (layer_idx * num_state_types + state_type_idx)
    state_base_addrs_ptr,  # base address of each state tensor
    state_block_strides_ptr,  # bytes per block for each state
    state_elem_sizes_ptr,  # element size for each state
    state_inner_sizes_ptr,  # number of elements in inner dimensions
    state_conv_widths_ptr,  # conv width for conv states (0 for temporal)
    state_group_indices_ptr,  # maps state_idx to group index in block table
    # Output: num_accepted_tokens update (for src==dst case)
    num_accepted_tokens_out_ptr,
    # Runtime parameter (varies per batch - NOT constexpr to avoid recompilation)
    num_reqs,
    # Compile-time constants (fixed after model initialization)
    # block_size: determined by model config, constant for all invocations
    block_size: tl.constexpr,
    # COPY_BLOCK_SIZE: fixed tuning parameter for memory copy loop
    COPY_BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GPU kernel for postprocess_mamba that computes decisions AND performs
    mamba state copies without any CPU-GPU synchronization.

    Grid: (num_reqs, num_layers * num_state_types)
    - program_id(0) = request index
    - program_id(1) = state_idx (flattened index into layer/state_type metadata)

    Note: num_layers and num_state_types are not passed as kernel parameters
    because the kernel indexes directly into pre-flattened metadata arrays
    using program_id(1). The grid dimensions encode the total state count.
    """
    req_idx = tl.program_id(0)
    state_idx = tl.program_id(1)

    # Bounds check
    if req_idx >= num_reqs:
        return

    # Compute decision logic (mirrors postprocess_mamba Python reference)
    num_accepted = tl.load(num_accepted_tokens_ptr + req_idx)
    src_block_idx = tl.load(mamba_state_idx_ptr + req_idx)
    num_scheduled = tl.load(num_scheduled_tokens_ptr + req_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_idx)
    num_draft = tl.load(num_draft_tokens_ptr + req_idx)

    num_tokens_running_state = num_computed + num_scheduled - num_draft
    new_num_computed = num_tokens_running_state + num_accepted - 1
    aligned_new_computed = (new_num_computed // block_size) * block_size

    needs_copy = aligned_new_computed >= num_tokens_running_state

    if not needs_copy:
        return

    # Compute copy parameters
    accept_token_bias = aligned_new_computed - num_tokens_running_state
    dest_block_idx = aligned_new_computed // block_size - 1

    # Load state metadata for this layer/state_type
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    # Load the group index for this state, then index into the correct
    # group's block table. Each mamba group has independently allocated
    # physical blocks.
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)

    # Load block IDs from block table. Cast to typed pointer BEFORE arithmetic
    # to ensure element-wise (not byte-wise) pointer advancement.
    # Promote loaded block ids to int64 *before* multiplying with the int64
    # block stride. block_table stores int32 ids; state_block_stride can be
    # tens of GB (base_addr + block_id * stride). Without the explicit cast
    # Triton may evaluate `block_id * stride` as int32, silently truncating
    # the product once block_id * stride exceeds 2**31 and producing a
    # wrong src/dst address for large mamba caches.
    block_table_typed = block_table_ptr.to(tl.pointer_type(tl.int32))
    block_table_base = (
        block_table_typed
        + group_idx * block_table_stride_group
        + req_idx * block_table_stride_req
    )
    src_block_id = tl.load(block_table_base + src_block_idx).to(tl.int64)
    dest_block_id = tl.load(block_table_base + dest_block_idx).to(tl.int64)

    # Compute source and destination addresses based on state type
    # conv_width > 0 means this is a conv state (get_conv_copy_spec logic)
    # conv_width == 0 means this is a temporal state (get_temporal_copy_spec logic)
    is_conv_state = conv_width > 0

    if is_conv_state:
        # Conv state: copy from state[src_block_id, accept_token_bias:] to
        # state[dest_block_id] Source offset is accept_token_bias elements into
        # the conv dimension
        src_offset = accept_token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        # Number of elements to copy:
        # (conv_width - accept_token_bias) * inner_size
        num_elems_to_copy = (conv_width - accept_token_bias).to(
            tl.int64
        ) * state_inner_size
        copy_size = num_elems_to_copy * state_elem_size
    else:
        # Temporal state: copy from state[src_block_id + accept_token_bias]
        # to state[dest_block_id]
        actual_src_block_idx = src_block_idx + accept_token_bias
        actual_src_block_id = tl.load(block_table_base + actual_src_block_idx).to(
            tl.int64
        )
        src_addr = state_base_addr + actual_src_block_id * state_block_stride
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        # Use natural block data size (inner_size * elem_size), NOT
        # state_block_stride which is the page stride and can exceed the
        # actual data when the state tensor uses as_strided page padding.
        copy_size = state_inner_size * state_elem_size

    # Match Python reference (collect_mamba_copy_meta): skip copy only when
    # src and dest are the same logical block with no token bias.
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        if state_idx == 0:
            tl.store(num_accepted_tokens_out_ptr + req_idx, 1)
        return

    # Perform the memory copy
    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for i in range(0, copy_size, COPY_BLOCK_SIZE):
        mask = (i + offsets) < copy_size
        curr_src = (src_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst = (dst_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        data = tl.load(curr_src, mask=mask)
        tl.store(curr_dst, data, mask=mask)

    # If src and dest block indices are the same (copy within same block),
    # set num_accepted_tokens to 1. This matches the Python postprocess_mamba
    # logic: "if src_block_idx == dest_block_idx: num_accepted_tokens_cpu[i] = 1"
    # Only one thread (state_idx == 0) should perform this update.
    if src_block_idx == dest_block_idx and state_idx == 0:
        tl.store(num_accepted_tokens_out_ptr + req_idx, 1)


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    BLOCK_SIZE = 1024
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    mamba_group_ids: list[int]
    mamba_spec: MambaSpec
    offset: int = 0

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaCopyBuffers":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
        entries_per_req = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        ) * len(copy_funcs)
        n = max_num_reqs * entries_per_req
        return cls(
            src_ptrs=make_buffer(n, dtype=torch.int64),
            dst_ptrs=make_buffer(n, dtype=torch.int64),
            sizes=make_buffer(n, dtype=torch.int32),
            mamba_group_ids=mamba_group_ids,
            mamba_spec=mamba_spec,
        )


@dataclasses.dataclass
class MambaGPUContext:
    """
    Context for GPU-side Mamba state copy operations.

    Precomputes memory layout metadata (base addresses, strides, element sizes)
    so the GPU kernel can perform state copies without CPU-GPU sync.

    State types are distinguished by conv_width: >0 for conv states (sliding
    window with offset-based copies), 0 for temporal states (full block copies).
    """

    # Per-state metadata tensors (shape: [num_layers * num_state_types])
    # These are populated from forward_context during the first forward pass
    state_base_addrs: torch.Tensor  # int64: base address of each state tensor
    state_block_strides: torch.Tensor  # int64: bytes per block
    state_elem_sizes: torch.Tensor  # int32: element size in bytes
    state_inner_sizes: torch.Tensor  # int64: elements in inner dimensions
    state_conv_widths: torch.Tensor  # int32: conv width (0 for temporal states)
    state_group_indices: torch.Tensor  # int32: maps state_idx to group index

    # Configuration
    block_size: int
    num_layers: int
    num_state_types: int
    mamba_group_ids: list[int]
    num_groups: int

    # Output buffer for num_accepted_tokens updates
    num_accepted_tokens_out: torch.Tensor

    # Flag to track if metadata has been populated
    is_initialized: bool = False

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        num_state_types: int,
        device: torch.device,
    ) -> "MambaGPUContext":
        """Create context with allocated buffers (metadata populated later)."""
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)

        # Count total layers across all mamba groups
        num_layers = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        )
        total_states = num_layers * num_state_types

        return cls(
            state_base_addrs=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_block_strides=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_elem_sizes=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            state_inner_sizes=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_conv_widths=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            state_group_indices=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            block_size=mamba_spec.block_size,
            num_layers=num_layers,
            num_state_types=num_state_types,
            mamba_group_ids=mamba_group_ids,
            num_groups=len(mamba_group_ids),
            num_accepted_tokens_out=torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            is_initialized=False,
        )

    def initialize_from_forward_context(
        self,
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],
        mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    ) -> None:
        """
        Extract and cache memory layout metadata from Mamba state tensors.

        This method populates the pre-allocated metadata tensors with information
        needed by `postprocess_mamba_fused_kernel` to perform state copies entirely
        on the GPU without CPU-GPU synchronization.

        For each Mamba layer and state type, the following metadata is extracted:
        - state_base_addrs: GPU memory address (data_ptr) of the state tensor
        - state_block_strides: Bytes between consecutive blocks (stride * elem_size)
        - state_elem_sizes: Element size in bytes (e.g., 2 for float16)
        - state_inner_sizes: For conv states, elements per conv position (stride(1)),
          used to compute offset when slicing state[block, offset:]. For temporal
          states, this field is unused (set to 1).
        - state_conv_widths: Conv dimension size for conv states, 0 for temporal states

        The conv vs temporal state type is detected by inspecting the copy function
        name: functions containing "conv" are treated as conv states.

        This method is idempotent - it only executes once (guarded by is_initialized
        flag) since the metadata is static after model loading.

        Design note:
            Triton kernels cannot accept PyTorch tensor objects directly. By
            pre-computing pointer arithmetic parameters (base addresses, strides,
            element sizes), the GPU kernel can calculate source/destination
            addresses using simple arithmetic:
                src_addr = base_addr + block_id * block_stride + offset
            This avoids CPU involvement during inference and eliminates the
            CPU-GPU synchronization that would otherwise be required.

        Args:
            kv_cache_config: Configuration containing KV cache group info and
                layer name mappings.
            forward_context: Dictionary mapping layer names to attention objects,
                populated after the model is loaded. Each attention object must
                have a `kv_cache` attribute containing the list of state tensors.
            mamba_state_copy_funcs: Tuple of copy functions (one per state type)
                used to determine whether each state is a conv or temporal state.
        """
        if self.is_initialized:
            return

        idx = 0
        for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
            for layer_name in layer_names:
                attention = forward_context[layer_name]
                kv_caches: list[torch.Tensor] = attention.kv_cache

                for state_type_idx, state in enumerate(kv_caches):
                    # Base address
                    self.state_base_addrs[idx] = state.data_ptr()

                    # Block stride (bytes between consecutive blocks)
                    # state shape: [num_blocks, ...], stride(0) = elements per block
                    if state.dim() > 1:
                        block_stride_elems = state.stride(0)
                    else:
                        block_stride_elems = state.numel()
                    self.state_block_strides[idx] = (
                        block_stride_elems * state.element_size()
                    )

                    # Element size
                    self.state_elem_sizes[idx] = state.element_size()

                    copy_func = mamba_state_copy_funcs[state_type_idx]
                    assert (
                        copy_func is get_conv_copy_spec
                        or copy_func is get_temporal_copy_spec
                    ), f"unexpected copy func: {copy_func}"
                    if copy_func is get_conv_copy_spec:
                        # Conv state: conv_width is state.size(1)
                        # inner_size is stride(1) = elements per conv position,
                        # used to compute byte offset for state[block, offset:]
                        conv_w = state.size(1) if state.dim() > 1 else 0
                        self.state_conv_widths[idx] = conv_w
                        if state.dim() > 2:
                            # stride(1) = product of dims[2:] for contiguous tensor
                            self.state_inner_sizes[idx] = state.stride(1)
                        else:
                            # 2D tensor: [num_blocks, conv_dim], no inner dims
                            self.state_inner_sizes[idx] = 1
                    else:
                        # Temporal state: inner_size = natural elements per
                        # block (prod of inner dims).  The kernel uses this
                        # to compute copy_size = inner_size * elem_size,
                        # which gives the correct byte count even when the
                        # state tensor is as_strided with padded page strides
                        # (state_block_stride would be the page size, too big).
                        self.state_conv_widths[idx] = 0
                        self.state_inner_sizes[idx] = (
                            state[0].numel() if state.dim() > 1 else 1
                        )

                    self.state_group_indices[idx] = group_local_idx
                    idx += 1

        self.is_initialized = True

    def run_fused_postprocess(
        self,
        num_reqs: int,
        num_accepted_tokens_gpu: torch.Tensor,
        mamba_state_idx_gpu: torch.Tensor,
        num_scheduled_tokens_gpu: torch.Tensor,
        num_computed_tokens_gpu: torch.Tensor,
        num_draft_tokens_gpu: torch.Tensor,
        block_table_gpu: torch.Tensor,
    ) -> None:
        """
        Run the fused postprocess_mamba kernel on GPU.

        This computes decisions and performs mamba state copies entirely on GPU,
        eliminating the CPU-GPU sync that was previously needed.

        Args:
            num_reqs: Number of active requests
            num_accepted_tokens_gpu: [num_reqs] accepted token counts
            mamba_state_idx_gpu: [num_reqs] source block indices
            num_scheduled_tokens_gpu: [num_reqs] scheduled token counts
            num_computed_tokens_gpu: [num_reqs] computed token counts
            num_draft_tokens_gpu: [num_reqs] draft token counts
            block_table_gpu: [num_groups, num_reqs, max_blocks] block IDs
                per group per request
        """
        if num_reqs == 0 or not self.is_initialized:
            return

        # Initialize output to current values (unchanged unless src==dst)
        self.num_accepted_tokens_out[:num_reqs].copy_(
            num_accepted_tokens_gpu[:num_reqs]
        )

        total_states = self.num_layers * self.num_state_types
        grid = (num_reqs, total_states)

        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_gpu,
            mamba_state_idx_gpu,
            num_scheduled_tokens_gpu,
            num_computed_tokens_gpu,
            num_draft_tokens_gpu,
            block_table_gpu,
            block_table_gpu.stride(0),
            block_table_gpu.stride(1),
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            self.num_accepted_tokens_out,
            num_reqs,
            block_size=self.block_size,
            COPY_BLOCK_SIZE=1024,
        )


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        return
    batch_memcpy(
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
    )


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.

    Uses input_batch.mamba_state_idx_cpu[req_index] to track which block
    contains the running mamba state for each request. -1 means "no previous
    state" (new/resumed request, compute from num_computed_tokens).
    """
    assert input_batch.mamba_state_idx_cpu is not None, (
        "mamba_state_idx_cpu is None - preprocess_mamba should only be called "
        "for hybrid models (is_hybrid=True)"
    )
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size

    # Clear mamba_state_idx for finished/preempted/resumed requests.
    # Mirrors the pre-refactor dict-based path: sweep all three sets as a
    # safety net. remove_request() already resets finished/preempted slots,
    # but force-preemption paths (e.g. reset_prefix_cache / KV cache flush)
    # can surface a request in resumed_req_ids without a matching
    # preempted_req_ids entry, leaving a stale curr_state_idx that would
    # point past the new (smaller) block allocation.
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        req_index = input_batch.req_id_to_index.get(req_id)
        if req_index is not None:
            input_batch.mamba_state_idx_cpu[req_index] = -1

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = input_batch.mamba_state_idx_cpu[i]
        if prev_state_idx == -1:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        input_batch.mamba_state_idx_cpu[i] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)
