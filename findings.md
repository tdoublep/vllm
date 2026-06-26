# Breakable CUDA Graphs + Inductor Compilation in vLLM

## Question

Can the breakable CUDA graph feature
(`VLLM_USE_BREAKABLE_CUDAGRAPH=1`) be combined with vLLM's
Inductor-based torch.compile pipeline (`CompilationMode.VLLM_COMPILE`)?

When breakable was first introduced, enabling it unconditionally
forced `CompilationMode.NONE`, making the two features appear
mutually exclusive. The question is whether that exclusion reflects a
real technical incompatibility or a current implementation choice
that can be relaxed.

## Conclusion

The exclusion is not a hard constraint. Breakable cudagraphs and
Inductor compilation are orthogonal and compose cleanly with two
small changes (see Patch). A Qwen2-1.5B-Instruct smoke test under
`mode=VLLM_COMPILE` with breakable enabled compiles via Dynamo +
Inductor, captures the model under the breakable wrapper, and
produces byte-identical output to the standard piecewise-cudagraph
baseline.

## Background

Three concerns are often conflated. They are independent:

1. **Dynamo** traces the Python model into an FX graph.
2. **Inductor** lowers the FX graph into fused / Triton kernels.
3. **CUDA graph capture** records the resulting GPU work for replay.

vLLM's `CompilationMode` controls (1) and (2). Cudagraph capture
strategy is separate — historically two strategies:

- **Piecewise cudagraphs**: `split_graph` in
  `vllm/compilation/backends.py` slices the FX graph at
  `splitting_ops` (attention ops, KV cache update ops). Inductor
  compiles each non-attention region; attention ops stay as opaque
  custom-op calls. Each compiled region is wrapped in
  `CUDAGraphWrapper` for capture.
- **Breakable cudagraphs**: a single capture context drives the whole
  forward. Attention custom ops carry an
  `@eager_break_during_capture` decorator that, at runtime, ends the
  current cudagraph segment, runs attention eagerly, and resumes
  capture. Code in `vllm/compilation/breakable_cudagraph.py`.

These two strategies solve the same problem — keeping uncapturable
attention out of the cudagraph — via different mechanisms.
Piecewise solves it at compile time (via FX splitting); breakable
solves it at runtime (via stream-capture breaks). They are
substitutes for each other, not for Inductor.

## What was blocking the combination

Two pieces of code treated breakable as exclusive with the entire
torch.compile pipeline rather than just with piecewise capture:

1. `vllm/config/vllm.py:1099-1104` — when
   `VLLM_USE_BREAKABLE_CUDAGRAPH` is set, the post-init unconditionally
   set `CompilationMode.NONE`, disabling Dynamo and Inductor entirely.

2. `vllm/compilation/backends.py:650-654` —
   `wrap_with_cudagraph_if_needed` installs the per-piece
   `CUDAGraphWrapper(PIECEWISE)` whenever
   `cudagraph_mode.has_piecewise_cudagraphs()` is true. With
   breakable enabled the outer `BreakableCUDAGraphWrapper` is the
   sole capture mechanism, so the per-piece wrapper would be a
   redundant nested capture.

Neither of these is structural. The breakable wrapper is installed
*after* the model is built (`gpu_model_runner.py:5307`), so it
naturally wraps whatever callable `support_torch_compile` produced
— compiled or eager.

## Patch

Two files, seven net lines added. Full diff at the end of this
document.

**`vllm/config/vllm.py`** — only default `mode = NONE` when the user
did not explicitly pick a compilation mode:

```python
if envs.VLLM_USE_BREAKABLE_CUDAGRAPH and self.compilation_config.mode in (
    None,
    CompilationMode.NONE,
):
    self.compilation_config.mode = CompilationMode.NONE
```

**`vllm/compilation/backends.py`** — in `wrap_with_cudagraph_if_needed`,
suppress the per-piece `CUDAGraphWrapper` when breakable is on:

```python
from vllm.compilation.breakable_cudagraph import (
    is_breakable_cudagraph_enabled,
)

if (
    not compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
    or compilation_config.use_inductor_graph_partition
    or is_breakable_cudagraph_enabled()
):
    return piecewise_backend
```

The dispatcher assertion at `vllm/v1/cudagraph_dispatcher.py:49-61`
already had `is_breakable_cudagraph_enabled()` as an accepted branch
alongside `is_attention_compiled_piecewise()`; with the patch, both
are simultaneously true and the assertion passes.

## How it runs after the patch

Forward pass under `mode=VLLM_COMPILE` + breakable enabled, default
`splitting_ops` (the attention ops):

1. Dynamo traces the `@support_torch_compile` module into an FX graph.
2. `VllmBackend.split_graph` splits at attention / kv-cache ops into
   ~`2N + 1` subgraphs for an N-layer model: alternating non-attention
   regions and single-op attention regions.
3. `PiecewiseCompileInterpreter` sends every non-attention region to
   Inductor (via `compiler_manager.compile` →
   `PiecewiseBackend.compile_all_ranges`). Custom vLLM FX passes run
   on each region before Inductor lowering. Attention regions are
   not sent to Inductor.
4. With the patch, `wrap_with_cudagraph_if_needed` returns the
   `PiecewiseBackend` unwrapped — no per-piece cudagraph.
5. `BreakableCUDAGraphWrapper` wraps the whole compiled model
   (`gpu_model_runner.py:5307`).
6. At runtime, the dispatcher selects a batch descriptor. On first
   call for a given descriptor, breakable opens a capture context.
   Inductor-compiled regions execute on the capture stream and
   their kernels are recorded; between regions, attention's
   `@eager_break_during_capture` decorator ends the current segment,
   runs attention eagerly, and starts a fresh segment. On subsequent
   calls for the same descriptor, the recorded segments are replayed
   in order.

## Verification

Smoke test: `Qwen/Qwen2-1.5B-Instruct`, single-GPU H100, greedy
decode, prompt `"Hello, my name is"`, `max_tokens=16`.

| Configuration | Output |
|---|---|
| `mode=VLLM_COMPILE`, breakable off (baseline piecewise) | ` John and I am a 2017 graduate of the University of South` |
| `mode=VLLM_COMPILE`, breakable on (this patch) | ` John and I am a 2017 graduate of the University of South` |

Output bytes are identical. Compile and capture both ran:

```
INFO breakable_cudagraph.py:288 Breakable CUDA graph enabled
INFO backends.py:1153 Dynamo bytecode transform time: 3.55 s
INFO backends.py:393 Compiling a graph for compile range (1, 16384) takes 5.59 s
INFO monitor.py:53 torch.compile took 11.57 s in total
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|...| 51/51
Capturing CUDA graphs (decode, FULL): 100%|...| 51/51
INFO core.py:337 init engine took 22.68 s (compilation: 11.57 s)
```

Unit tests in `tests/v1/cudagraph/test_breakable_cudagraph.py` pass
(12/12) — these exercise the wrapper's primitives independently of
the model runner, so they cover the breakable side regardless of
compilation mode.

## Recommended usage

For users who want Inductor optimization plus breakable capture:

```bash
VLLM_USE_BREAKABLE_CUDAGRAPH=1 vllm serve <model> -O.mode=3
```

Or via `LLM(...)`:

```python
from vllm.config import CompilationConfig, CompilationMode

LLM(
    model=...,
    compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
)
# with VLLM_USE_BREAKABLE_CUDAGRAPH=1 in the environment
```

Leave `splitting_ops` at the default. FX-splitting at attention is
the right configuration here: it produces the same per-region
Inductor compilation that piecewise cudagraphs already relied on,
and breakable replaces the per-region capture with one outer
capture broken at the same boundaries.

Setting `splitting_ops=[]` (one monolithic Inductor region with the
attention op left inline) also works and was tested — it preserves
output bit-exactly — but extends Inductor's fusion scope across
attention boundaries at the cost of significantly longer compile
times (e.g., 23 s vs 6 s on Qwen2-1.5B). Whether the extra fusion
opportunities pay off is empirical and likely model-dependent.

## What was not investigated

- **Inductor graph partition path**
  (`use_inductor_graph_partition=True`). This is a different splitting
  strategy that operates inside Inductor codegen rather than at the
  FX level. The patch leaves it untouched. Combining it with
  breakable would need its own validation.
- **Models with custom attention ops not decorated with
  `@eager_break_during_capture`.** The decorator is the runtime
  signal that ends a capture segment; any custom op that should be
  treated as a break point must carry it.
- **Larger models, multi-GPU (TP > 1) configurations, speculative
  decoding paths.** The breakable wrapper has its own draft-model
  hook in `gpu_model_runner.py:5310`, but only the single-GPU
  non-spec path was exercised here.
- **Performance comparison.** Functional correctness was verified;
  whether breakable + Inductor is faster or slower than
  piecewise + Inductor on real workloads is a separate measurement.

## Full diff

```diff
diff --git a/vllm/compilation/backends.py b/vllm/compilation/backends.py
index dc12acbaf..c9301e630 100644
--- a/vllm/compilation/backends.py
+++ b/vllm/compilation/backends.py
@@ -647,9 +647,14 @@ def wrap_with_cudagraph_if_needed(
     Returns:
         The wrapped backend if CUDA graphs are enabled, otherwise the original backend
     """
+    from vllm.compilation.breakable_cudagraph import (
+        is_breakable_cudagraph_enabled,
+    )
+
     if (
         not compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
         or compilation_config.use_inductor_graph_partition
+        or is_breakable_cudagraph_enabled()
     ):
         return piecewise_backend

diff --git a/vllm/config/vllm.py b/vllm/config/vllm.py
index ba7d26c93..30b807817 100644
--- a/vllm/config/vllm.py
+++ b/vllm/config/vllm.py
@@ -1096,10 +1096,13 @@ class VllmConfig:
                 "Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out."
             )

-        if envs.VLLM_USE_BREAKABLE_CUDAGRAPH:
+        if envs.VLLM_USE_BREAKABLE_CUDAGRAPH and self.compilation_config.mode in (
+            None,
+            CompilationMode.NONE,
+        ):
             logger.warning_once(
-                "VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's "
-                "torch.compile pipeline. Equivalent to -cc.mode=none."
+                "VLLM_USE_BREAKABLE_CUDAGRAPH is set with no explicit "
+                "compilation mode, defaulting to -cc.mode=none."
             )
             self.compilation_config.mode = CompilationMode.NONE
```
