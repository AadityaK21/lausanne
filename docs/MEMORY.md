# The 16 GB / 8 GB problem

Why this pipeline is built out of subprocesses instead of a single script.

## The budget

| | |
|---|---|
| System RAM | 16 GB, of which ~11–12 GB is usable after Windows |
| VRAM | 8188 MB (RTX 4060 Laptop) |
| Windows display reservation | ~900 MB, always |
| **Usable VRAM** | **~7288 MB** |

Stage ceilings, as declared in `config.yaml`:

| Stage | Ceiling | What is resident |
|---|---|---|
| ASR | 3600 MB | Whisper large-v3-turbo, int8_float16, batch 8 |
| Diarization | 2200 MB | pyannote community-1 segmentation + embedding |
| LLM notes | 6200 MB | 8B at Q4_K_M + quantised KV cache |
| **Combined** | **12000 MB** | vs 7288 MB usable |

The three stages cannot be co-resident. That is not a tuning problem; it is
1.65× over budget. The pipeline must sequence them, and — this is the part that
bites — it must sequence them in a way that actually returns the memory.

## Why `del model` is not enough

Three separate mechanisms hold memory after you think you have released it:

**PyTorch's caching allocator.** `torch` does not return freed CUDA blocks to
the driver. `torch.cuda.empty_cache()` helps but does not reliably return
everything, and the CUDA *context* itself (a few hundred MB) lives until the
process exits. Diarization loads torch; ASR does not (CTranslate2 is separate),
so the ordering matters less than the fact that neither fully unwinds
in-process.

**Ollama's residency.** By default Ollama keeps a model loaded for five minutes
after the last request, on the reasonable assumption that another request is
coming. Here, the next thing that wants VRAM is the *next run's ASR stage*, and
5.2 GB of resident weights is exactly what it cannot have. Hence
`keep_alive: 0` on every call, `OLLAMA_MAX_LOADED_MODELS=1`, and an explicit
unload before and after the LLM stage.

**Windows WDDM.** This is the one that makes debugging miserable. On Linux,
exceeding VRAM raises a CUDA OOM and you know immediately. On Windows the WDDM
driver silently spills to shared system memory. Nothing fails. The stage
completes. It is roughly fifty times slower, and there is no error anywhere to
tell you why. A run that should take ten minutes takes eight hours and looks,
from the outside, like the model is just "thinking".

That third one is why `run.py` logs VRAM state after every stage and why
`src/util/gpu.py::assert_headroom` warns before each allocation. The failure
being guarded against is not a crash — it is a silent slowdown.

## The design

`run.py` never imports torch, faster-whisper, pyannote, or anything CUDA-linked.
It shells out:

```python
subprocess.run([sys.executable, "-u", "src/stages/asr.py", audio,
                "--out", "work/x/transcript.json"], check=True)
```

Each stage reads JSON from disk, does its work, writes JSON to disk, and exits.
Process exit is the only eviction primitive that is guaranteed to work — the OS
reclaims the CUDA context whether or not the library cooperated.

Three consequences, two of them free:

- **Memory is actually released.** The point of the exercise.
- **Crash recovery.** Every intermediate is on disk. A failure in the LLM stage
  costs you the LLM stage, not the eight minutes of ASR that preceded it.
  `run.py` skips stages whose artifacts already exist; `--force` overrides.
- **Independent testability.** Each stage is a CLI with a typed input and a
  typed output, so `tests/test_pipeline.py` exercises every CPU path with
  synthetic fixtures, no GPU and no network.

## Context budget for the LLM stage

An 8B model at Q4_K_M is ~5.2 GB. That leaves roughly 1 GB inside the 6200 MB
ceiling for the KV cache.

At fp16 the cache costs ~128 KB per token for an 8B model, so 8K context is
~1 GB — the entire remaining headroom, with nothing left for activations.
Quantising the cache to `q8_0` halves it to ~64 KB/token, so 8K context costs
~0.5 GB and fits comfortably. Quality cost is negligible; this is set in
`src/util/ollama_client.py` and applies to every call.

Chunking follows from that: 5000 transcript tokens per map call with 500
overlap, leaving ~3000 tokens for instructions and the numbered output. A
one-hour hearing is 15–20K tokens, so 4–5 map calls plus one reduce.

## If you upgrade the RAM

64 GB of DDR5 (2×32, around ₹10,000) is worth doing for pipeline stability —
it removes the system-RAM pressure entirely and lets you raise `batch_size` and
run stages without watching the page file.

It does **not** rescue the 70B. With 8 GB of VRAM to offload into, Llama-3.3-70B
at Q4_K_M runs at roughly 1–1.5 tok/s. A 2000-token summary is a 25-minute
wait, per chunk. The 70B tier belongs on rented hardware; see
`docs/PAPER_REVISION_NOTES.md` §1.
