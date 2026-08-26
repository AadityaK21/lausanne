# What is in this folder

Installed into `C:\lausanne` on 25 August 2026.

## Your original prototype is preserved

Everything that was here before now lives under **`prototype/`** — `a2n/`,
`main.py`, `make_sample.py`, the old `README.md`, `requirements.txt`,
`tests/`, `output/`, `.vscode/`, and `sample.wav`.

Nothing was deleted. That code is the system Section 8 of your survey
describes, which makes it evidence rather than clutter: if a reviewer asks
what the paper's prototype actually did, the answer needs to still exist.

`prototype/a2n/chatbot.py` is worth a second look — see the note at the
bottom of this file.

## What is new at the root

| Path | What it is |
|---|---|
| `app/` | The local application. Standard library only, no build step. |
| `src/stages/` | The five pipeline stages, each a separate process. |
| `src/eval/` | The evaluation harness — WER, DER, faithfulness, judge. |
| `src/util/` | I/O, logging, GPU budget, Ollama client, provenance. |
| `tests/` | 96 checks that need no GPU, no models, no network. |
| `docs/` | Setup, memory design, design principles, Section 8 rewrite, report. |
| `prompts/` | Editable prompt templates. |
| `data/audio/sample.wav` | A working copy of your sample recording. |

`docs/legal_ai_survey_paper.pdf` moved into `docs/` with the other paper
material.

## First thing to run

No installation needed for this — it opens the application against synthetic
fixtures:

```powershell
cd C:\lausanne
python app\server.py --demo
```

Then the offline test suites:

```powershell
python tests\test_pipeline.py     # 68 checks
python tests\test_app.py          # 28 checks
```

Full setup for the real pipeline: `docs\SETUP_WINDOWS.md`.

## Two housekeeping notes

`_to_delete\` holds the installer archive. The bridge I used cannot delete
files on your machine, so I moved it there rather than leaving it at the root.
Remove that folder whenever you like.

Your old `.vscode/` config pointed at `main.py` and moved to `prototype/` with
the rest. You will want a fresh run configuration for `app\server.py` and
`run.py`.

## One thing your prototype has that the new build does not

`prototype/a2n/chatbot.py` is a local conversational assistant over a
transcript, with genuinely careful context management — token budgeting,
system-prompt pinning, and eviction in whole user/assistant pairs so the model
never reads an answer to a question it cannot see.

There is nothing equivalent in the new pipeline, and "ask a question about this
hearing" is an obvious thing to want from the application.

If it gets integrated, it needs the treatment the abstractive mode already
gets: a chat answer is generated text, so it carries full fabrication risk and
must be visually marked as paraphrase with citations into the record, exactly
as `render.py` and the interface already do for abstractive notes. Folding it
in without that would put an unmarked generative surface into a tool whose
whole argument is that the user can always see where trust is being spent.

---

# Transcribing your own audio

`--demo` never touches your file. It substitutes a synthetic sample so the
interface can be exercised with nothing installed. To process a real recording
you must run without it.

## Minimum install — no CUDA, no Hugging Face token, no Ollama

Speech recognition alone needs none of those. This is enough:

```powershell
cd C:\lausanne
pip install faster-whisper pydantic requests PyYAML python-docx
```

`faster-whisper` uses CTranslate2, not PyTorch, so this pulls no multi-gigabyte
GPU stack and decodes .m4a on its own.

## Run it

```powershell
python run.py "data\audio\Recording _3_.m4a" --mode extractive-rule --no-diarize --device cpu --asr-model small
```

What each flag buys you:

- `--mode extractive-rule` — no language model, so nothing to pull from Ollama.
- `--no-diarize` — skips speaker identification, which is the part that needs a
  Hugging Face token and gated model downloads.
- `--device cpu` — no CUDA required.
- `--asr-model small` — `large-v3-turbo` on CPU is slow; `small` is a
  reasonable first pass. Drop the flag once the GPU stack is installed.

Output lands in `output\Recording__3_\`. The first run downloads the Whisper
weights (~500 MB for `small`), so it takes a few minutes; later runs do not.

## Then add the rest, in this order

1. **CUDA PyTorch** — `docs\SETUP_WINDOWS.md` step 2. Drop `--device cpu` and
   `--asr-model small` afterwards.
2. **Hugging Face token** — step 4. Drop `--no-diarize` to get speaker labels.
3. **Ollama models** — step 5. Switch to `--mode extractive-llm`.

Each step is independently useful. You do not need all three to get a real
transcript of your own audio.

## A note on resuming

Every stage writes its result to `work\<name>\` and `run.py` skips stages whose
output already exists. That saves re-transcribing when you only want to change
the summarisation — but it also means a stale artefact gets reused silently.
Pass `--force` when you want a genuinely fresh run.
