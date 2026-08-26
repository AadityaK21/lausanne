# Setup — Windows, RTX 4060 (8 GB), 16 GB RAM

Every step, in order. Roughly 45 minutes, most of it downloads.

Two decisions are already made for you, and both reverse advice you may have
seen elsewhere:

- **Do not use WSL2.** Ollama runs natively on Windows and a WSL2 VM reserves
  its own memory pool on top of it. With 16 GB of system RAM that reservation
  is the one resource you cannot spare. `faster-whisper` and `pyannote` both
  work fine on native Windows with CUDA.
- **Do not run this from `System32` or any admin-owned directory.** Everything
  below assumes `C:\lausanne`.

---

## 0. Prerequisites

| Requirement | Check | If missing |
|---|---|---|
| NVIDIA driver ≥ 550 | `nvidia-smi` | [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx) |
| Python 3.11 or 3.12 | `py -0p` (lists every Python) | [python.org](https://www.python.org/downloads/) — tick *Add to PATH* |
| ffmpeg | `ffmpeg -version` | `winget install Gyan.FFmpeg` |
| Ollama | `ollama --version` | [ollama.com/download](https://ollama.com/download) |
| Git (optional) | `git --version` | `winget install Git.Git` |

Python 3.13 and 3.14 are not recommended yet — several audio wheels still lag,
`ctranslate2` in particular. If you have more than one Python installed, note
which one you want *before* creating the venv: `py -0p` lists them with paths,
and the one marked `*` is what a bare `python` resolves to.

---

## 1. Project directory and virtual environment

```powershell
mkdir C:\lausanne -Force
cd C:\lausanne
# copy the project files here first

py -3.12 -m venv .venv          # name the version explicitly
.\.venv\Scripts\Activate.ps1
```

`py -3.12` rather than `python`, because `python` resolves to whichever
install is first on PATH — which on a machine with several is rarely the one
you meant.

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Your prompt should now start with `(.venv)`. Every command below assumes it.

```powershell
python -m pip install --upgrade pip
```

---

## 2. PyTorch with CUDA — this must come first

**Order matters.** If you install `requirements.txt` first, pip pulls a
CPU-only torch as a transitive dependency, everything appears to install
cleanly, and diarization then runs 5× slower with no error message.

```powershell
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify before continuing — do not skip this:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want `True` and `NVIDIA GeForce RTX 4060 Laptop GPU`. If you get `False`,
your driver is too old or you installed the CPU wheel; uninstall
(`pip uninstall torch torchaudio`) and repeat.

---

## 3. The rest of the dependencies

```powershell
python -m pip install -r requirements.txt
```

### `Could not locate cudnn_ops64_9.dll` — handled for you

This used to be the single most common failure on Windows, and **the pipeline
now fixes it itself**. Nothing in this section is a step you have to perform;
it is here so the mechanism is not a mystery, and because most of the advice
you will find online is wrong on Windows.

`faster-whisper` >= 1.1 uses CTranslate2 >= 4.5, which links cuDNN 9 and cuBLAS
at extension-load time and resolves them through the OS loader. The CUDA build
of PyTorch from step 2 already ships those exact DLLs — on Windows it bundles
them in `.venv\Lib\site-packages\torch\lib`, a directory the loader has no
reason to search. Card fine, driver fine, DLLs on disk, import fails.

`src/stages/asr.py` registers that directory with `os.add_dll_directory()`
before importing `faster_whisper`. Python 3.8+ deliberately stopped honouring
`PATH` for extension-module dependencies on Windows, so this is the supported
mechanism — and it affects only that one process, leaving your environment
untouched.

**Do not follow the `import nvidia.cudnn` advice** found in most tutorials and
in earlier versions of this file. It is written for Linux, where
`pip install torch --index-url .../cu126` pulls the CUDA runtime in as separate
`nvidia-cudnn-cu12` distributions. The Windows wheel has no such dependency —
there is no `nvidia` package to import — so that command can only ever raise
`ModuleNotFoundError` on a correctly installed Windows environment. Seeing it
is not a sign that anything is missing.

To confirm the DLLs are visible:

```powershell
python run.py doctor      # reports where cuDNN/cuBLAS were found
```

If it reports them missing, the cause is that torch is a CPU-only build.
Reinstall from the CUDA index (step 2) rather than installing anything
cuDNN-specific:

```powershell
python -m pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## 4. Hugging Face token (pyannote is gated)

Do this now. Discovering it mid-build is a classic time sink.

1. Accept the conditions at
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   *and* [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   — the pipeline pulls both.
2. Create a **read** token at [hf.co/settings/tokens](https://hf.co/settings/tokens).
3. Set it permanently:

```powershell
setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxx"
```

`setx` only affects *new* terminals. Close this one and open a new one, then
re-activate the venv.

---

## 5. Ollama models and environment

```powershell
ollama pull qwen3:8b        # generator, ~5.2 GB
ollama pull llama3.1:8b     # evaluation judge, ~4.7 GB
```

Set these permanently so Ollama never quietly holds a second model resident:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE","0","User")
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS","1","User")
```

**Restart Ollama from the system tray** after setting those — it reads its
environment at start.

Confirm nothing is resident:

```powershell
ollama ps      # empty output = good
```

> Note: `curl` in PowerShell is an alias for `Invoke-WebRequest` and does not
> accept `-d`. Use `ollama stop <model>` to unload, or
> `Invoke-RestMethod` if you need the raw HTTP call. From Python it is just
> `requests.post(...)`, which is what this pipeline does.

---

## 6. Verify everything

```powershell
python run.py doctor
```

This checks the GPU, the VRAM budget, every Python dependency, the CUDA build
of torch, `HF_TOKEN`, Ollama reachability, both models, and whether anything is
currently resident in VRAM. Resolve every ✗ before continuing.

Then run the offline test suite — no models, no network, ~2 seconds:

```powershell
python tests\test_pipeline.py
```

187 checks should pass.

---

## 7. First real run

Get public court audio with a gold transcript:

```powershell
python src\eval\fetch_oyez.py --term 2019 --limit 2
```

Run the pipeline:

```powershell
python run.py data\audio\2019_17-1618.mp3 --mode all --material public
```

Runs are classified `privileged` by default, which forbids any non-local
inference endpoint. The Oyez recordings are public, so `--material public` is
correct for them — and is the only setting under which `--tier hosted` will run
at all. Use it for published court recordings, never to silence the guard on
material you would not publish.

Expect roughly, for one hour of audio:

| Stage | Wall clock | Peak VRAM |
|---|---|---|
| ASR (large-v3-turbo, int8, batched) | 4–8 min | ~2.5 GB |
| Diarization (community-1) | 2–4 min | ~1.5 GB |
| Merge | < 1 s | 0 |
| Notes, extractive-rule | < 1 s | 0 |
| Notes, extractive-llm (qwen3:8b) | 3–5 min | ~5.5 GB |
| Notes, abstractive (qwen3:8b) | 3–5 min | ~5.5 GB |

Output lands in `output\<name>\` as Markdown, Word, two transcript files, and
`attestation.txt` — the confidentiality record for the run. Re-check it later
with:

```powershell
python run.py --verify work\2019_17-1618
```

---

## 8. Evaluation

```powershell
# transcription, diarization, coverage, verbatim rate — no LLM needed
python src\eval\report.py --gold data\gold --work work

# add the checklist judge over abstractive claims (slow)
python src\eval\report.py --gold data\gold --work work --judge

# measure whether the judge can be hijacked by the text it evaluates
python src\eval\report.py --gold data\gold --work work --judge --injection-test
```

For action-item and decision precision/recall you need expert annotations:

```powershell
python src\eval\extraction_pr.py --gold data\gold\<case>.json --make-template
#   ... fill in the generated .annotate.txt file ...
python src\eval\extraction_pr.py --gold data\gold\<case>.json --apply-annotations data\gold\<case>.annotate.txt
```

Results are written to `results\results.md` and `results\results.json`.

---

## Mixed Python versions

The failure looks like this, and it is worth recognising on sight:

```
ImportError: Importing the numpy C-extensions failed.
The following compiled module files exist, but seem incompatible
with either python 'cpython-314' or the platform 'win32':
  * _multiarray_umath.cp312-win_amd64.pyd
```

The venv is running one Python; some of the packages inside it were built for
another. `numpy`, `ctranslate2`, `lxml` and `torch` all ship version-locked
binaries, so whichever one imports first is the one that reports the error —
which is why the message names NumPy even when NumPy is not what you installed
last.

**How a venv ends up in this state.** `pip` and `python -m pip` are not the
same command. `python -m pip` is guaranteed to be the pip belonging to the
interpreter you are running. Bare `pip` is whatever `pip.exe` is first on PATH,
and that may belong to an entirely different install. The subtlety is that such
a pip still honours `VIRTUAL_ENV` when deciding *where to write* — so it
downloads wheels for **its own** interpreter and installs them into **your**
venv. No warning, no error, correct-looking output. Activating the venv does
not protect you, because activation changes PATH but a system-wide pip may
still win.

**Diagnose it:**

```powershell
python run.py doctor
```

`doctor` reads the wheel metadata of everything installed and lists any package
built for a different Python than the one running. It does this without
importing anything, so it works even when the environment is too broken to
import NumPy.

To see it by hand:

```powershell
python -c "import sys; print(sys.version)"
py -0p                       # every Python on the machine, * marks the default
Get-Command pip | Select-Object -ExpandProperty Source
```

If the `pip` path is not inside `C:\lausanne\.venv\Scripts\`, that is the
culprit.

**Fix it.** Do not try to repair the mixture package by package — you will not
find them all, and a partially repaired environment fails the same way later.
Rebuild on one interpreter:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python run.py doctor
```

Nothing outside `.venv` is touched — no downloaded models, no work artifacts,
no Ollama models. Only the packages are refetched.

**Use `python -m pip` for the rest of the project's life.** It is three extra
characters and it removes this entire class of failure.

---

## Troubleshooting

**Everything is suddenly 50× slower.** Something is resident that should not
be. `ollama ps` and `nvidia-smi`. On Windows the driver spills to shared system
memory instead of raising OOM, so this fails silently rather than loudly.

**`Pipeline.from_pretrained` returns `None`.** The gate has not been accepted
for your account, or the token lacks read scope. Accept *both* repos in step 4.

**Out of memory during the LLM stage.** Lower `chunk_tokens` in `config.yaml`,
or drop to a smaller generator (`llama3.1:8b`, `mistral:7b`, `gemma3:4b`).
Anything 12B+ at Q4 collides with the ~900 MB Windows reserves for display.

**ASR produces repeated text.** `condition_on_previous_text` is already `false`
in the config, which is the usual cause. If it persists, the audio likely has a
long silence — `vad_filter: true` should handle it.

**A stage crashed and I don't want to re-run ASR.** You don't have to. Every
stage writes its artifact to `work\<name>\` and `run.py` skips completed stages.
Use `--force` only when you actually want a rerun.
