# Audio-to-Notes

A **fully local, API-free** pipeline that turns spoken audio into structured,
searchable notes. This is the reference implementation of the system described
in the paper *"Natural Language Processing and Large Language Models for Legal
AI"* (Section 8).

Everything runs on your own machine:

1. **Transcription** — local **OpenAI Whisper** (on-device, no API).
2. **Note generation** — two local modes:
   - **`extractive`** — deterministic, rule-based. Builds a frequency-based
     summary, and extracts **action items**, **decisions**, **keywords**, and
     a **time-stamped transcript**. Because every output sentence is copied
     from the transcript, this mode **cannot fabricate content**.
   - **`llm`** — fluent prose notes from a **local Meta Llama-3.3-70B-Instruct**
     (4-bit quantized via `bitsandbytes`), run with Hugging Face Transformers.
     No API; the transcript never leaves the machine.
   - **`both`** — runs both and writes both note sets.
3. **Output** — `transcript.txt`, `timestamped_transcript.txt`, `notes_*.md`,
   `notes_*.docx`, and the raw Whisper `raw_output.json`.

> **Why local?** Audio of privileged legal proceedings is highly confidential.
> Running both models on-device avoids sending any audio or transcript to a
> third-party service.

---

## 1. Requirements

- **Python 3.10+**
- **ffmpeg** on your `PATH` (Whisper needs it to decode audio)
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg` (or download a static build)
- For the **`llm`/`both`** modes: an **NVIDIA GPU**.
  Llama-3.3-70B in 4-bit needs roughly **~40 GB VRAM** (fits a 48 GB card such
  as an RTX A6000). The **`extractive`** mode runs fine on CPU.

---

## 2. Setup

```bash
# from the project folder
python -m venv .venv

# activate it
source .venv/bin/activate         # Linux / macOS
# .venv\Scripts\activate          # Windows (PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

### Zero-config quick start (no audio file needed)

Generate a short spoken sample with offline text-to-speech, then run the
pipeline on it. This proves the whole chain (Whisper → notes → `.docx`) works
on your machine with no GPU and no API:

```bash
python make_sample.py                 # writes sample.wav (offline TTS)
python main.py --audio sample.wav     # extractive notes -> output/
```

On Windows this uses the built-in System.Speech voice (no install). On
Linux/macOS, `pip install pyttsx3` first, or just supply your own audio.

### Access to Llama-3.3-70B (only for llm/both modes)

Llama-3.3-70B-Instruct is a **gated** model on Hugging Face:

1. Open the model page and accept the license:
   `https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct`
2. Authenticate locally once:
   ```bash
   huggingface-cli login
   ```

No GPU or no access yet? Test the LLM path with a small open model. The loader
**auto-detects your hardware**: with a CUDA GPU + `bitsandbytes` it uses the
paper's 4-bit path; on a CPU-only machine (e.g. Windows without CUDA) it falls
back to plain float32 automatically — no crash, just slower, so use a small
model:

```bash
# Runs Llama on CPU (Windows/macOS/Linux). Small model recommended.
python main.py --audio sample.wav --mode llm \
  --llm-model meta-llama/Llama-3.2-1B-Instruct
```

> **Windows + Llama:** the 70B model needs a 48 GB-class GPU and will not run
> on a typical laptop. On a CPU-only Windows machine, stick to a small
> `--llm-model` (1B–3B), or use the default `extractive` mode, which needs no
> GPU at all. `bitsandbytes` 4-bit is Linux/CUDA-only; the loader skips it
> automatically elsewhere.

---

## 3. Usage

```bash
# Extractive (default) — fast, CPU-friendly, no LLM
python main.py --audio meeting.m4a

# Local Llama notes only
python main.py --audio hearing.wav --mode llm --whisper-model medium

# Both modes, larger transcription model
python main.py --audio deposition.mp3 --mode both --whisper-model medium
```

Key flags (`python main.py --help` for all):

| Flag | Default | Meaning |
|------|---------|---------|
| `--audio` | *(required)* | input audio file |
| `--mode` | `extractive` | `extractive` \| `llm` \| `both` |
| `--whisper-model` | `base` | `tiny`/`base`/`small`/`medium`/`large`/`turbo` |
| `--llm-model` | `meta-llama/Llama-3.3-70B-Instruct` | HF model id |
| `--language` | auto | force a language, e.g. `en` |
| `--output-dir` | `output` | where files are written |
| `--no-4bit` | off | disable 4-bit (needs much more VRAM) |
| `--no-docx` | off | skip the `.docx` export |

---

## 4. Running in VS Code

1. **Open the folder**: `File → Open Folder…` → select this project.
2. **Select the interpreter**: `Ctrl/Cmd+Shift+P` → *Python: Select
   Interpreter* → choose `.venv`.
3. **Run / debug**: open `main.py`, press **F5**, and pick a configuration
   (*"Audio-to-Notes: extractive"* or *"… local Llama (both)"*).
   The configs live in `.vscode/launch.json`; edit the `--audio` path to your
   file (they point at `sample.wav` by default).
4. **Or use the terminal** inside VS Code with the commands in Section 3.
5. **Run tests**: the Testing panel is pre-configured for `pytest`, or run
   `pytest` in the terminal.

---

## 5. Project layout

```
audio_to_notes_project/
├── main.py                 # CLI entry point
├── make_sample.py          # offline TTS -> sample.wav (zero-config demo)
├── requirements.txt
├── README.md
├── a2n/
│   ├── transcribe.py       # local Whisper
│   ├── extractive.py       # extractive + rule-based notes (pure Python)
│   ├── llm_notes.py        # local Llama-3.3-70B notes
│   ├── docx_export.py      # markdown -> .docx
│   └── io_utils.py
├── tests/
│   └── test_extractive.py  # runs without any model
└── .vscode/                # F5 launch + interpreter/test settings
```

---

## 6. Limitations (see the paper)

- **Not evaluated on legal audio.** No accuracy numbers are claimed.
- ASR errors on legal terms, names, and citations propagate downstream.
- The `llm` mode is private (local) but still **generative**, so it can
  hallucinate or omit; review its output. The `extractive` mode is the
  faithful, audit-friendly option for substantive work.
