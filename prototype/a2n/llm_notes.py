"""
Local generative note-taking with Meta Llama-3.3-70B-Instruct.

This is the "local-generative mode" from the paper: an open-weight,
instruction-tuned LLM run on-device via Hugging Face Transformers -- NOT
through any hosted API. Keeping the model local removes the confidentiality
risk of sending privileged audio/transcripts to a third party. It does NOT
remove the hallucination risk, so the prompt explicitly forbids inventing
facts and the output should still be reviewed by a human.

Hardware note
-------------
Llama-3.3-70B-Instruct does not fit on a single ~48 GB GPU in fp16
(~140 GB needed). It is therefore loaded with 4-bit weight quantization
(bitsandbytes, NF4), which brings it to roughly ~35-40 GB and fits a single
48 GB-class card (e.g. RTX A6000). Set load_in_4bit=False only if you have
enough VRAM (multi-GPU / H100-class) for higher precision.

Access note
-----------
Llama-3.3-70B-Instruct is a *gated* model on Hugging Face. You must (1) accept
the license on the model page and (2) authenticate locally once with:
    huggingface-cli login
For quick testing without GPU/gated access, pass a smaller open model via
--llm-model, e.g. "meta-llama/Llama-3.2-3B-Instruct" or
"Qwen/Qwen2.5-7B-Instruct".
"""

from __future__ import annotations

from typing import Tuple

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

SYSTEM_PROMPT = (
    "You are a careful documentation assistant for legal and professional "
    "settings. Convert the provided transcript into clean, well-structured "
    "notes using clear headings, numbering, and bullet points. Group content "
    "into sensible sections (e.g. Summary, Key Points, Decisions, Action "
    "Items). Use ONLY information that appears in the transcript: do not "
    "invent facts, names, dates, figures, or citations. If something is "
    "ambiguous or missing, say so explicitly rather than guessing."
)


def _bitsandbytes_available() -> bool:
    """True only if bitsandbytes can actually be imported (Linux/CUDA)."""
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def load_llm(model_id: str = DEFAULT_MODEL, load_in_4bit: bool = True) -> Tuple:
    """
    Load tokenizer and model locally, adapting to the available hardware.

    Returns (tokenizer, model).

    Three paths, chosen automatically:
      * GPU + 4-bit  : the paper's setup -- Llama-3.3-70B in NF4 on a ~48 GB
                       card (needs CUDA *and* bitsandbytes). This is the
                       default when a CUDA GPU is present.
      * GPU, no 4-bit: full/half precision on GPU (needs lots of VRAM).
      * CPU          : no quantization, float32, runs anywhere. 70B will not
                       fit in CPU RAM -- use a small --llm-model here
                       (e.g. meta-llama/Llama-3.2-1B-Instruct).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    has_cuda = torch.cuda.is_available()

    # 4-bit only makes sense on CUDA with a working bitsandbytes install.
    use_4bit = load_in_4bit and has_cuda and _bitsandbytes_available()
    if load_in_4bit and not use_4bit:
        if not has_cuda:
            print(
                "       [info] No CUDA GPU detected -- loading the LLM on CPU in "
                "float32 (no 4-bit). This is slow; use a small --llm-model such "
                "as meta-llama/Llama-3.2-1B-Instruct."
            )
        else:
            print(
                "       [info] bitsandbytes unavailable (it is Linux/CUDA-only) -- "
                "loading without 4-bit. Ensure you have enough VRAM, or use WSL2."
            )

    quant_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model_kwargs = {"device_map": "auto" if has_cuda else None}
    if use_4bit:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["torch_dtype"] = torch.float16
    elif has_cuda:
        model_kwargs["torch_dtype"] = torch.float16
    else:
        # CPU: float16 is unsupported/slow for matmul; use float32.
        model_kwargs["torch_dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not has_cuda:
        model = model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_notes_llm(
    transcript: str,
    tokenizer,
    model,
    max_new_tokens: int = 1200,
) -> str:
    """Generate prose notes from a transcript using a loaded local LLM."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Turn the following transcript into clean, structured notes.\n\n"
                f"TRANSCRIPT:\n{transcript}"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy -> deterministic, lower fabrication
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def notes_to_markdown(title: str, notes_body: str, audio_file: str, model_id: str) -> str:
    """Wrap the LLM output with a small metadata header."""
    from datetime import datetime

    header = (
        f"# {title}\n\n"
        f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Audio file:** {audio_file}\n"
        f"**Note model (local):** {model_id}\n"
        "**Note mode:** local generative (no API)\n\n"
        "> Generated by a local LLM. Review for accuracy before relying on it; "
        "generative models can omit or misstate details.\n\n"
        "---\n\n"
    )
    return header + notes_body + "\n"
