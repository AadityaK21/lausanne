"""
Multi-turn conversational assistant — fully local, no external API.

Companion to the Audio-to-Notes pipeline: once a hearing has been transcribed,
you often want to *ask questions about it* rather than read the whole
transcript. This module provides that, under the same constraint as the rest of
the project — the model runs on-device, so no transcript content is ever
transmitted.

The engineering problem in a multi-turn chatbot is not calling the model. It is
**context management**: a conversation grows without bound, but the model's
context window does not. Three things have to be handled deliberately:

1. **Token budgeting.** History must be trimmed to fit, and trimming has to
   drop the right things.
2. **System-prompt pinning.** The instruction that defines behaviour must never
   be evicted, or the assistant silently changes personality mid-conversation.
3. **Turn-pair integrity.** Dropping a user message but keeping its assistant
   reply leaves the model reading an answer to a question it cannot see.
   Messages are evicted in whole user/assistant pairs.

The `Conversation` class below is pure Python with no ML dependency, so all of
that logic is unit-testable without loading a model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful, concise assistant. Answer using only what you are told "
    "or asked. If you do not know something, say so plainly rather than "
    "guessing. Keep answers short unless asked to expand."
)

# Rough chars-per-token for English. Deliberately conservative: over-estimating
# token count causes harmless early trimming, under-estimating causes a hard
# failure when the model rejects the prompt.
CHARS_PER_TOKEN = 3.6


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate that needs no tokenizer."""
    return max(1, int(len(text) / CHARS_PER_TOKEN) + 1)


@dataclass
class Conversation:
    """
    Ordered message history with a bounded token budget.

    Pure data structure — no model, no I/O. That separation is what lets the
    trimming rules be tested exhaustively in milliseconds.
    """
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_context_tokens: int = 3072
    reserve_for_reply: int = 512
    messages: List[Message] = field(default_factory=list)

    # -- mutation --------------------------------------------------------
    def add_user(self, content: str) -> None:
        self.messages.append(Message("user", content))

    def add_assistant(self, content: str) -> None:
        self.messages.append(Message("assistant", content))

    def reset(self) -> None:
        self.messages.clear()

    # -- accounting ------------------------------------------------------
    @property
    def budget(self) -> int:
        """Tokens available for history, after the system prompt and the reply."""
        return self.max_context_tokens - self.reserve_for_reply - estimate_tokens(self.system_prompt)

    def token_count(self) -> int:
        return estimate_tokens(self.system_prompt) + sum(
            estimate_tokens(m.content) for m in self.messages
        )

    # -- the interesting part --------------------------------------------
    def build_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return the message list to send to the model, trimmed to fit.

        Strategy: keep the system prompt always, then keep the most recent
        turns that fit, walking backwards. Eviction happens in whole
        user/assistant pairs so the model never sees an orphaned reply.

        Recency-first is the right default for open conversation — the last
        few turns carry most of the relevant context. A summarisation-based
        approach (compress old turns rather than drop them) would retain more,
        and is noted as future work; it costs an extra model call per trim.
        """
        kept: List[Message] = []
        used = 0

        # Walk backwards in pairs.
        i = len(self.messages) - 1
        while i >= 0:
            # Identify the block ending at i: an assistant reply plus the user
            # message that prompted it, where such a pair exists.
            if (
                self.messages[i].role == "assistant"
                and i > 0
                and self.messages[i - 1].role == "user"
            ):
                block = [self.messages[i - 1], self.messages[i]]
                step = 2
            else:
                block = [self.messages[i]]
                step = 1

            cost = sum(estimate_tokens(m.content) for m in block)
            if used + cost > self.budget and kept:
                break                      # budget exhausted; stop trimming back
            used += cost
            kept = block + kept
            i -= step

        return [Message("system", self.system_prompt).to_dict()] + [m.to_dict() for m in kept]

    def was_trimmed(self) -> bool:
        # -1 for the system message, which is not part of `self.messages`.
        return len(self.build_prompt_messages()) - 1 < len(self.messages)

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "system_prompt": self.system_prompt,
            "max_context_tokens": self.max_context_tokens,
            "reserve_for_reply": self.reserve_for_reply,
            "messages": [m.to_dict() for m in self.messages],
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Conversation":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        conv = cls(
            system_prompt=data.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            max_context_tokens=data.get("max_context_tokens", 3072),
            reserve_for_reply=data.get("reserve_for_reply", 512),
        )
        conv.messages = [Message(m["role"], m["content"]) for m in data.get("messages", [])]
        return conv


# --------------------------------------------------------------------------
# Model backend
# --------------------------------------------------------------------------
class LocalChatModel:
    """
    Thin wrapper over a local Hugging Face causal LM.

    Loads lazily and adapts to the hardware, exactly like `llm_notes.py`:
    CUDA plus bitsandbytes gets the 4-bit path; anything else falls back to
    float32 on CPU so the program runs (slowly) rather than crashing.
    """

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
                 use_4bit: bool = True, max_new_tokens: int = 400,
                 temperature: float = 0.7):
        self.model_id = model_id
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {}
        if self.use_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
                print(f"[chatbot] loading {self.model_id} in 4-bit on GPU")
            except ImportError:
                print("[chatbot] bitsandbytes unavailable; loading in fp16")
                kwargs["torch_dtype"] = torch.float16
                kwargs["device_map"] = "auto"
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            print(f"[chatbot] loading {self.model_id} in fp16 on GPU")
        else:
            print(f"[chatbot] no CUDA — loading {self.model_id} on CPU (slow)")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def reply(self, messages: List[Dict[str, str]]) -> str:
        self._load()
        import torch

        # Every instruct model has its own turn-delimiter format. Using the
        # tokenizer's own chat template rather than hand-rolling the string is
        # what makes this portable across models.
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Slice off the prompt so only the newly generated tokens are decoded.
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class EchoModel:
    """Deterministic stand-in used by the tests and by `--dry-run`."""

    def reply(self, messages: List[Dict[str, str]]) -> str:
        last = next((m for m in reversed(messages) if m["role"] == "user"), None)
        turns = sum(1 for m in messages if m["role"] == "user")
        return f"[echo | turn {turns}] {last['content'] if last else ''}"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
BANNER = """\
Local chatbot — nothing leaves this machine.
Commands:  /reset  clear history      /save <file>  write transcript
           /stats  context usage      /load <file>  restore transcript
           /exit   quit
"""


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Local multi-turn chatbot")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                   help="Hugging Face model id")
    p.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--max-context", type=int, default=3072)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--transcript", default=None,
                   help="prime the conversation with a transcript file")
    p.add_argument("--dry-run", action="store_true",
                   help="use the echo model — exercises the loop with no download")
    args = p.parse_args(argv)

    system = args.system
    if args.transcript:
        text = Path(args.transcript).read_text(encoding="utf-8")
        system = (
            f"{args.system}\n\nYou are answering questions about the following "
            f"transcript. Base every answer on it and say so when it does not "
            f"contain the answer.\n\n---\n{text}\n---"
        )
        print(f"Primed with {Path(args.transcript).name} "
              f"({estimate_tokens(text)} est. tokens)")

    conv = Conversation(system_prompt=system, max_context_tokens=args.max_context)
    model = EchoModel() if args.dry_run else LocalChatModel(
        args.model, use_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )

    print(BANNER)
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            conv.reset()
            print("(history cleared)\n")
            continue
        if user == "/stats":
            print(f"  turns in history : {len(conv.messages)}")
            print(f"  est. tokens used : {conv.token_count()} / {conv.max_context_tokens}")
            print(f"  history budget   : {conv.budget}")
            print(f"  trimming active  : {conv.was_trimmed()}\n")
            continue
        if user.startswith("/save "):
            conv.save(Path(user.split(" ", 1)[1].strip()))
            print("(saved)\n")
            continue
        if user.startswith("/load "):
            conv = Conversation.load(Path(user.split(" ", 1)[1].strip()))
            print(f"(loaded {len(conv.messages)} messages)\n")
            continue

        conv.add_user(user)
        prompt_messages = conv.build_prompt_messages()
        if conv.was_trimmed():
            print("  (older turns trimmed to fit the context window)")

        answer = model.reply(prompt_messages)
        conv.add_assistant(answer)
        print(f"bot > {answer}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
