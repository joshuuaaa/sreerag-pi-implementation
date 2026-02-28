# src/llm/engine.py
"""
LLM engine using llama-cpp-python with Phi-3 Mini chat template.

Phi-3 Mini prompt format
────────────────────────
<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n

Stop tokens: <|end|>, <|user|>, <|system|>
"""

import logging
import os
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    _LLAMA_OK = True
except ImportError:
    logger.warning("llama-cpp-python not installed – LLM engine disabled")
    _LLAMA_OK = False
    Llama = None


# ── Phi-3 stop / special tokens ───────────────────────────────────────────────
_PHI3_STOP = [
    "<|end|>", "<|user|>", "<|system|>", "<|endoftext|>",
    # Prevent the model from roleplaying user responses
    "\nUser:", "\nuser:", "\nHuman:", "\nhuman:",
    "\nPerson:", "\nperson:", "\nSupport:", "\nsupport:",
    "\nPatient:", "\npatient:", "\nYou:",
]


class LLMEngine:
    """
    Wraps a Phi-3 Mini GGUF model via llama-cpp-python.

    Provides two calling conventions:
    - ``generate_chat(system, messages)``  – structured chat interface (preferred)
    - ``generate(prompt)``                 – raw prompt passthrough (legacy)
    """

    def __init__(self, config: dict):
        self.model_path  = os.path.expanduser(config.get("model_path", ""))
        self.n_ctx       = config.get("n_ctx", 4096)
        self.n_threads   = config.get("n_threads", 4)
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens  = config.get("max_tokens", 150)
        # Use GPU layers only if specified (default 0 = CPU only)
        self.n_gpu_layers = config.get("n_gpu_layers", 0)

        self.llm: Optional[Llama] = None

        if not _LLAMA_OK:
            logger.error("llama-cpp-python unavailable – LLM disabled")
            return

        if not os.path.exists(self.model_path):
            logger.error("LLM model file not found: %s", self.model_path)
            return

        try:
            logger.info("Loading LLM: %s", self.model_path)
            logger.info("This may take 1-2 minutes on first load…")
            self.llm = Llama(
                model_path    = self.model_path,
                n_ctx         = self.n_ctx,
                n_threads     = self.n_threads,
                n_gpu_layers  = self.n_gpu_layers,
                verbose       = False,
            )
            logger.info("✅ LLM engine ready (Phi-3 Mini, ctx=%d)", self.n_ctx)
        except Exception as e:
            logger.error("LLM init failed: %s", e)
            self.llm = None

    # ── public: chat interface ────────────────────────────────────────────────

    def generate_chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response using the Phi-3 chat template.

        Args:
            system_prompt: Content to inject as <|system|> turn.
            messages:      List of ``{"role": "user"|"assistant", "content": …}``
                           in chronological order.
            max_tokens:    Override default generation length.

        Returns:
            Model response as a plain string.
        """
        if not self.llm:
            return self._fallback()

        prompt = self._build_phi3_prompt(system_prompt, messages)
        return self._run(prompt, max_tokens)

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Legacy raw-prompt generation.  Kept for backward compatibility with
        the orchestrator's ``_generate_response`` method.

        Args:
            prompt:     Fully-assembled prompt string (any format).
            max_tokens: Override default generation length.

        Returns:
            Model response as a plain string.
        """
        if not self.llm:
            return self._fallback()
        return self._run(prompt, max_tokens)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_phi3_prompt(
        system_prompt: str,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Assemble the canonical Phi-3 Mini chat prompt.

        Template:
            <|system|>\\n{system}<|end|>\\n
            <|user|>\\n{user_1}<|end|>\\n<|assistant|>\\n{assistant_1}<|end|>\\n
            …
            <|user|>\\n{user_N}<|end|>\\n<|assistant|>\\n
        """
        parts: List[str] = []

        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt.strip()}<|end|>\n")

        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "user":
                parts.append(f"<|user|>\n{content}<|end|>\n<|assistant|>\n")
            elif role == "assistant":
                parts.append(f"{content}<|end|>\n")

        # Ensure the prompt ends with the assistant turn-start token
        if not parts[-1].endswith("<|assistant|>\n"):
            parts.append("<|assistant|>\n")

        return "".join(parts)

    def _run(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Execute inference and clean up the output."""
        tok_limit = max_tokens if max_tokens is not None else self.max_tokens
        try:
            out = self.llm(
                prompt,
                max_tokens  = tok_limit,
                temperature = self.temperature,
                stop        = _PHI3_STOP,
                echo        = False,
            )
            text = out["choices"][0]["text"]
            return self._clean(text)
        except Exception as e:
            logger.error("LLM generation error: %s", e)
            return self._fallback()

    @staticmethod
    def _clean(text: str) -> str:
        """Strip Phi-3 artefacts and normalise whitespace."""
        # Remove any leaked special tokens
        for tok in _PHI3_STOP:
            text = text.replace(tok, "")
        # Cut off anything that looks like a simulated user/patient reply
        # e.g. "support: Yes...", "User: I'm doing it", "Patient: okay"
        text = re.split(
            r"(?i)\b(user|human|person|patient|support|you)\s*:",
            text,
        )[0]
        # Collapse multiple newlines / spaces
        text = re.sub(r"\n{2,}", " ", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    @staticmethod
    def _fallback() -> str:
        return "I'm here to help. Please tell me what's happening."
