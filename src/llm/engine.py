# src/llm/engine.py
"""
LLM engine using llama-cpp-python
"""

import os
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:
    print("⚠️ llama-cpp-python not installed")
    Llama = None

class LLMEngine:
    def __init__(self, config: dict):
        """
        Initialize LLM engine
        
        Args:
            config: LLM configuration
        """
        self.model_path = os.path.expanduser(config.get("model_path", ""))
        self.n_ctx = config.get("n_ctx", 2048)
        self.n_threads = config.get("n_threads", 4)
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 150)
        
        self.llm: Optional[Llama] = None
        
        if not Llama:
            print("❌ LLM engine disabled - llama-cpp-python not available")
            return
            
        if not os.path.exists(self.model_path):
            print(f"❌ LLM model not found: {self.model_path}")
            return
            
        try:
            print(f"🔄 Loading LLM model: {self.model_path}")
            print("   This may take 1-2 minutes...")
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            
            print("✅ LLM engine initialized")
            
        except Exception as e:
            print(f"❌ LLM initialization error: {e}")
            self.llm = None
            
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not self.llm:
            return "I'm here to help. Please tell me more about the situation."
            
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                stop=["USER:", "ASSISTANT:", "\n\n"],
                echo=False
            )
            
            text = response["choices"][0]["text"].strip()
            return text
            
        except Exception as e:
            print(f"❌ LLM generation error: {e}")
            return "I understand. Let me help you with that."
