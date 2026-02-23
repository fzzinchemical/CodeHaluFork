import re
import random
import subprocess
import shlex

import numpy as np

import torch
from typing import Optional


def set_random_seed(seed: Optional[int]):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class StopAtSpecificTokenCriteria(torch.nn.Module):
    # kept minimal compatibility shim if needed elsewhere
    def __init__(self, token_id_list=None):
        super().__init__()
        self.token_id_list = token_id_list or []

    def __call__(self, input_ids, scores, **kwargs):
        return self.forward(input_ids, scores, **kwargs)


class LLMModel:
    """Lightweight LLM wrapper that uses an execution backend (default: `ollama`).

    Parameters
    - name: logical name
    - model_id: backend model identifier (for ollama this is the ollama model name)
    - backend: 'ollama' or 'hf' (hf kept for compatibility but default is 'ollama')
    - prompt_template / extract_pattern as before
    """

    def __init__(self,
                 name: str,
                 model_id: str,
                 tokenizer_id: Optional[str] = None,
                 access_token: Optional[str] = None,
                 prompt_template: Optional[str] = None,
                 extract_pattern: Optional[str] = None,
                 backend: str = 'ollama',
                 trust_remote_code: bool = True):
        self.name = name
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id or model_id
        self.access_token = access_token
        self.prompt_template = prompt_template
        self.extract_pattern = extract_pattern
        self.backend = backend
        self.trust_remote_code = trust_remote_code

        # HF-specific attributes (only used when backend=='hf')
        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.device = None

    def _ensure_loaded_hf(self):
        if self._loaded:
            return
        if self.access_token:
            try:
                from huggingface_hub import login as _login
                _login(self.access_token)
            except Exception:
                pass

        from transformers import AutoTokenizer as _AT, AutoModelForCausalLM as _AM
        self.tokenizer = _AT.from_pretrained(
            self.tokenizer_id,
            use_fast=True,
            trust_remote_code=self.trust_remote_code)
        self.model = _AM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=self.trust_remote_code,
            device_map='auto')
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self._loaded = True

    def _build_prompt(self, prompt: str) -> str:
        if self.prompt_template:
            try:
                return self.prompt_template.format(prompt=prompt)
            except Exception:
                return f"{self.prompt_template} {prompt}"
        return prompt

    def generate(self,
                 prompt: str,
                 temperature: float = 0.0,
                 max_new_tokens: int = 2048):
        full_prompt = self._build_prompt(prompt)
        if not full_prompt.endswith("\nANSWER:\n"):
            full_prompt = full_prompt + "\nANSWER:\n"

        if self.backend == 'ollama':
            # Use the ollama CLI to generate. This keeps the code dependency-free
            # for Ollama usage and works with local Ollama runtimes.
            cmd = f"ollama generate {shlex.quote(self.model_id)} --temperature {temperature} --max {int(max_new_tokens)} {shlex.quote(full_prompt)}"
            try:
                proc = subprocess.run(shlex.split(cmd),
                                      capture_output=True,
                                      text=True,
                                      check=False)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"ollama generate failed: {proc.stderr}")
                response = proc.stdout
                return response, full_prompt
            except FileNotFoundError:
                raise RuntimeError(
                    "ollama CLI not found; ensure Ollama is installed and on PATH"
                )

        # fallback to huggingface transformers if requested
        if self.backend == 'hf':
            self._ensure_loaded_hf()
            inputs = self.tokenizer(full_prompt,
                                    return_tensors="pt").to(self.device)
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id)
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output, skip_special_tokens=True)
            return response, full_prompt

        raise RuntimeError(f"Unsupported backend: {self.backend}")

    def extract_code(self, response: str) -> str:
        pattern = self.extract_pattern or r'```python(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1]
        matches = re.findall(r'```([\s\S]*?)```', response, re.DOTALL)
        if matches:
            return matches[-1]
        return response