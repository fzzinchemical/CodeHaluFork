import re
import random

import numpy as np

import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


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
    """Lightweight LLM wrapper that lazily loads HF models and provides uniform generate/extract.

    Instantiate module-level objects (e.g. `mixtral_8x_7B`) instead of defining a class
    per-model.
    """

    def __init__(self,
                 name: str,
                 model_id: str,
                 tokenizer_id: Optional[str] = None,
                 access_token: Optional[str] = None,
                 prompt_template: Optional[str] = None,
                 extract_pattern: Optional[str] = None,
                 trust_remote_code: bool = True):
        self.name = name
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id or model_id
        self.access_token = access_token
        self.prompt_template = prompt_template
        self.extract_pattern = extract_pattern
        self.trust_remote_code = trust_remote_code

        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.device = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.access_token:
            try:
                login(self.access_token)
            except Exception:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            use_fast=True,
            trust_remote_code=self.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
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
            # some HF models manage device_map themselves
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
        self._ensure_loaded()
        full_prompt = self._build_prompt(prompt)
        # Keep a trailing ANSWER marker for consistency with previous code
        if not full_prompt.endswith("\nANSWER:\n"):
            full_prompt = full_prompt + "\nANSWER:\n"

        inputs = self.tokenizer(full_prompt,
                                return_tensors="pt").to(self.device)
        try:
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
        except Exception as e:
            raise

    def extract_code(self, response: str) -> str:
        pattern = self.extract_pattern or r'```python(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1]
        # fallback: any fenced block
        matches = re.findall(r'```([\s\S]*?)```', response, re.DOTALL)
        if matches:
            return matches[-1]
        return response


# Module-level preconfigured model instances. Import these rather than instantiating
# classes for each model. These are lazily loaded when `generate()` is called.

mixtral_8x_7B = LLMModel(name='mixtral_8x_7B',
                         model_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
                         access_token='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',
                         prompt_template='<s>[INST] {prompt} [/INST] ',
                         extract_pattern=r'```python(.*?)```')

mixtral_7B = LLMModel(name='mixtral_7B',
                      model_id='mistralai/Mistral-7B-Instruct-v0.2',
                      access_token='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',
                      prompt_template='<s>[INST] {prompt} [/INST] ',
                      extract_pattern=r'```python(.*?)```')

gemma_7b = LLMModel(name='gemma_7b',
                    model_id='google/gemma-7b',
                    access_token='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',
                    prompt_template='{prompt}',
                    extract_pattern=r'ANSWER:(.*?)<eos>')

codellama_7b = LLMModel(name='codellama_7b',
                        model_id='codellama/CodeLlama-7b-Instruct-hf',
                        prompt_template='<s>[INST] {prompt} [/INST]',
                        extract_pattern=r'```([\s\S]*?)```')

deepseek_coder_6_7b = LLMModel(
    name='deepseek_coder_6_7b',
    model_id='deepseek-ai/deepseek-coder-6.7b-instruct',
    prompt_template=
    '''You are an AI programming assistant, utilizing the DeepSeek Coder model.\n### Instruction:\n{prompt}\n### Response:\n''',
    extract_pattern=r'### Response:.*?```python(.*?)```')

llama2 = LLMModel(name='llama2',
                  model_id='meta-llama/Llama-2-7b-chat-hf',
                  prompt_template='<s>[INST] {prompt} [/INST] ',
                  extract_pattern=r'```([\s\S]*?)```')

llama3 = LLMModel(name='llama3',
                  model_id='meta-llama/Meta-Llama-3-8B-Instruct',
                  prompt_template='{prompt}',
                  extract_pattern=r'```python(.*?)```')
