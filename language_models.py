import json
import os
import time
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
from openai import AsyncOpenAI, BadRequestError, APIError, APIConnectionError, APITimeoutError, RateLimitError
from together import AsyncTogether, Together
import nltk
from nltk.tokenize import sent_tokenize
import asyncio
import functools

import configs
from rich import print as rprint
import sys

class NLIModel(object):
    LABEL_MAPPING = {'contradict': 0, 'entail': 1, 'neutral': 2, 'invalid': 9}

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(configs.nli_model).to("cuda:0")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(configs.nli_model)

    @torch.no_grad()
    def evaluate(self, claim_set1, claim_set2, batch_size=128):
        sents2eval = [(s1, s2) for s1 in claim_set1 for s2 in claim_set2]
        n_rows = len(sents2eval)

        labels = np.full(n_rows, fill_value=self.LABEL_MAPPING['invalid'], dtype=int)
        probs = np.zeros((n_rows, 3), dtype=float)
        start_index = 0
        for stop_index in range(batch_size, n_rows+batch_size, batch_size):
            stop_index = min(stop_index, n_rows)
            s = sents2eval[start_index:stop_index]

            inputs = self.tokenizer(s, padding=True, return_tensors="pt").to("cuda:0")
            outputs = self.model(**inputs).logits
            output_probs = F.softmax(outputs, dim=1).cpu().numpy()
            output_labels = outputs.argmax(dim=1).cpu().numpy()

            labels[start_index:stop_index] = output_labels
            probs[start_index:stop_index, :] = output_probs

            start_index += batch_size

        labels = labels.reshape((len(claim_set1), -1))
        probs = probs.reshape((len(claim_set1), len(claim_set2), 3))

        return labels, probs

    def compute_claim_entail(self, claim_set1, claim_set2):
        labels, probs = self.evaluate(claim_set1, claim_set2)
        is_entail = (labels == self.LABEL_MAPPING['entail'])

        return is_entail

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # Time between requests in seconds
        self.last_request_time = None
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            
            if self.last_request_time is None:
                self.last_request_time = now
                return

            time_since_last = now - self.last_request_time
            
            if time_since_last < self.interval:
                wait_time = self.interval - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.monotonic()

class LanguageModel(object):
    def __init__(self, model=None, api_key=None, base_url=None, requests_per_minute=300):
        self.model_name = model
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)

        if base_url and api_key:
            self.openai_api = AsyncOpenAI(api_key=api_key,
                                          base_url=base_url)
        else:
            self.openai_api = AsyncOpenAI(api_key=configs.openai_api_key)

    async def query(self, prompt, **params):
        await self.rate_limiter.acquire()
        response = await self.openai_api.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params
        )

        return response

    async def query_logprobs(self, prompt, **params):
        if len(params) == 0:
            params = configs.generate_diverse_params[self.model_name]

        params.update({"logprobs": True})
        response = await self.openai_api.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params
        )

        text = response.choices[0].message.content.strip()
        tokens = []
        logprobs = []
        for c in response.choices[0].logprobs.content:
            tokens.append(c.token)
            logprobs.append(c.logprob)

        return text, tokens, logprobs

class OpenAIModel:
    AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo"]

    def __init__(self, 
                model: str, 
                api_key: str=None,
                base_url: str=None,
                requests_per_minute: int=300):
        # super().__init__(model=model, api_key=api_key, base_url=base_url)

        self.model_name = None
        for name in self.AVAILABLE_MODELS:
            if model.lower() in name.lower():
                self.model_name = name
                break
        if self.model_name is None:
            raise NotImplementedError(f"{model} is not supported. Please choose from {self.AVAILABLE_MODELS}")
        
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)

        if base_url and api_key:
            self.openai_api = AsyncOpenAI(api_key=api_key,
                                          base_url=base_url,
                                          timeout=120)
        else:
            self.openai_api = AsyncOpenAI(api_key=configs.openai_api_key,
                                          timeout=120)

    @property
    def api_name(self):
        return "openai"

    @property
    def name(self):
        return self.model_name

    async def query(self, prompt, **params):
        await self.rate_limiter.acquire()
        response = await self.openai_api.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        return response.choices[0].message.content.strip()

    async def query_logprobs(self, prompt, **params):
        await self.rate_limiter.acquire()
        response = await self.openai_api.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            **params
        )

        text = response.choices[0].message.content.strip()
        tokens = []
        logprobs = []
        for c in response.choices[0].logprobs.content:
            tokens.append(c.token)
            logprobs.append(c.logprob)

        return text, tokens, logprobs

    async def query_structured(self, sys_prompt, usr_prompt, spec):
        await self.rate_limiter.acquire()
        response = await self.openai_api.responses.parse(
            model=self.model_name,
            input=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt}
            ],
            text_format=spec,
        )
        
        return response.output_parsed

class TogetherAIModel:
    AVAILABLE_MODELS = ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"]

    def __init__(self, 
                 model: str, 
                 api_key: str=configs.together_api_key,
                 requests_per_minute: int=300):

        self.model_name = None
        for name in self.AVAILABLE_MODELS:
            if model.lower() in name.lower():
                self.model_name = name
                break
        if self.model_name is None:
            raise NotImplementedError(f"{model} is not supported. Please choose from {self.AVAILABLE_MODELS}")
        
        self.client = Together(api_key=api_key, timeout=120)
        self.async_client = AsyncTogether(api_key=api_key, timeout=300)
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)

    @property
    def api_name(self):
        return "togetherai_DEPRECATED"
    
    @property
    def name(self):
        return self.model_name

    def query_stream(self, prompt, **params):
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            logprobs=True,
            **params
        )

        text = ""
        tokens = []
        logprobs = []
        for chunk in stream:
            text += chunk.choices[0].delta.content
            tokens.append(chunk.choices[0].delta.content)
            if logprobs is not None:
                logprobs.append(chunk.choices[0].logprobs)

        return text, tokens, logprobs

    async def query(self, prompt, **params):
        await self.rate_limiter.acquire()
        response = await self.async_client.chat.completions.create(
            # model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params
        )

        return response.choices[0].message.content.strip()

    async def query_logprobs(self, prompt, **params):
        await self.rate_limiter.acquire()
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            **params
        )

        text = response.choices[0].message.content.strip()
        tokens = response.choices[0].logprobs.tokens
        logprobs = response.choices[0].logprobs.token_logprobs

        if len(tokens) != len(logprobs):
            print(f"WARNING: Number of tokens and logprobs do not match for model {self.model_name}, prompt: {prompt}")

        return text, tokens, logprobs

    async def query_structured(self, sys_prompt, usr_prompt, spec):
        await self.rate_limiter.acquire()
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt}
            ],
            response_format={
                    "type": "json_object",
                    "schema": spec.model_json_schema()
            },
        )

        return spec.model_validate_json(response.choices[0].message.content)

class LlamaModel(LanguageModel):
    def __init__(self,
                 model: str="meta-llama/llama-3-70b-instruct", 
                 api_key: str=configs.novita_api_key, 
                 base_url: str=configs.novita_base_url):
        super().__init__(model=model, api_key=api_key, base_url=base_url)

    async def query(self, prompt, **params):
        if len(params) == 0:
            params = configs.generate_diverse_params[self.model_name]
            
        response = await super().query(prompt, **params)
        return response.choices[0].message.content.strip()

    def test_contain_logprobs(self, prompt):
        from openai import OpenAI
        openai_api = OpenAI(api_key=configs.novita_api_key,
                            base_url=configs.novita_base_url)
        # for testing if the api returns logprobs
        params = configs.generate_diverse_params["meta-llama/llama-3-70b-instruct"]
        
        params.update({"logprobs": True})
        response = openai_api.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )

        tokens = []
        logprobs = []
        for c in response.choices[0].logprobs.content:
            tokens.append(c.token)
            logprobs.append(c.logprob)

        return tokens, logprobs
