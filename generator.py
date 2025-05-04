import asyncio
import numpy as np
from textwrap import dedent
from language_models import LlamaModel, OpenAIModel
from utils import retry_if_unsuccessful, Constants
import configs

from rich import print as rprint

class Generation:
    INVALID_PROMPT = dedent("""
    Consider the following question and answer:

    <Question>
    {question}
    </Question>

    <Response>
    {response}
    </Response>
    
    Does the response express incapability to answer the question? Return "yes" or "no", with no additional text.
    """)
    PROMPT_PREFIX = "Answer the following question in plain text, without any additional formatting:\n\n{prompt}"
    REFUSAL_PHRASES = [
        "i couldn't find",
        "i am unable to",
        "i'm unable to",
        "i cannot confirm",
        "i do not have",
        "i don't have",
        "i apologize",
        "i don't know",
        "i'm not sure",
        "i'm sorry",
        "i'm not familiar",
        "unfortunately",
    ]

    def __init__(self, model):
        self.model = model

    def ignore_generation(self, text):
        if any([phrase.lower() in text.lower() for phrase in self.REFUSAL_PHRASES]):
            return True
        return False

    # async def generate_diverse(self, prompt, num_gens=5):
    #     generations = []
    #     params = configs.generate_diverse_params[self.model.name]
    #     for gen in asyncio.as_completed([self._generate(prompt, params) for _ in range(num_gens)]):
    #         try:
    #             generation = await gen
    #             generations.append(generation)
    #         except AssertionError:
    #             continue
            
    #     if len(generations) == 0:
    #         return Constants.INVALID_GENERATION

    #     return generations

    # async def generate_most_likely(self, prompt):
    #     params = {"temperature": 0.0, "max_tokens": 512}
    #     try:
    #         generation = await self._generate(prompt, params)
    #         return generation
    #     except AssertionError:
    #         return Constants.INVALID_GENERATION

    async def generate_diverse(self, prompt, num_gens=5):
        generations = []
        params = configs.generate_diverse_params[self.model.name]
        for gen in asyncio.as_completed([self._generate(prompt, params) for _ in range(num_gens)]):
            generation = await gen
            generations.append(generation)

        if any([generation == Constants.INVALID_GENERATION for generation in generations]):
            return Constants.INVALID_GENERATION
        else:
            return generations

    async def generate_most_likely(self, prompt):
        generation = await self._generate(prompt, {"temperature": 0.0, "max_tokens": configs.max_tokens})
        return generation
            
    async def _generate(self, prompt, params):
        max_retries = 3
        for _ in range(max_retries):
            prompt = self.PROMPT_PREFIX.format(prompt=prompt)
            generation = await self.model.query(prompt, **params)

            is_invalid = await self.is_invalid_generation(prompt, generation)
            if is_invalid:
                continue
            else:
                return generation

        return Constants.INVALID_GENERATION

    async def is_invalid_generation(self, prompt, generation):
        prompt = self.INVALID_PROMPT.format(question=prompt, response=generation)
        response = await self.model.query(prompt, **{"temperature": 0.0, "max_tokens": 10})

        if "yes" in response.lower():
            return True
        else:
            return False
