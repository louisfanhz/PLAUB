import json
from json.decoder import JSONDecodeError
import asyncio
import os
import time
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Set, Dict, Any, Optional, Tuple

import configs
from language_models import LlamaModel, OpenAIModel
from evaluator import SelfConsistencyEvaluator
from prompts import responder_prompts, uncertainty_metrics_prompts
from result_formats import Claim
from utils import retry_if_unsuccessful
from rich import print as rprint
import sys

class Responder:
    def __init__(self, 
                 model,
                #  eval_model: str,
                 uncertainty_metrics: Any=None):
        self.model=model
        self.sc_evaluator = SelfConsistencyEvaluator(model=model)
        self.uncertainty_metrics = uncertainty_metrics
        # self.ref_evaluator = ref_evaluator = FactScoreEvaluator(
        #     ref_model="gpt-4o",
        #     db_path="./dataset/factscore/enwiki-20230401.db",
        #     retrieval_type="gtr"
        # )

    async def respond(self,
                    topic: str, 
                    topic_context: str, 
                    topic_generations: List[str],
                    all_claims: List[str], 
                    claim: str, 
                    claim_idx: int, 
                    question: str, 
                    n_res_per_question: int=3):
        prompt = responder_prompts["respond"].format(context=topic_context, question=question)
        answers, uncertainty_scores = await self._respond(topic, prompt, claim, n_res_per_question)
        consistency_with_claim = await self._evaluate_consistency_with_claim(claim, answers)
        self_consistency = await self._evaluate_self_consistency(answers)
        impact_tasks = [asyncio.create_task(self._measure_impact(context=all_claims, claim_idx=claim_idx, answer=answer)) for answer in answers]
        impact = await asyncio.gather(*impact_tasks)
        # correctnesses = self.ref_evaluator.evaluate_claims(topic=topic, claims=answers)

        # supported_scores = await self.sc_evaluator.evaluate_claims(topic_generations, answers)

        impact_tasks2 = [asyncio.create_task(self._measure_impact2(context=all_claims, claim_idx=claim_idx, answer=answer)) for answer in answers]
        impact2 = await asyncio.gather(*impact_tasks2)

        result = [
        {
            "text": answers[i],
            "correctness": None, #correctnesses[i],
            # "ans_supported_score": supported_scores[i],
            "consistency_with_claim": consistency_with_claim[i],
            "self_consistency": self_consistency[i],
            "impact": impact[i],
            "impact2": impact2[i],
            **uncertainty_scores[i],
        } for i in range(len(answers))]

        return result

    @retry_if_unsuccessful()
    async def _respond(self, topic: str, prompt: str, claim: str, n_res: int) -> Tuple[List[str], List[float]]:
        answers_texts = []
        answers_logprobs = []
        query_tasks = [asyncio.create_task(self.model.query_logprobs(prompt, **configs.generate_diverse_params[self.model.model_name])) 
                        for _ in range(n_res)]
        query_results = await asyncio.gather(*query_tasks)
        for query_result in query_results:
            text, _, logprobs = query_result
            answers_texts.append(text)
            answers_logprobs.append(logprobs)

        assert len(answers_texts) == n_res

        if self.uncertainty_metrics is not None:
            uncertainties = self.uncertainty_metrics(n_res=n_res, 
                                                    claim=claim,
                                                    log_probs=answers_logprobs,
                                                    texts=answers_texts)

        return answers_texts, uncertainties

    @retry_if_unsuccessful()
    async def _measure_impact(self, context: str, claim_idx: int, answer: str) -> float:
        if isinstance(context, list) and isinstance(context[0], str):
            ### implementation 1
            if claim_idx == len(context) - 1:
                return 0.0
            context = "\n".join(context[claim_idx+1:])

            ### implementation 2
            # context = "\n".join(context[claim_idx::-1])
        else:
            raise ValueError(f"Context should be a list of strings: {context}")

        prompt = responder_prompts["impact"].format(statement=answer, context=context)
        response = await self.model.query(prompt, temperature=0.0)

        # reverse the response to match the percentage number at the end
        percentage_reversed = re.search(r'%?(\d+)', response[::-1])
        percentage = int(percentage_reversed.group(1)[::-1])
        assert percentage is not None, f"No numerical value found in response: {response}"
        assert percentage <= 100, f"Impact percentage should be less than 100: {percentage}, response: {response}"
        assert percentage >= 0, f"Impact percentage should be greater than 0: {percentage}, response: {response}"
        
        return percentage / 100.0

    @retry_if_unsuccessful()
    async def _measure_impact2(self, context: str, claim_idx: int, answer: str) -> float:
        if isinstance(context, list) and isinstance(context[0], str):
            ### implementation 1
            # if claim_idx == len(context) - 1:
            #     return 0.0
            # context = "\n".join(context[claim_idx+1:])

            ### implementation 2
            context = "\n".join(context[claim_idx::-1])
        else:
            raise ValueError(f"Context should be a list of strings: {context}")

        prompt = responder_prompts["impact"].format(statement=answer, context=context)
        response = await self.model.query(prompt, temperature=0.0)

        # reverse the response to match the percentage number at the end
        percentage_reversed = re.search(r'%?(\d+)', response[::-1])
        percentage = int(percentage_reversed.group(1)[::-1])
        assert percentage is not None, f"No numerical value found in response: {response}"
        assert percentage <= 100, f"Impact percentage should be less than 100: {percentage}, response: {response}"
        assert percentage >= 0, f"Impact percentage should be greater than 0: {percentage}, response: {response}"
        
        return percentage / 100.0

    async def _evaluate_consistency_with_claim(self, claim: str, answers: List[str]) -> float:
        eval_tasks = [asyncio.create_task(self._evaluate_consistency(claim, answer)) for answer in answers]
        ratings = await asyncio.gather(*eval_tasks)

        # for r in ratings:
        #     assert len(r) == 1, f"Only one rating is expected"
        # return [r[0] for r in ratings]

        return ratings

    async def _evaluate_self_consistency(self, answers: List[str]) -> float:
        eval_tasks = []
        for i in range(len(answers)):
            other_answers = [answers[j] for j in range(len(answers)) if i != j]
            eval_tasks.append(asyncio.create_task(self._evaluate_consistency(answers[i], other_answers)))
        ratings = await asyncio.gather(*eval_tasks)

        n = len(answers) - 1
        avg_ratings = [np.mean(r).item() for r in ratings]

        return avg_ratings

    @retry_if_unsuccessful()
    async def _evaluate_consistency(self, test_claim, candidate_claims):
        n_claims = 1
        if isinstance(candidate_claims, list) and isinstance(candidate_claims[0], str):
            n_claims = len(candidate_claims)
            candidate_claims = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(candidate_claims)])

        sys_prompt = uncertainty_metrics_prompts["self_consistency_system_prompt"]
        usr_prompt = uncertainty_metrics_prompts["self_consistency_user_prompt"].format(test_claim=test_claim, 
                                                                                        candidate_claims=candidate_claims)
        response = await self.model.query(usr_prompt, temperature=0.0)
        # extract numerical rating from response
        rating = re.search(r'(\d+)', response)
        rating = int(rating.group(1))
        assert rating is not None, f"No numerical value found in response: {response}"
        assert rating <= 10, f"Rating should be less than 10: {rating}, response: {response}"
        assert rating >= 0, f"Rating should be greater than 0: {rating}, response: {response}"
        
        return rating / 10.0