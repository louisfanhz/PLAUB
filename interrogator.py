import json
from json.decoder import JSONDecodeError
import os
import time
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Set, Dict, Any, Optional
import asyncio
import openai
from openai import OpenAI, BadRequestError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field, ValidationError
import configs
from evaluator import FactScoreEvaluator, SelfConsistencyEvaluator, LongFactEvaluator
from language_models import OpenAIModel, LlamaModel, NLIModel
from prompts import self_eval_prompts, interrogator_prompts
from utils import retry_if_unsuccessful
from result_formats import Claim, ClaimAnalysis
from rich import print as rprint
import sys

class Interrogator:
    # def __init__(self, eval_model: str, ref_model: str):
    def __init__(self, model, dataset_name: str):
        self.model = model
        # self.eval_model_name = eval_model
        # self.eval_model = OpenAIModel(model=eval_model)
        self.nli_model = NLIModel()
        self.sc_evaluator = SelfConsistencyEvaluator(model=model)
        self.ref_evaluator = None
        if dataset_name == "factscore":
            self.ref_evaluator = FactScoreEvaluator(
                db_path="./dataset/factscore/enwiki-20230401.db",
                retrieval_type="gtr"
            )
        elif dataset_name == "longfact":
            self.ref_evaluator = LongFactEvaluator(
                dataset_path="./dataset/longfact/selected_prompts.json"
            )
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported")

    async def extract_atomic_claims(self, topic: str, context: str, text: str) -> List[Claim]:
        acs = await self._extract(context, text)

        # supported_scores = await self.sc_evaluator.evaluate_claims(texts, acs)
        if self.ref_evaluator:
            # correctnesses = await self.ref_evaluator.evaluate_claims(topic, cleaned_acs, batch_size=10)
            correctnesses = await self.ref_evaluator.evaluate_claims(topic, acs, one_by_one=True)
        else:
            correctnesses = None

        return [Claim(content=acs[i],
                      correctness=correctnesses[i] if correctnesses else None) for i in range(len(acs))]

    @retry_if_unsuccessful(max_retries=2)
    async def _extract(self, context: str, text: str) -> List[str]:
        sys_prompt = interrogator_prompts["extract_ac_system_prompt"]
        usr_prompt = interrogator_prompts["extract_ac_user_prompt"].format(context=context, text=text)

        class AtomicClaims(BaseModel):
            atomic_claims: list[str] = Field(description="A list of atomic claims extracted from the text.")

        response = await self.model.query_structured(sys_prompt, usr_prompt, spec=AtomicClaims)
        atomic_claims = response.atomic_claims

        return atomic_claims

    async def _clean_atomic_claims(self, ac_lists: List[List[str]]) -> List[str]:
        assert len(ac_lists) > 0, "No atomic claims to union"

        valid_ac_lists = [ac_list for ac_list in ac_lists if len(ac_list) > 0]
        if len(valid_ac_lists) == 0:
            raise ValueError("No valid atomic claims to union")
        
        cleaned_acs = valid_ac_lists[0]
        for ac_list in valid_ac_lists[1:]:
            ac_list_dedup = self._remove_duplicate_claims(cleaned_acs, ac_list)
            cleaned_acs = await self._remove_redundant_claims(cleaned_acs, ac_list_dedup)

        return cleaned_acs

    def _remove_duplicate_claims(self, ac_list1: List[str], ac_list2: List[str]) -> List[str]:
        # remove duplicate claims by NLI model
        entail_mat_1 = self.nli_model.compute_claim_entail(ac_list1, ac_list2)
        entail_mat_2 = self.nli_model.compute_claim_entail(ac_list2, ac_list1).T

        duplicate_claims = np.logical_and(entail_mat_1, entail_mat_2).any(axis=0).nonzero()[0]
        ac_list2 = [ac_list2[i] for i in range(len(ac_list2)) if i not in duplicate_claims]

        return ac_list2

    @retry_if_unsuccessful(max_retries=1)
    async def _remove_redundant_claims(self, ac_list1: List[str], ac_list2: List[str]) -> List[str]:
        # remove redundant claims by LLM
        sys_prompt = interrogator_prompts["rm_redundant_ac_system_prompt"]
        ac_list1_formatted = "\n".join([f"{i+1}. {ac}" for i, ac in enumerate(ac_list1)])
        ac_list2_formatted = "\n".join([f"{i+1}. {ac}" for i, ac in enumerate(ac_list2)])
        usr_prompt = interrogator_prompts["rm_redundant_ac_user_prompt"].format(claim_list_A=ac_list1_formatted, 
                                                                                claim_list_B=ac_list2_formatted)

        class RedundantClaimIndices(BaseModel):
            redundant_claim_indices: list[int] = Field(description="indices of the redundant claims.")

        response = await self.model.query_structured(sys_prompt, usr_prompt, spec=RedundantClaimIndices)
        redundant_indices = response.redundant_claim_indices
        redundant_indices = [idx-1 for idx in redundant_indices]  # correct for 1-indexing
        redundant_indices = [idx for idx in redundant_indices if idx in range(len(ac_list2))]  # remove out-of-range indices

        res_claims = ac_list1.copy()
        res_claims.extend([ac_list2[idx] for idx in range(len(ac_list2)) if idx not in redundant_indices])

        return res_claims

    async def raise_questions_from_claims_single(self, context: str, claim_objects: List[Claim], num_q_per_claim: int = 3):
        claims = [claim.content for claim in claim_objects]
        q_lists = await asyncio.gather(*[self._raise_questions_from_claims_single(context, claim) for claim in claims for _ in range(num_q_per_claim)])
        q_lists = [q_lists[i:i+num_q_per_claim] for i in range(0, len(q_lists), num_q_per_claim)]

        # For every sublist in q_lists, remove EXACT-MATCHED questions
        for i in range(len(q_lists)):
            unique_questions = []
            seen = set()
            for question in q_lists[i]:
                if question not in seen:
                    seen.add(question)
                    unique_questions.append(question)
            q_lists[i] = unique_questions

        assert len(q_lists) == len(claims), f"Number of question lists does not equal number of claims: {len(q_lists)} != {len(claims)} \n\n {claims} \n\n {q_lists}"

        return q_lists

    @retry_if_unsuccessful(max_retries=1)
    async def _raise_questions_from_claims_single(self, context: str, claim: str):
        sys_prompt = interrogator_prompts["q_from_single_claim_system_prompt"]
        usr_prompt = interrogator_prompts["q_from_single_claim_user_prompt"].format(context=context, claim=claim)

        response = await self.model.query(usr_prompt, **configs.generate_diverse_params[self.model.name])

        return response