import os
import numpy as np
import json
import gc
from tqdm import tqdm
from functools import wraps
from typing import List, Set, Dict, Any, Optional

from prompts import uncertainty_metrics_prompts
from language_models import NLIModel
from pprint import pprint
import configs
import sys


def uncertainty_metric(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper._is_metric = True
    return wrapper

class UncertaintyMetrics:
    """
    This class is used to compute uncertainty scores for a single LLM response, possibly using
    output tokens, log probabilities, etc.
    """
    def __init__(self):
        self.metric_functions = []
        
        # register all methods decorated with @metric
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_is_metric') and attr._is_metric:
                self.metric_functions.append(attr)
    
    def __call__(self, **data) -> Dict[str, Any]:
        """
        Call all metric functions to compute uncertainty scores given the data.
        Data is expected to contain all necessary fields for each metric function.
        """
        n_items = data["n_res"]
        results = []
        for idx in range(n_items):
            item_results = {}
            for metric_fn in self.metric_functions:
                metric_name = metric_fn.__name__
                item_results[metric_name] = metric_fn(idx, data)
            results.append(item_results)
        return results

    @property
    def metric_names(self) -> List[str]:
        return [metric_fn.__name__ for metric_fn in self.metric_functions]

    @uncertainty_metric
    def perplexity(self, idx, data):
        log_probs = data["log_probs"][idx]
        clipped_log_probs = np.clip(log_probs, -16, 0)  # Clip to prevent underflow/overflow
        perplexity = np.exp(-np.sum(clipped_log_probs) / len(clipped_log_probs))
        return perplexity

    @uncertainty_metric
    def entropy(self, idx, data):
        log_probs = data["log_probs"][idx]
        clipped_log_probs = np.clip(log_probs, -16, 0)  # Clip to prevent underflow/overflow
        entropy = -np.sum(np.exp(clipped_log_probs) * clipped_log_probs) # no length normalization
        return entropy

    # def build_consistency_check_prompt(self, test_claim, candidate_claims):
    #     candidate_claims = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(candidate_claims)])

    #     sys_prompt = uncertainty_metrics_prompts["self_consistency_system_prompt"]
    #     usr_prompt = uncertainty_metrics_prompts["self_consistency_user_prompt"].format(test_claim=test_claim, 
    #                                                                                     candidate_claims=candidate_claims)
        
    #     response_spec = {
    #         "name": "evaluate_consistency",
    #         "schema": {
    #             "type": "object",
    #             "properties": {
    #                 "ratings_list": {
    #                     "type": "array", 
    #                     "items": {
    #                         "type": "integer",
    #                         "description": "The rating of the candidate claim",
    #                         "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #                     },
    #                 },
    #             },
    #             "required": ["ratings_list"],
    #             "additionalProperties": False
    #         }
    #     }

    #     return sys_prompt, usr_prompt, response_spec

    # @uncertainty_metric
    # def consistency_with_claim(self, idx, data):
    #     test_claim = data["claim"]
    #     candidate_claims = [data["texts"][idx]]

    #     sys_prompt, usr_prompt, response_spec = self.build_consistency_check_prompt(test_claim, candidate_claims)

    #     response = self.oracle_model.query_structured(sys_prompt, usr_prompt, response_spec)
    #     ratings = json.loads(response.output_text)["ratings_list"]

    #     assert len(ratings) == 1, f"Only one candidate claim rating is expected"

    #     return ratings[0] / 10.0

    # @uncertainty_metric
    # def consistency_with_other_answers(self, idx, data):
    #     test_claim = data["texts"][idx]
    #     candidate_claims = [f"{claim.strip()}" for i, claim in enumerate(data["texts"]) if i != idx]

    #     sys_prompt, usr_prompt, response_spec = self.build_consistency_check_prompt(test_claim, candidate_claims)

    #     response = self.oracle_model.query_structured(sys_prompt, usr_prompt, response_spec)
    #     ratings = json.loads(response.output_text)["ratings_list"]

    #     return np.mean(ratings) / 10.0


class UncertaintyScores:
    def __init__(self, llm_model: str="llama-3-70b", eval_type: str="claim_level"):
        self.nli_model = NLIModel()

        if eval_type == "claim_level":
            self.eval_type = "claim_level"

    def get_num_semantic_equal_sets(self, claims: List[str]) -> int:
        def get_semantic_equal_sets(entail_mat: np.ndarray) -> List[Set[int]]:
            # ensure symmetric entailment
            entail_mat = entail_mat & (entail_mat == entail_mat.T)

            SE = [set(np.flatnonzero(row).tolist()) for row in np.triu(entail_mat)]
    
            merged = True
            while merged:
                merged = False
                i = 0
                while i < len(SE):
                    j = i + 1
                    while j < len(SE):
                        # Check if sets i and j share any elements
                        if SE[i] & SE[j]:  # Set intersection
                            # Merge set j into set i
                            SE[i] |= SE[j]  # Set union
                            # Remove set j
                            SE.pop(j)
                            merged = True
                        else:
                            j += 1
                    i += 1
            
            return SE

        entail_mat = self.nli_model.compute_claim_entail(claims, claims)
        SE = get_semantic_equal_sets(entail_mat)

        return len(SE)
    
        
