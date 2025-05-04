import numpy as np
from typing import List, Dict, Optional, Union, Any, Callable
from pydantic import BaseModel, Field
import sys

class ClaimAnalysis(BaseModel):
    question: str
    # specificity: Union[int, None] = Field(default=None)
    answers: Union[List[Dict[str, Any]], None] = Field(default=None)

    def is_populated(self):
        return self.answers is not None

    def gather_answer_scores(self, metric: str, callback: Callable[List[float], float]):
        return callback([ans[metric] for ans in self.answers])

class Claim(BaseModel):
    content: str
    correctness: Union[str, None] = Field(default=None)
    supported_score: Union[float, None] = Field(default=None)
    claim_analysis: List[ClaimAnalysis] = Field(default=None)

    def is_populated(self):
        if len(self.claim_analysis) == 0:
            return False
        return all([ca.is_populated() for ca in self.claim_analysis])

    def gather_claim_analysis_scores(self, metric: str, callback: Callable[List[float], float], reduction: str):
        ans_scores = [ca.gather_answer_scores(metric, callback) for ca in self.claim_analysis]
        if reduction == "mean":
            claim_score = np.mean(ans_scores).item()
        elif reduction == "max":
            claim_score = np.max(ans_scores).item()
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")

        return claim_score

class GenSampleResult(BaseModel):
    gen_idx: int
    all_claims: List[str]
    all_questions: List[str]
    claims: List[Claim]

    def is_populated(self):
        if len(self.claims) == 0:
            return False
        return all([claim.is_populated() for claim in self.claims])

    def gather_claim_scores(self, ca_reduction: str, ans_callbacks: Dict[str, Callable[List[float], float]]):
        claim_scores = {metric: [] for metric in ans_callbacks.keys()}

        for metric, callback in ans_callbacks.items():
            for claim in self.claims:
                claim_scores[metric].append(claim.gather_claim_analysis_scores(metric, callback, ca_reduction))

        return claim_scores

    def gather_impacts(self):
        impacts = []
        for claim in self.claims:
            impacts.append(claim.gather_claim_analysis_scores("impact2", lambda x: np.mean(x).item(), "mean"))
        impacts = np.array(impacts)# > 0.5

        # cumulative_impacts = 1 / np.exp(np.clip(np.cumsum(impacts), 0, 6))
        # return cumulative_impacts

        weight_func = np.exp(-np.arange(len(impacts)))
        weights = np.convolve(impacts, weight_func)[:len(impacts)]
        impacts = 1 / np.exp(weights)
        return impacts

        # impacts = impacts[::-1]
        # weight_func = np.exp(-np.arange(len(impacts)))
        # weights = np.convolve(impacts, weight_func)[:len(impacts)]
        # impacts = 1 / np.exp(weights)
        # return impacts[::-1]

    def gather_correctness(self):
        return [claim.correctness for claim in self.claims]

    def gather_supported_score(self):
        return [claim.supported_score for claim in self.claims]

    def gather_claim_contents(self):
        return [claim.content for claim in self.claims]

class TopicResult(BaseModel):
    gen_analysis: List[GenSampleResult] = Field(default=[])

    def is_populated(self):
        if len(self.gen_analysis) == 0:
            return False
        return all([gen.is_populated() for gen in self.gen_analysis])

    # def gather_
