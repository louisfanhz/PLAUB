import os
import asyncio
import json
import random
from json.decoder import JSONDecodeError
from pydantic import ValidationError
from openai import RateLimitError
from together.error import ServiceUnavailableError, APIError, APIConnectionError, InvalidRequestError, RateLimitError
import re
import shelve
from functools import wraps
from typing import List, Dict, Any, Optional, Union
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn import metrics
import matplotlib.pyplot as plt

from result_formats import ClaimAnalysis, Claim, TopicResult
from rich import print as rprint
import sys

class Constants:
    INVALID_GENERATION = "invalid"
    TASK_CANCELLED = "task_cancelled"

def retry_if_unsuccessful(max_retries: int=1, errors: tuple=(AssertionError, 
                                                            JSONDecodeError, 
                                                            ValidationError, 
                                                            # APIError,
                                                            # APIConnectionError,
                                                            RateLimitError)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            num_retries = 0
            max_retries_server_error = 3
            num_retries_server_error = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except errors as e:
                    topic = args[0] if args else kwargs.get('topic', 'unknown')
                    rprint(f"encountered {e.__class__.__name__} for {topic}, retrying... {num_retries}/{max_retries}")
                    await asyncio.sleep(1)
                    
                    num_retries += 1
                    if num_retries > max_retries:
                        raise e
                except (ServiceUnavailableError, APIError, APIConnectionError) as e:
                    topic = args[0] if args else kwargs.get('topic', 'unknown')
                    rprint(f"encountered {e.__class__.__name__} for {topic}, retrying... {num_retries_server_error}/{max_retries_server_error}")

                    await asyncio.sleep(5 * random.random() * num_retries_server_error)
                    num_retries_server_error += 1
                    if num_retries_server_error > max_retries_server_error:
                        raise e
        return wrapper
    return decorator

def cancel_if_error(func, errors: tuple=(AssertionError, 
                                        JSONDecodeError, 
                                        ValidationError, 
                                        ServiceUnavailableError,
                                        APIError,
                                        APIConnectionError,
                                        RateLimitError)):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except errors as e:
            topic = args[0] if args else kwargs.get('topic', 'unknown')
            print(f"Error in {func.__name__} for {topic} after retries: {str(e)}. Cancelling task.")
            return Constants.TASK_CANCELLED
        except Exception as e:
            raise e
    return wrapper

class CacheFileManager:
    def __init__(self, cache_path: str, from_jsonl: str=None, from_json: str=None):
        self.cache_path = cache_path
        self.cache_dir = os.path.dirname(cache_path)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self._cache = shelve.open(cache_path, writeback=True)

        if from_jsonl is not None:
            self._from_jsonl(from_jsonl)

        if from_json is not None:
            self._from_json(from_json)

    @property
    def cache(self):
        return self._cache

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value
    
    def __getitem__(self, key: str) -> Any:
        return self._cache[key]

    def sync(self):
        self._cache.sync()

    def _from_jsonl(self, jsonl_path: str):
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self._cache[data["topic"]] = data

    def _from_json(self, json_path: str):
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
            for topic, data in data_dict.items():
                self._cache[topic] = data

    def to_json(self):
        json_path = self.cache_path + ".json"
        with open(json_path, 'w') as f:
            json.dump(dict(self._cache), f, indent=2, ensure_ascii=False)

    def __del__(self):
        self._cache.close()

class Plotting:
    def __init__(self, results: Dict[str, Any], metric_names: List[str]):
        self.results = results
        
        ans_callbacks = {metric_name: lambda x: -np.mean(x).item() for metric_name in metric_names}
        ca_reduction = "mean"

        metric_scores = {name: [] for name in metric_names}
        metric_scores["claim_supported_score"] = []
        metric_scores["claim_supported_score_w_impact"] = []
        metric_scores["claim_supp_score_w_impact_w_settling_sc"] = []
        metric_scores["self_consistency_w_impact"] = []
        # metric_scores["ans_supported_score_w_impact"] = []

        metric_scores["closeness_centrality_with_node_confidence"] = []
        metric_scores["closeness_centrality"] = []

        # metric_scores["eigenvector_centrality"] = []
        # metric_scores["betweenness_centrality"] = []
        # metric_scores["pagerank"] = []
        # metric_scores["claim_c_and_self_c_w_impact"] = []
        metric_names.append("claim_supported_score")
        metric_names.append("claim_supported_score_w_impact")
        metric_names.append("claim_supp_score_w_impact_w_settling_sc")
        metric_names.append("self_consistency_w_impact")
        # metric_names.append("ans_supported_score_w_impact")
        metric_names.append("closeness_centrality_with_node_confidence")
        metric_names.append("closeness_centrality")

        # metric_names.append("eigenvector_centrality")
        # metric_names.append("betweenness_centrality")
        # metric_names.append("pagerank")
        # metric_names.append("claim_c_and_self_c_w_impact")
        labels = []
        invalid_labels = []
        claim_contents = []
        for topic in self.results.keys():
            # if topic in ['Nobuhiro Shimatani', 'Theophilus Annorbaah', 'Stephanus Swart', 'Maxime Masson', 'Christian Almeida', 'Muhammad Alhamid', 'Phillip Gillespie']:
            #     continue
            topic_res = TopicResult(**self.results[topic])
            for ga in topic_res.gen_analysis:

                metric_scores["closeness_centrality_with_node_confidence"].extend(self.get_graph_based_results(topic, ga.gen_idx)["closeness_centrality_with_node_confidence"])
                metric_scores["closeness_centrality"].extend(self.get_graph_based_results(topic, ga.gen_idx)["closeness_centrality"])
                # metric_scores["eigenvector_centrality"].extend(self.get_graph_based_results(topic, ga.gen_idx)["eigenvector_centrality"])
                # metric_scores["betweenness_centrality"].extend(self.get_graph_based_results(topic, ga.gen_idx)["betweenness_centrality"])
                # metric_scores["pagerank"].extend(self.get_graph_based_results(topic, ga.gen_idx)["pagerank"])

                claim_uncertainties = ga.gather_claim_scores(ca_reduction, ans_callbacks)
                uncertainty_scores_holder = {name: [] for name in metric_names}
                for name, uncertainty in claim_uncertainties.items():
                    uncertainty_scores_holder[name].extend(uncertainty)
                for name, uncertainty in claim_uncertainties.items():
                    uncertainty_scores_holder[name] = np.array(uncertainty)

                for name in metric_names:
                    metric_scores[name].extend(uncertainty_scores_holder[name])

                claim_supported_scores = ga.gather_supported_score()
                metric_scores["claim_supported_score"].extend(claim_supported_scores)

                labels.extend(ga.gather_correctness())

                impacts = ga.gather_impacts()
                metric_scores["self_consistency_w_impact"].extend(uncertainty_scores_holder["self_consistency"] * impacts)
                metric_scores["claim_supported_score_w_impact"].extend(claim_supported_scores * impacts)
                # print(topic)

                unsure_claim_mask = np.logical_and(np.array(claim_supported_scores) < 0.7, np.array(claim_supported_scores) > 0.3)
                settling_sc_scores = uncertainty_scores_holder["self_consistency"][unsure_claim_mask]
                csswi = claim_supported_scores * impacts
                csswi[unsure_claim_mask] = settling_sc_scores * csswi[unsure_claim_mask]
                metric_scores["claim_supp_score_w_impact_w_settling_sc"].extend(csswi)
                

                # metric_scores["ans_supported_score_w_impact"].extend(uncertainty_scores_holder["ans_supported_score"] * impacts)


                claim_contents.extend(ga.gather_claim_contents())

                # print(metric_scores["self_consistency"])
                # print(impacts)
                # print(metric_scores["self_consistency_w_impact"])

        # normalize_bw_0_1 = lambda x: (x - min(x)) / (max(x) - min(x))
        # metric_scores["closeness_centrality_with_node_confidence"] = np.array(metric_scores["closeness_centrality_with_node_confidence"])
        # metric_scores["closeness_centrality_with_node_confidence"] = normalize_bw_0_1(metric_scores["closeness_centrality_with_node_confidence"])
        # metric_scores["closeness_centrality_with_node_confidence"] = metric_scores["closeness_centrality_with_node_confidence"].tolist()

        # metrics2test = ["claim_supported_score_w_impact", "closeness_centrality_with_node_confidence"]
        # test_metric_A = np.array(metric_scores[metrics2test[0]])
        # test_metric_B = np.array(metric_scores[metrics2test[1]])
        # labels = np.array(labels)

        # # Analyze distributions of both metrics
        # print("\nSupported Score Distribution:")
        # print(f"Unique values: {np.unique(supported_score_test)}")
        # print(f"Number of unique values: {len(np.unique(supported_score_test))}")
        
        # print("\nSelf Consistency with Impact Distribution:")
        # print(f"Unique values: {np.unique(self_consistency_w_impact_test)}")
        # print(f"Number of unique values: {len(np.unique(self_consistency_w_impact_test))}")

        # possible_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        # rounded_metric_A = np.zeros_like(test_metric_A)
        # for i, value in enumerate(test_metric_A):
        #     diffs = np.abs(possible_values - value)
        #     closest_idx = np.argmin(diffs)
        #     rounded_metric_A[i] = possible_values[closest_idx]
        # test_metric_A = rounded_metric_A

        # rounded_metric_B = np.zeros_like(test_metric_B)
        # for i, value in enumerate(test_metric_B):
        #     diffs = np.abs(possible_values - value)
        #     closest_idx = np.argmin(diffs)
        #     rounded_metric_B[i] = possible_values[closest_idx]
        # test_metric_B = rounded_metric_B

        # # Compute TPR and FPR for both metrics
        # fpr_metric_A, tpr_metric_A, thresholds_metric_A = metrics.roc_curve(labels, test_metric_A)
        # print(f"\n{metrics2test[0]}:")
        # print(f"TPR: {tpr_metric_A}")
        # print(f"FPR: {fpr_metric_A}")
        # print(f"thresholds: {thresholds_metric_A}")
        
        # fpr_metric_B, tpr_metric_B, thresholds_metric_B = metrics.roc_curve(labels, test_metric_B)
        # print(f"\n{metrics2test[1]}:")
        # print(f"TPR: {tpr_metric_B}")
        # print(f"FPR: {fpr_metric_B}")
        # print(f"thresholds: {thresholds_metric_B}")

        # # plt.plot(fpr_metric_A, tpr_metric_A, label="Metric A")
        # # plt.plot(fpr_metric_B, tpr_metric_B, label="Metric B")
        # # plt.legend()
        # # plt.savefig(f"./results/roc_plot_rounded.png")
        
        # print(len(self.results))
        # print(len(labels))
        # print(f"{metrics2test[0]}: max={max(test_metric_A)}, min={min(test_metric_A)}")
        # print(f"{metrics2test[1]}: max={max(test_metric_B)}, min={min(test_metric_B)}")
        # print(np.mean(test_metric_A[labels == True] - test_metric_B[labels == True]))
        # print(test_metric_A[labels == True].sum() - test_metric_B[labels == True].sum())
        # sys.exit()

        ### exclude correctness = not enough information ###
        labels = np.array(labels) 
        rprint(np.unique(labels))
        metric_scores = {name: np.array(scores) for name, scores in metric_scores.items()}
        correct_labels = labels == "correct"
        invalid_labels = labels == "irrelevant"
        # correct_labels = correct_labels[~invalid_labels]
        # for name in metric_names:
        #     metric_scores[name] = metric_scores[name][~invalid_labels]

        # print(f"total_entries: {total_entries}, num_correct: {num_correct}, num_incorrect: {num_incorrect}, num_not_enough_info: {num_not_enough_info}")
        self.plot_auroc("uncertainty_metrics", metric_scores, correct_labels, metric_names)
        # self.plot_auroc("SC_score", self.orig_claim_lv_baseline_scores, self.orig_claim_lv_correctness, ["SC_score", "CWAA_score"])

    def plot_auroc(self, save_name: str, metric_scores: Dict[str, np.ndarray], labels: np.ndarray, metric_names: List[str]):
        # auroc, auprc, auprc_n = metrics.roc_auc_score(dict_collection['corr'], value), metrics.average_precision_score(dict_collection['corr'], value), metrics.average_precision_score(1 - dict_collection['corr'], -value)

        for metric_name in metric_names:
            auroc = metrics.roc_auc_score(labels, metric_scores[metric_name])
            auprc = metrics.average_precision_score(labels, metric_scores[metric_name])
            auprc_n = metrics.average_precision_score(1 - labels, -metric_scores[metric_name])

            fpr, tpr, _ = metrics.roc_curve(labels, metric_scores[metric_name])
            plt.plot(fpr, tpr, label=f"{metric_name}, auc={round(auroc, 3)}, auprc={round(auprc, 3)}, auprc_n={round(auprc_n, 3)}")

            print(f"{metric_name}: auroc={auroc}, auprc={auprc}, auprc_n={auprc_n}")

        plt.legend()
        plt.savefig(f"./results/{save_name}_roc_plot.png")

    def get_graph_based_results(self, topic: str, gen_idx: int):
        # with open("./cache/gpt-4o_factscore_baseline_results.json", "r") as f:
        # with open("./cache/gpt-4o_longfact_baseline_results.json", "r") as f:
        # with open("./cache/gpt-4o-mini_factscore_baseline_results.json", "r") as f:
        # with open("./cache/gpt-4o-mini_longfact_baseline_results.json", "r") as f:
        # with open("./cache/gpt-4.1-mini_factscore_baseline_results.json", "r") as f:
        # with open("./cache_4omini_30sample_longfact/gpt-4o-mini_longfact__baseline_results.json", "r") as f:
        # with open("./cache/meta-llama/Llama-3.3-70B-Instruct-Turbo_factscore_baseline_results.json", "r") as f:
        with open("./cache/meta-llama/Llama-3.3-70B-Instruct-Turbo_longfact_baseline_results.json", "r") as f:
        # with open("./cache/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8_factscore_baseline_results.json", "r") as f:
            g_results = json.load(f)

        return g_results[topic][str(gen_idx)]

class TextColor():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    LIGHT_GRAY = '\033[37m'
    DARK_GRAY = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    WHITE = '\033[97m'

    RESET = '\033[0m' # called to return to standard terminal text color

def colorstr(text, color):
    assert color.upper() in TextColor.__dict__.keys()
    code = TextColor.__dict__[color.upper()]
    return f"{code}{text}{TextColor.RESET}"

def color_token_with_probs(generated_tokens, transition_probs):
    assert len(generated_tokens) == len(transition_probs), f"dimension of tokens ({len(generated_tokens)}) should match dimension of probabilities ({len(transition_probs)})"

    prob_labels = [
        (0.8, "green"),
        (0.5, "yellow"),
        (1e-20, "red"),
    ]
    colored_output = ""
    for token, prob in zip(generated_tokens, transition_probs):
        color = None
        assert 0. <= prob <= 1.0
        for min_prob, label in prob_labels:
            if prob >= min_prob:
                color = label
                break
        colored_output += colorstr(token.replace("▁", " "), color)

    return colored_output

# Adapted from FactScoreLite.atomic_facts.py
# https://github.com/armingh2000/FactScoreLite/tree/main/FactScoreLite
def detect_initials(text: str) -> list:
    """
    Detects initials in the text.

    Args:
        text (str): The text to detect initials in.

    Returns:
        list: A list of detected initials.
    """
    pattern = r"[A-Z]\. ?[A-Z]\."
    return re.findall(pattern, text)

# Adapted from FactScoreLite.atomic_facts.py
# https://github.com/armingh2000/FactScoreLite/tree/main/FactScoreLite
def fix_sentence_splitter(sentences: list, initials: list) -> list:
    """
    Fixes sentence splitting issues based on detected initials, handling special cases.

    Args:
        sentences (list): List of sentences to fix.
        initials (list): List of detected initials.

    Returns:
        list: Sentences with corrected splitting issues.

    This method corrects sentence splitting issues by merging incorrectly split sentences
    based on detected initials. It also addresses special cases such as sentences
    containing only one word or starting with a lowercase letter to ensure proper formatting.
    """
    for initial in initials:
        if not np.any([initial in sent for sent in sentences]):
            alpha1, alpha2 = [
                t.strip() for t in initial.split(".") if len(t.strip()) > 0
            ]
            for i, (sent1, sent2) in enumerate(zip(sentences, sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    sentences = (
                        sentences[:i]
                        + [sentences[i] + " " + sentences[i + 1]]
                        + sentences[i + 2 :]
                    )
                    break

    results = []
    combine_with_previous = None

    for sent_idx, sent in enumerate(sentences):
        if len(sent.split()) <= 1 and sent_idx == 0:
            assert not combine_with_previous
            combine_with_previous = True
            results.append(sent)
        elif len(sent.split()) <= 1:
            assert sent_idx > 0
            results[-1] += " " + sent
            combine_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, results
            results[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            results[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            results.append(sent)

    return results

def text_to_sentences(text):
    initials = detect_initials(text)
    sentences = sent_tokenize(text)
    sentences = fix_sentence_splitter(sentences, initials)

    return sentences