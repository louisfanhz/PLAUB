from utils import CacheFileManager
from language_models import LlamaModel
from uncertainty_metrics import UncertaintyMetrics
import pytest
import os
import re
import json
import queue
import shelve
import tempfile
import pandas as pd
from pprint import pprint
import numpy as np
from sklearn import metrics
from uncertainty_metrics import UncertaintyMetrics
from evaluator import LongFactEvaluator
from prompts import uncertainty_metrics_prompts
from utils import Plotting
from language_models import LlamaModel, TogetherAIModel
from evaluator import GraphBasedEvaluator
from result_formats import TopicResult
import asyncio
import time
from rich import print as rprint

import sys

def test_llama_evaluator():
    evaluator = LlamaModel()
    prompt = "What is the capital of France?"
    tokens, logprobs = evaluator.query_logprobs(prompt)
    assert tokens is not None
    assert logprobs is not None

def test_cache_file_manager_to_json():
    # Create a temporary test file
    test_data = {
        "James": {"age": 20, "gender": "male", "city": "New York"},
        "John": {"age": 25, "gender": "male", "city": "Los Angeles"},
        "Jane": {"age": 30, "gender": "female", "city": "Chicago"}
    }
    
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        # Initialize CacheFileManager
        cfm = CacheFileManager(temp_path.replace(".json", ""))
        for k, v in test_data.items():
            cfm.cache[k] = v
        
        # Test writing to JSON
        json_path = temp_path.replace(".json", "_new.json")
        cfm.to_json(json_path)
        
        # Verify the JSON file contents
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
            assert len(loaded_data) == 3
            assert loaded_data["James"]["age"] == 20
            assert loaded_data["John"]["age"] == 25
            assert loaded_data["Jane"]["age"] == 30
        
        # Test updating cache and writing back to JSON
        cfm.cache["James"]["age"] = 21
        cfm.to_json(json_path)
        
        # Verify the updated JSON file contents
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
            assert loaded_data["James"]["age"] == 21
            assert loaded_data["John"]["age"] == 25
            assert loaded_data["Jane"]["age"] == 30
            
    finally:
        # Cleanup
        os.unlink(temp_path)
        if os.path.exists(json_path):
            os.unlink(json_path)

def test_uncertainty_metrics():
    # Initialize UncertaintyMetrics
    metrics = UncertaintyMetrics()
    
    # Test perplexity calculation
    log_probs = [-0.5, -0.3, -0.4]
    perplexity_result = metrics.perplexity({"log_probs": log_probs})
    expected_perplexity = np.exp(-np.sum(log_probs) / len(log_probs))
    assert np.isclose(perplexity_result, expected_perplexity)
    
    # Test entropy calculation
    entropy_result = metrics.entropy({"log_probs": log_probs})
    expected_entropy = -np.sum(np.exp(log_probs) * log_probs)
    assert np.isclose(entropy_result, expected_entropy)
    
    # Test calling all metrics at once
    results = metrics(log_probs=log_probs)
    assert "perplexity" in results
    assert "entropy" in results
    assert np.isclose(results["perplexity"], expected_perplexity)
    assert np.isclose(results["entropy"], expected_entropy)
    
    # Test with different log probabilities
    log_probs2 = [-0.1, -0.2, -0.3]
    results2 = metrics(log_probs=log_probs2)
    assert "perplexity" in results2
    assert "entropy" in results2
    
    # Test with edge cases
    log_probs3 = [0.0, 0.0, 0.0]  # All probabilities are 1.0
    results3 = metrics(log_probs=log_probs3)
    assert np.isclose(results3["perplexity"], 1.0)  # Perplexity should be 1.0 when all probabilities are 1.0
    assert np.isclose(results3["entropy"], 0.0)  # Entropy should be 0.0 when all probabilities are 1.0

# def test_plot_auroc():
#     keys_mapping = {
#         f'sc_score_{sample_num}samples': 'SC',
#         'verbalized_confidence_with_options': 'PH-VC',
#         f'breakdown_closeness_centrality_{sample_num}samples': r'$C_C$',
#         f'sc_+_vc': 'SC+VC',
#     }

#     auroc = metrics.roc_auc_score(dict_collection['corr'], value)
#     auprc = metrics.average_precision_score(dict_collection['corr'], value)
#     auprc_n = metrics.average_precision_score(1 - dict_collection['corr'], -value)
#     results = {'auroc': auroc, 'auprc': auprc, 'auprc_n': auprc_n}
#     if key in keys_to_names:
#         fpr, tpr, _ = metrics.roc_curve(dict_collection['corr'], value)
#         plt.plot(fpr, tpr, label=f"{keys_to_names[key]}, auc={round(auroc, 3)}, auprc={round(auprc, 3)}, auprc_n={round(auprc_n, 3)}")
#     result_dict[key] = results

def test_auroc():
    with open('results/graph_based_uncertainty_result/gens_0to3/factscore_m_gpt-3.5-turbo_bipartite_sc_6samples_5matches.json', 'r') as f:
        results = json.load(f)

    plot_auroc(results)

def plot_auroc(results):
    first_n, sample_num = len(results), 5
    keys_to_names = {
        f'sc_score_{sample_num}samples': 'SC',
        'verbalized_confidence_with_options': 'PH-VC',
        f'breakdown_closeness_centrality_{sample_num}samples': r'$C_C$',
        f'sc_+_vc': 'SC+VC',
    }

    dict_collection = collect_data_for_plotting(results=results)

    pprint(dict_collection)
    sys.exit()
    result_dict = {}
    plt.figure(0).clf()

    data_size = len(dict_collection['corr'])
    result_dict['data_size'] = data_size
    dict_bootstrap_auroc, dict_bootstrap_auprc, dict_bootstrap_auprc_n = {}, {}, {}

    for key, value in dict_collection.items():
        if key != 'corr':
            auroc, auprc, auprc_n = metrics.roc_auc_score(dict_collection['corr'], value), metrics.average_precision_score(dict_collection['corr'], value), metrics.average_precision_score(1 - dict_collection['corr'], -value)
            results = {'auroc': auroc, 'auprc': auprc, 'auprc_n': auprc_n}
            if key in keys_to_names:
                fpr, tpr, _ = metrics.roc_curve(dict_collection['corr'], value)
                plt.plot(fpr, tpr, label=f"{keys_to_names[key]}, auc={round(auroc, 3)}, auprc={round(auprc, 3)}, auprc_n={round(auprc_n, 3)}")
            result_dict[key] = results

    df = pd.DataFrame(dict_collection)
    plt.legend(loc=0)
    save_path = f'{self.folder_name}/plot_roc_curve_{self.args.model}_{self.args.num_samples_for_claims}matches_{sample_num + 1}samples.png'
    plt.savefig(save_path)
    save_stats_path = f'{self.folder_name}/plot_roc_curve_{self.args.model}_{self.args.num_samples_for_claims}matches_{sample_num + 1}samples.json'
    with open(save_stats_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

def collect_data_for_plotting(results):
    df = pd.DataFrame(results)
    sample_num = 5

    corr = collect_all_values(df, 'gpt-score')
    index = np.where((corr == 'Y') | (corr == 'N'))[0].astype(int)
    corr = (corr == 'Y').astype(float)
    dict_collection = {'corr': corr}
    collect_keys = [key for key in df['pointwise_dict'][0][0].keys() if key not in ["claim", "correctness", "gpt-score", "gpt_annotation_result"]]

    for key in collect_keys:
        dict_collection[key] = collect_all_values(df, key)
        index = np.intersect1d(index, np.where((dict_collection[key] != -1) & (dict_collection[key] != np.inf) & (~np.isnan(dict_collection[key])))[0].astype(int))
        
    for key in dict_collection:
        dict_collection[key] = np.array(dict_collection[key])[index]

    return dict_collection

def collect_all_values(df, key_required):
    corr = []
    for index, row in df.iterrows():
        for i, item in enumerate(row['pointwise_dict']):            
            corr.append(item[key_required])

    return np.array(corr)

def test_cache_file_manager():
    cfm = CacheFileManager(cache_path="./cache/test_cache_file_manager")
    cfm["test"] = "test"
    cfm.close()

    cfm2 = CacheFileManager(cache_path="./cache/test_cache_file_manager")
    assert "test" in cfm2.cache
    cfm2["test2"] = "test2"
    cfm2.close()

    cfm3 = CacheFileManager(cache_path="./cache/test_cache_file_manager")
    for k, v in cfm3.cache.items():
        print(k, v)

    cfm3.to_json("./cache/test_cache_file_manager.json")

    cfm4 = CacheFileManager(cache_path="./cache/test_cache_file_manager2",
                           from_json="./cache/test_cache_file_manager.json")
    print("loaded from json")
    for k, v in cfm4.cache.items():
        print(k, v)

def test_self_consistency():
    metrics = UncertaintyMetrics()
    texts = ["Shigeru Fukudome played as an outfielder in baseball.",
             "Shigeru Fukudome played outfield, primarily in right field, during his professional baseball career.", 
             "Shigeru Fukudome played as an outfielder in baseball.",
             "Shigeru Fukudome is a Japanese former professional baseball outfielder."]
    ratings = metrics.self_consistency(idx=0, data={"texts": texts})
    print(ratings)

def test_build_claim():
    prompt = uncertainty_metrics_prompts["self_consistency_user_prompt"].format(test_claim="Shigeru Fukudome played as an outfielder in baseball.", 
                                                                              candidate_claims=["Shigeru Fukudome played outfield, primarily in right field, during his professional baseball career.", 
                                                                                               "Shigeru Fukudome played as an outfielder in baseball.", 
                                                                                               "Shigeru Fukudome is a Japanese former professional baseball outfielder."])
    print(prompt)
    print()

    prompt = uncertainty_metrics_prompts["self_consistency_user_prompt"].format(test_claim="Shigeru Fukudome played as an outfielder in baseball.", 
                                                                              candidate_claims=["Shigeru Fukudome played outfield, primarily in right field, during his professional baseball career."]) 

    print(prompt)

def test_contain_logprobs():
    model = LlamaModel(model="deepseek/deepseek-v3-0324")
    prompt = "What is the capital of France?"
    tokens, logprobs = model.test_contain_logprobs(prompt)
    print(tokens)
    print(logprobs)

def test_longfact_evaluator():
    longfact_evaluator = LongFactEvaluator(dataset_path="./dataset/longfact/selected_prompts.json")
    
    topic = "Can you tell me about the Crab Nebula?"
    query = ["EY boards would vote on whether to split EY into two businesses.",
             "EY is a multinational professional services network headquartered in London, England."]
    passages = longfact_evaluator.retrieve_relevant_passages(topic, query)
    rprint(passages)

def test_asyncio():
    async def task1(sec):
        await asyncio.sleep(sec)
        print(f"Task 1 finished at {time.strftime('%X')}")

    async def task2(sec):
        await asyncio.sleep(sec)
        print(f"Task 2 finished at {time.strftime('%X')}")

    async def main(max_concurrent_tasks=3):
        print(f"Main started at {time.strftime('%X')}")

        tasks_queue = queue.Queue()
        tasks_queue.put(asyncio.create_task(task1(4)))
        tasks_queue.put(asyncio.create_task(task2(2)))

        while not tasks_queue.empty():
            task = tasks_queue.get()
            await task

        # for sec in [4, 2]:
        #     await task1(sec)
        
        print(f"Main finished at {time.strftime('%X')}")

    async def x(i):
        await asyncio.sleep(1)
        print(f"{i} printed at {time.strftime('%X')}")
        return i

    async def main2():
        for f in asyncio.as_completed([x(i) for i in range(10)]):
            result = await f
            print(result)

    asyncio.run(main2())

async def test_graph_based_evaluator():
    g_evaluator = GraphBasedEvaluator(model="gpt-4o-mini")
    generations = CacheFileManager(cache_path="./cache/llama-3-70b-instruct_factscore", 
                                   from_jsonl="./results/llama-3-70b-instruct_factscore.jsonl")
    
    with open("results/llama-3-70b-instruct_interrogate_results.json", "r") as f:
        results = json.load(f)

    for topic in results.keys():
        topic_res = TopicResult(**results[topic])
        question = topic
        all_claims = topic_res.gather_claim_contents()

        result = await g_evaluator.compute_centrality(topic,
                                                [generations[topic]["most_likely_generation"]] + generations[topic]["diverse_generations"], 
                                                all_claims)

        print(result)
        sys.exit()

def generation_viewer(topic: str):
    generations = CacheFileManager(cache_path="./cache/llama-3-70b-instruct_factscore", 
                               from_json="./cache_with_generation_filtering/llama-3-70b-instruct_factscore_generations.json")

    # pprint(generations[topic]["diverse_generations"])
    print(generations[topic].keys())
    print(len(generations.keys()))

def test_togetherai():
    model = TogetherAIModel(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    prompt = "Hi."
    model.query_stream(prompt)

def play_ground():
    impacts = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1])
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    sigmoid_kernel = sigmoid(np.arange(len(impacts)))
    exp_kernel = np.exp(-np.arange(len(impacts)))
    linear_decay_kernel = np.linspace(1, 0, len(impacts))
    pos_impacts = impacts == 0
    neg_impacts = impacts == 1
    pos_weights = np.convolve(pos_impacts, linear_decay_kernel)[:len(pos_impacts)]
    neg_weights = np.convolve(neg_impacts, linear_decay_kernel)[:len(neg_impacts)]
    final_weights = pos_weights - neg_weights

    # rprint(pos_weights)
    # rprint(neg_weights)
    # rprint(final_weights)
    # rprint(sigmoid(final_weights))

def test_plotting():
    # with open("./cache/gpt-4o_factscore_analysis_results.json", "r") as f:
    # with open("./cache/gpt-4o_longfact_analysis_results.json", "r") as f:
    # with open("./cache/gpt-4o-mini_factscore_analysis_results.json", "r") as f:
    # with open("./cache/gpt-4o-mini_longfact_analysis_results.json", "r") as f:
    # with open("./cache/gpt-4.1-mini_factscore_analysis_results.json", "r") as f:
    # with open("./cache_4omini_30sample_longfact/gpt-4o-mini_longfact_analysis_results.json", "r") as f:
    # with open("./cache/meta-llama/Llama-3.3-70B-Instruct-Turbo_factscore_analysis_results.json", "r") as f:
    with open("./cache/meta-llama/Llama-3.3-70B-Instruct-Turbo_longfact_analysis_results.json", "r") as f:
    # with open("./cache/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8_factscore_analysis_results.json", "r") as f:
        results = json.load(f)
        rprint(f"number of topics loaded: {len(results.keys())}")

    plotting = Plotting(results, ["self_consistency", "consistency_with_claim", "entropy", "perplexity"])

if __name__ == "__main__":
    # test_llama_evaluator()
    # test_uncertainty_metrics()
    # test_cache_file_manager_to_json()

    # test_auroc()
    # test_cache_file_manager()
    # test_self_consistency()
    # test_build_claim()
    # test_contain_logprobs()
    test_plotting()
    # test_asyncio()
    # asyncio.run(test_graph_based_evaluator())
    # test_longfact_evaluator()
    # generation_viewer()
    # play_ground()
    # test_togetherai()

    print("All tests passed!")

