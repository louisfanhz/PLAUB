import os
import asyncio
import json
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from language_models import OpenAIModel, TogetherAIModel
from generator import Generation
from interrogator import Interrogator
from responder import Responder
from evaluator import FactScoreEvaluator, SelfConsistencyEvaluator, GraphBasedEvaluator
from uncertainty_metrics import UncertaintyMetrics
from result_formats import TopicResult, GenSampleResult, ClaimAnalysis
from utils import CacheFileManager, cancel_if_error, Constants
from get_args import get_args
import configs

from rich import print as rprint
import sys

async def generate(topic: str, prompt: str):
    most_likely_generation = await generator.generate_most_likely(prompt)
    if most_likely_generation == Constants.INVALID_GENERATION:
        return topic
    diverse_generations = await generator.generate_diverse(prompt)
    if diverse_generations == Constants.INVALID_GENERATION:
        return topic

    generations[topic] = {}
    generations[topic]["generation_prompt"] = prompt
    generations[topic]["most_likely_generation"] = most_likely_generation
    generations[topic]["diverse_generations"] = diverse_generations

    generations.sync()

@cancel_if_error
async def interrogate(topic, generation):
    sc_evaluator = SelfConsistencyEvaluator(model=llm)
    context = generation["generation_prompt"]
    generation_samples = generation["diverse_generations"]

    result = TopicResult()
    for gen_idx, gen in enumerate(generation_samples):
        atomic_claims = await interrogator.extract_atomic_claims(topic, context, gen)
        supported_scores = await sc_evaluator.evaluate_claims(generation_samples, atomic_claims)
        for claim, score in zip(atomic_claims, supported_scores):
            claim.supported_score = score
        question_lists = await interrogator.raise_questions_from_claims_single(context, atomic_claims, num_q_per_claim=3)

        for i, claim in enumerate(atomic_claims):
            claim.claim_analysis = [ClaimAnalysis(question=question_lists[i][j]) for j in range(len(question_lists[i]))]

        result.gen_analysis.append(GenSampleResult(gen_idx=gen_idx,
                                                all_claims=[c.content for c in atomic_claims], 
                                                all_questions=[q for q_list in question_lists for q in q_list],
                                                claims=atomic_claims))

    analysis_results[topic] = result.model_dump()
    analysis_results.sync()

@cancel_if_error
async def respond(topic: str, topic_res: TopicResult, generation: dict, num_res_per_question: int):
    topic_context = generation["generation_prompt"]
    # gen_context = generation["diverse_generations"][0]

    for ga in topic_res.gen_analysis:
        for claim_idx, claim in enumerate(ga.claims):
            for ca in claim.claim_analysis:
                answers = await responder.respond(topic=topic,
                                                topic_context=topic_context,
                                                topic_generations=generation["diverse_generations"],
                                                all_claims=ga.all_claims,
                                                claim=claim.content,
                                                claim_idx=claim_idx,
                                                question=ca.question,
                                                n_res_per_question=num_res_per_question)
                ca.answers = answers

    analysis_results[topic] = topic_res.model_dump()
    analysis_results.sync()

    return topic_res

async def process_all_topics(dataset_name: str, 
                            num_topics: int, 
                            num_res_per_question: int,
                            max_concurrent_tasks: int=10, 
                            compute_baselines: bool=False):
    if dataset_name == "factscore":
        from dataset.parse_factscore import generate_dataset
        dataset = generate_dataset(db_path=configs.factscore_db_path,
                                    prompt_entities_path=configs.factscore_prompt_entities_path,
                                    save_copy=False).shuffle(seed=configs.seed).select(range(num_topics))
    elif dataset_name == "longfact":
        from dataset.parse_longfact import generate_dataset
        # dataset = generate_dataset(data_path=configs.longfact_data_path,
        #                             evenly_select_from_topics=True).shuffle(seed=configs.seed).select(range(num_topics))
        dataset = generate_dataset(data_path=configs.longfact_data_path).shuffle(seed=configs.seed).select(range(num_topics))
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

    all_topics_to_process = [sample["topic"] for sample in dataset]
    print("Topics to process: ", all_topics_to_process)

    generate_tasks = []
    topics_unable_to_generate = []
    for sample in dataset:
        if not sample["topic"] in generations.cache:
            generate_tasks.append(asyncio.create_task(generate(sample["topic"], sample["prompt_text"])))
    for idx in tqdm(range(0, len(generate_tasks), max_concurrent_tasks), desc=f"Running Generation Process at batch_size={max_concurrent_tasks}"):
        tasks = generate_tasks[idx:idx+max_concurrent_tasks]
        topics_unable_to_generate.extend(await atqdm.gather(*tasks))
    topics_unable_to_generate = [topic for topic in topics_unable_to_generate if topic]
    
    print(f"{llm.name} incapable of generating the following topics: {topics_unable_to_generate}")

    generations.to_json()

    interrogate_tasks = []
    for topic in generations.cache.keys():
        if topic not in analysis_results.cache:
            interrogate_tasks.append(asyncio.create_task(interrogate(topic, generations[topic])))
    for idx in tqdm(range(0, len(interrogate_tasks), max_concurrent_tasks), desc=f"Running Interrogation Process at batch_size={max_concurrent_tasks}"):
        tasks = interrogate_tasks[idx:idx+max_concurrent_tasks]
        await atqdm.gather(*tasks)

    analysis_results.to_json()

    respond_tasks = []
    for topic, topic_res in analysis_results.cache.items():
        topic_res = TopicResult(**topic_res)
        if not topic_res.is_populated():
            respond_tasks.append(asyncio.create_task(respond(topic, topic_res, generations[topic], num_res_per_question)))
    for idx in tqdm(range(0, len(respond_tasks), max_concurrent_tasks), desc=f"Running Respond Process at batch_size={max_concurrent_tasks}"):
        tasks = respond_tasks[idx:idx+max_concurrent_tasks]
        await atqdm.gather(*tasks)

    analysis_results.to_json()

    if compute_baselines:
        # if os.path.exists(f"./cache/{llm.name}_{dataset_name}_graph_based_results.json"):
        #     baseline_results = json.load(open(f"./cache/{llm.name}_{dataset_name}_graph_based_results.json"))

        g_evaluator = GraphBasedEvaluator(model=llm)
        for topic, topic_res in tqdm(analysis_results.cache.items(), desc="Running Graph-based Evaluation"):
            if topic in baseline_results.cache.keys():
                continue
            topic_res = TopicResult(**topic_res)
            g_results = {gen_idx: [] for gen_idx in range(len(topic_res.gen_analysis))}
            for ga in topic_res.gen_analysis:
                all_claims = ga.gather_claim_contents()
                all_gens = [gen for gen in [generations[topic]["most_likely_generation"]] + generations[topic]["diverse_generations"]
                            if gen != Constants.INVALID_GENERATION]
                g_results[ga.gen_idx] = await g_evaluator.compute_centrality(topic,
                                                                            all_gens, 
                                                                            all_claims)
            baseline_results[topic] = g_results
            baseline_results.sync()

        baseline_results.to_json()

        # with open(f"./cache/{llm.name}_{dataset_name}_graph_based_results.json", "w") as f:
        #     json.dump(g_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    global llm, generator, interrogator, responder, generations, analysis_results

    args = get_args()
    # topics = [topic for topic in generations.cache.keys()][:80]

    if "gpt" in args.model_to_evaluate:
        llm = OpenAIModel(model=args.model_to_evaluate, requests_per_minute=1000)
    else:
        llm = TogetherAIModel(model=args.model_to_evaluate, requests_per_minute=500)

    generations_cache_path = f"./cache/{llm.name}_{args.dataset}_generations"
    analysis_results_cache_path = f"./cache/{llm.name}_{args.dataset}_analysis_results"
    baseline_results_cache_path = f"./cache/{llm.name}_{args.dataset}_baseline_results"

    generator = Generation(model=llm)
    interrogator = Interrogator(
        model=llm,
        dataset_name=args.dataset
    )
    responder = Responder(
        model=llm,
        uncertainty_metrics=UncertaintyMetrics(),
    )
    # generations = CacheFileManager(cache_path="./cache/llama-3-70b-instruct_factscore", 
    #                                from_jsonl="./results/llama-3-70b-instruct_factscore.jsonl")
    generations = CacheFileManager(cache_path=generations_cache_path)
    analysis_results = CacheFileManager(cache_path=analysis_results_cache_path)
    baseline_results = CacheFileManager(cache_path=baseline_results_cache_path)

    print(f"\nProcessing {args.dataset} dataset with {llm.name}...\n\n")
    asyncio.run(process_all_topics(dataset_name=args.dataset, 
                                    num_topics=args.num_topics, 
                                    num_res_per_question=args.num_res_per_question,
                                    max_concurrent_tasks=args.max_concurrent_tasks, 
                                    compute_baselines=args.compute_baselines))

