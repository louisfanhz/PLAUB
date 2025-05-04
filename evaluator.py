import os
import asyncio
import re
import numpy as np
import networkx as nx
import json
import wikipedia
from tqdm import tqdm
from enum import Enum
from json.decoder import JSONDecodeError
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import pickle

import configs
from prompts import evaluator_prompts
from utils import retry_if_unsuccessful
from language_models import OpenAIModel, LlamaModel
from factscore_utils import DocDB, RetrievalEasy, Retrieval

from rich import print as rprint
import sys

class CorrectnessEvaluator(object):

    @retry_if_unsuccessful(max_retries=1)
    async def _evaluate_claims(self, topic: str, reference: str, claims: List[str]):
        sys_prompt = evaluator_prompts["eval_claims_from_reference_system_prompt"]
        claims_formatted = "\n".join([f"claim {i+1}: {claim.strip()}" for i, claim in enumerate(claims)])
        usr_prompt = evaluator_prompts["eval_claims_from_reference_user_prompt"].format(topic=topic, 
                                                                                        reference=reference, 
                                                                                        claims=claims_formatted)

        class Label(str, Enum):
            correct = 'correct'
            incorrect = 'incorrect'
            not_enough_information = 'not_enough_information'

        class Correctness(BaseModel):
            claim_index: int = Field(description="The index of the claim being evaluated.")
            correctness: Label = Field(description="The correctness of the claim.")
            
        class CorrectnessList(BaseModel):
            correctness_list: List[Correctness] = Field(description="A list of correctness evaluations for the claims.")

        response = await self.eval_model.query_structured(sys_prompt, usr_prompt, spec=CorrectnessList)
        corr_list = response.correctness_list
        assert len(corr_list) == len(claims), f"Number of eval results does not equal number of claims: {len(corr_list)} != {len(claims)} \n\n {claims} \n\n {corr_list}"
        for i in range(len(corr_list)):
            assert corr_list[i].claim_index == i + 1, f"Evaluation result {i} should correspond to claim {i + 1}. Check LLM returned response."

        return [corr_list[i].correctness.value for i in range(len(corr_list))]

    @retry_if_unsuccessful(max_retries=1)
    async def _evaluate_claim_single(self, reference: str, claim: str):
        sys_prompt = evaluator_prompts["eval_claims_from_reference_system_prompt"]
        usr_prompt = evaluator_prompts["eval_claims_from_reference_user_prompt_single"].format(reference=reference, claim=claim)

        # class Label(str, Enum):
        #     correct = 'correct'
        #     incorrect = 'incorrect'
        #     irrelevant = 'irrelevant'

        # class Correctness(BaseModel):
        #     correctness: Label = Field(description="The correctness of the claim.")

        # response = await self.eval_model.query_structured(sys_prompt, usr_prompt, spec=Correctness)
        
        # return response.correctness.value

        ### second implementation
        class Label(str, Enum):
            supported = 'supported'
            not_supported = 'not supported'

        class Correctness(BaseModel):
            correctness: Label = Field(description="The correctness of the claim.")

        response = await self.eval_model.query_structured(sys_prompt, usr_prompt, spec=Correctness)

        corr_value = response.correctness.value
        assert corr_value in ["supported", "not supported"], f"Invalid response in {CorrectnessEvaluator.__name__}: {corr_value}"
        return "correct" if corr_value == "supported" else "incorrect"


class LongFactEvaluator(CorrectnessEvaluator):
    def __init__(self, 
                dataset_path: str, 
                eval_model: str="gpt-4o", 
                text_encoder: str="all-MiniLM-L6-v2") -> None:
        wikipedia.set_lang('en')
        self.eval_model = OpenAIModel(eval_model)
        self.text_encoder = SentenceTransformer("sentence-transformers/" + text_encoder).cuda().eval()
        self.text_chunks = {}
        self.text_embeddings = {}

        with open(dataset_path, "r") as f:
            longfact_dict = json.load(f)
            dataset = [{"topic": entry["prompt"], "wiki_entity": entry["wiki_entity"]} for longfact_file in longfact_dict.values() 
                                                                            for entry in longfact_file]
        for entry in tqdm(dataset, desc="Initializing longfact dataset"):
            wiki_page = self._get_wiki_page(entry["wiki_entity"])
            # wiki_page = wikipedia.search("Solomon R. Guggenheim Museum")
            self._make_text_embeddings(entry["topic"], entry["wiki_entity"], wiki_page)

    def _get_wiki_page(self, entity_name):
        try:
            page = wikipedia.page(entity_name, auto_suggest=False)
            return page.content
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error, multiple articles found: {e.options}")
        except wikipedia.exceptions.PageError:
            print(f"Page not found for {entity_name}")

    def _make_text_embeddings(self, topic: str, wiki_title: str, text: str) -> np.ndarray:
        try:
            chunks = self._create_chunks(wiki_title, text)
        except Exception as e:
            print(f"Error creating chunks for {wiki_title}: {e}")
            raise e
        self.text_chunks[topic] = chunks
        
        embeddings = self.text_encoder.encode(chunks, convert_to_numpy=True, device=self.text_encoder.device)
        self.text_embeddings[topic] = embeddings

    def _create_chunks(self, wiki_title: str, text: str, max_chunk_size: int = 256) -> List[str]:
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
        
            # If adding this sentence would exceed max_chunk_size, create a new chunk
            if current_chunk and current_word_count + sentence_words > max_chunk_size:
                chunk_text = wiki_title + ": " + " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_word_count = 0

            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = wiki_title + ": " + " ".join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks

    def get_query_embeddings(self, retrieval_query):
        if isinstance(retrieval_query, str):
            query_vectors = self.text_encoder.encode([retrieval_query],
                                                    convert_to_numpy=True,
                                                    device=self.text_encoder.device)[0]
        elif isinstance(retrieval_query, list) and len(retrieval_query) > 0:
            query_vectors = self.text_encoder.encode(retrieval_query,
                                                    convert_to_numpy=True,
                                                    device=self.text_encoder.device) 
        else:
            raise ValueError("Invalid retrieval query")
        return query_vectors

    def retrieve_relevant_passages(self, topic: str, query: Union[str, List[str]], k: int = 5) -> str:
        if topic not in self.text_embeddings:
            raise ValueError(f"{topic} should be in the LongFactEvaluator dataset, but is not.")
            
        # query_embedding = self.text_encoder.encode(query, convert_to_numpy=True, device=self.text_encoder.device)
        query_embeddings = self.get_query_embeddings(query)
        passage_embeddings = self.text_embeddings[topic]
        scores = np.inner(query_embeddings, passage_embeddings)
        indices = np.argsort(-scores, axis=-1)
        # indices = np.argsort(-scores, axis=-1)[:k]
        if indices.ndim > 1:
            # In most cases with batched claims evaluation, the ranked passages for multiple claims contain same passages in k<5
            indices = indices[0, :k]
        elif indices.ndim == 1:
            indices = indices[:k]

        ranked_passages = [self.text_chunks[topic][i] for i in indices]
        return "\n".join(ranked_passages)

    async def evaluate_claims(self, topic: str, claims: List[str], one_by_one: bool = False, batch_size: int = 10):
        # reference = self.retrieve_relevant_passages(topic, claims, k=configs.ref_doc_retrieval_k)

        if not one_by_one:
            eval_tasks = []
            for i in range(0, len(claims), batch_size):
                batched_claims = claims[i:i+batch_size]
                reference = self.retrieve_relevant_passages(topic, batched_claims, k=configs.ref_doc_retrieval_k)
                eval_tasks.append(asyncio.create_task(self._evaluate_claims(topic, reference, batched_claims)))
            correctness_lists = await asyncio.gather(*eval_tasks)
            correctness_list = [item for sublist in correctness_lists for item in sublist]
        else:
            # reference = self.retrieve_relevant_passages(topic, claims, k=configs.ref_doc_retrieval_k)
            eval_tasks = []
            for claim in claims:
                reference = self.retrieve_relevant_passages(topic, claim, k=configs.ref_doc_retrieval_k)
                eval_tasks.append(asyncio.create_task(self._evaluate_claim_single(reference, claim)))
            correctness_list = await asyncio.gather(*eval_tasks)

        assert len(correctness_list) == len(claims), f"Number of eval results not equal number of claims! {correctness_list}\n\n{claims}"

        return correctness_list

class FactScoreEvaluator(CorrectnessEvaluator):
    def __init__(self, 
                 eval_model: str="gpt-4o",
                 db_path: str="./dataset/factscore/enwiki-20230401.db",
                 retrieval_type: str="gtr"):
        self.eval_model = OpenAIModel(model=eval_model)

        self.retrieval_type = retrieval_type
        if self.retrieval_type == "easy":
            self.retrieval = RetrievalEasy(DocDB(db_path))
        else:
            dir_path = os.path.dirname(db_path)
            cache_path = os.path.join(dir_path, "retrieval-cache.json")
            embed_cache_path = os.path.join(dir_path, "retrieval-embed-cache.pkl")
            self.retrieval = Retrieval(DocDB(db_path), cache_path, embed_cache_path)

    def count_words(passage: str) -> int:
        try:
            from nltk.tokenize import word_tokenize
        except ImportError:
            # Download the necessary NLTK data if not present
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import word_tokenize
            
        words = word_tokenize(passage)
        return len(words)

    async def evaluate_claims(self, topic: str, claims: List[str], one_by_one: bool = False, batch_size: int = 10):
        # if self.retrieval_type == "easy":
        #     reference = self.retrieval.get_passages(topic)
        # else:
        #     reference = self.retrieval.get_passages_many_claims(topic, claims, k=configs.ref_doc_retrieval_k)

        if not one_by_one:
            eval_tasks = []
            for i in range(0, len(claims), batch_size):
                batched_claims = claims[i:i+batch_size]
                reference = self.retrieval.get_passages_many_claims(topic, batched_claims, k=configs.ref_doc_retrieval_k)
                eval_tasks.append(asyncio.create_task(self._evaluate_claims(topic, reference, batched_claims)))
            correctness_lists = await asyncio.gather(*eval_tasks)
            correctness_list = [item for sublist in correctness_lists for item in sublist]
        else:
            # if not self.retrieval_type == "easy":
            #     reference = self.retrieval.get_passages_many_claims(topic, claims, k=configs.ref_doc_retrieval_k)
            eval_tasks = []
            for claim in claims:
                reference = self.retrieval.get_passages(topic, claim, k=configs.ref_doc_retrieval_k)
                eval_tasks.append(asyncio.create_task(self._evaluate_claim_single(reference, claim)))
            correctness_list = await asyncio.gather(*eval_tasks)

        assert len(correctness_list) == len(claims), f"Number of eval results not equal number of claims! {correctness_list}\n\n{claims}"

        return correctness_list

class SelfConsistencyEvaluator(object):
    def __init__(self, model):
        self.model = model

    async def evaluate_claims(self, passages: List[str], claims: List[str]):
        n_psg = len(passages)
        eval_tasks = [asyncio.create_task(self._evaluate(passage, claim)) for claim in claims for passage in passages]
        is_supported_list = await asyncio.gather(*eval_tasks)

        claims_supp_scores = [np.mean(is_supported_list[i*n_psg:(i+1)*n_psg]).item() for i in range(len(claims))]

        return claims_supp_scores

    @retry_if_unsuccessful(max_retries=1)
    async def _evaluate(self, passage: str, claim: str):
        sys_prompt = evaluator_prompts["from_generations_system_prompt"]
        usr_prompt = evaluator_prompts["from_generations_user_prompt"].format(passage=passage, claim=claim)

        class IsSupported(BaseModel):
            is_supported: bool = Field(description="is the claim supported by the passage")

        response = await self.model.query_structured(sys_prompt, usr_prompt, spec=IsSupported)

        is_supported = response.is_supported
        return is_supported

class GraphBasedEvaluator(object):
    # The following code is adapted from https://github.com/jiangjmj/Graph-based-Uncertainty with minor modifications

    VC_OPTIONS = {'nochance': 0, 'littlechance': 0.2, 'lessthaneven': 0.4, 'fairlypossible': 0.6, 'verygoodchance': 0.8, 'almostcertain': 1.0}
    CENTRALITY_TYPE_LIST = ['eigenvector_centrality', 'betweenness_centrality', 'closeness_centrality', 'pagerank', 'closeness_centrality_with_node_confidence']

    def __init__(self, model):
        self.model = model
        self.sc_evaluator = SelfConsistencyEvaluator(model=model)

    async def compute_centrality(self, topic: str, passages: List[str], all_claims: List[str]):
        vc_eval_tasks = [asyncio.create_task(self.get_verbalized_confidence(topic, claim)) for claim in all_claims]
        vcs = await asyncio.gather(*vc_eval_tasks)
        edges = await self.compute_edges(passages, all_claims)

        centrality_scores_vcwo = self.calculate_bg_centrality(lsts=edges, length=len(all_claims), vc_lst=vcs)

        return centrality_scores_vcwo

    async def compute_edges(self, passages: List[str], claims: List[str]):
        n_clms = len(claims)
        eval_tasks = [asyncio.create_task(self.sc_evaluator._evaluate(passage, claim)) for passage in passages for claim in claims]
        is_supported_list = await asyncio.gather(*eval_tasks)

        claims_supp_result = [is_supported_list[i*n_clms:(i+1)*n_clms] for i in range(len(passages))]

        return claims_supp_result

    async def get_verbalized_confidence(self, question, claim, problem_type='qa', with_options=True):
        if problem_type == 'qa':
            prompt = f'You are provided with a question and a possible answer. Provide the probability that the possible answer is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\nProbability: <the probability that your guess is correct as a percentage, without any extra commentary whatsoever; just the probability!>\n\nThe question is: {question}\nThe possible answer is: {claim}'
        elif problem_type == 'fact':
            prompt = f'You are provided with some possible information about a person. Provide the probability that the information is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\nProbability: <the probability that your guess is correct as a percentage, without any extra commentary whatsoever; just the probability!>\n\nThe person is: {question}\nThe possible information is: {claim}'
        
        if with_options:
            if problem_type == 'qa':
                prompt = f'You are provided with a question and a possible answer. Describe how likely it is that the possible answer is correct as one of the following expressions:\nNo chance (0%)\nLittle chance  (20%)\nLess than even (40%)\nFairly possible (60%)\nVery good chance (80%)\nAlmost certain (100%)\n\nGive ONLY your confidence phrase, no other words or explanation. For example:\n\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>\n\nThe question is: {question}\nThe possible answer is: {claim}'
            elif problem_type == 'fact':
                prompt = f'You are provided with some possible information about a person. Describe how likely it is that the possible answer is correct as one of the following expressions:\nNo chance (0%)\nLittle chance  (20%)\nLess than even (40%)\nFairly possible (60%)\nVery good chance (80%)\nAlmost certain (100%)\n\nGive ONLY your confidence phrase, no other words or explanation. For example:\n\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>\n\nThe person is: {question}\nThe possible information is: {claim}'
                
            results = await self.model.query(prompt)

        confidence = self.parse_confidence(results, with_options=with_options)

        return confidence

    def parse_confidence(self, input_string, with_options=False):
        """
        Parses the input string to find a percentage or a float between 0.0 and 1.0 within the text.
        If a percentage is found, it is converted to a float.
        If a float between 0.0 and 1.0 is found, it is also converted to a float.
        In other cases, returns -1.

        :param input_string: str, the string to be parsed.
        :return: float, the parsed number or -1 if no valid number is found.
        """
        if with_options:
            split_list = input_string.split(':')
            if len(split_list) > 1:
                input_string = split_list[1]
            only_alpha = re.sub(r'[^a-zA-Z]', '', input_string).lower()
            if only_alpha in self.VC_OPTIONS:
                return self.VC_OPTIONS[only_alpha]
        else:
            # Search for a percentage in the text
            percentage_match = re.search(r'(\d+(\.\d+)?)%', input_string)
            if percentage_match:
                return float(percentage_match.group(1)) / 100

            # Search for a float between 0.0 and 1.0 in the text
            float_match = re.search(r'\b0(\.\d+)?\b|\b1(\.0+)?\b', input_string)
            if float_match:
                return float(float_match.group(0))

        # If neither is found, return -1
        return -1

    def calculate_bg_centrality(self, lsts, length, vc_lst=None):
        centrality_dict = {}
        for centrality_name in self.CENTRALITY_TYPE_LIST:
            centrality_dict[centrality_name] = np.ones(length) * -1    
        
        filtered_lists = np.array(lsts)
        gen_num, flitered_breakdown_len = np.shape(filtered_lists)[0], np.shape(filtered_lists)[1]
        adjacency_matrix = np.zeros((gen_num + flitered_breakdown_len, gen_num + flitered_breakdown_len))
        adjacency_matrix[:gen_num, gen_num:] = filtered_lists
        adjacency_matrix[gen_num:, :gen_num] = filtered_lists.T
        G = nx.from_numpy_array(adjacency_matrix)
        if vc_lst is not None:
            combined_c = self.bg_closeness_centrality_with_node_confidence(G, gen_num, vc_lst)

        centrality = np.ones(length) * -1 
        for function_name in centrality_dict:
            try:
                if function_name in ['eigenvector_centrality', 'pagerank']:
                    centrality = getattr(nx, function_name)(G, max_iter=5000)
                elif function_name == 'closeness_centrality_with_node_confidence' and vc_lst is not None:
                    centrality = combined_c
                else:
                    centrality = getattr(nx, function_name)(G)
            except:
                pass

            assert length == flitered_breakdown_len
            centrality = [centrality[i] for i in sorted(G.nodes())] # List of scores in the order of nodes
            centrality_dict[function_name] = centrality[gen_num:]
                    
        return centrality_dict

    def bg_closeness_centrality_with_node_confidence(self, G, gen_num, verb_conf):
        n = len(G)
        combined_centrality = np.ones(len(G)) * -1
        
        for i in range(gen_num, n):
            sum_shortest_path_to_gens, sum_vc, sum_shortest_path_to_gens_vc_unweighted, sum_shortest_path_to_gens_vc_product = 0, 0, 0, 0
            reachable = 0
            
            for gen_id in range(n):
                if i == gen_id:
                    continue
                try:
                    shortest_path = nx.shortest_path(G, source=i, target=gen_id)
                    shortest_path = [node for node in shortest_path if node >= gen_num]
                    verb_conf_sum = np.sum([1 - verb_conf[node - gen_num] for node in shortest_path])
                                    
                    path_length = nx.shortest_path_length(G, source=i, target=gen_id)
                    
                    sum_vc += verb_conf_sum
                    sum_shortest_path_to_gens += path_length + verb_conf_sum
                    
                    sum_shortest_path_to_gens_vc_unweighted += path_length + np.sum([1 - verb_conf[node - gen_num] for node in shortest_path[1:]])
                    reachable += 1
                except nx.NetworkXNoPath:
                    continue
            
            scaling_factor = reachable / (n - 1)
            combined_centrality[i] = scaling_factor * reachable / sum_shortest_path_to_gens if reachable > 0 else 0

        return combined_centrality
