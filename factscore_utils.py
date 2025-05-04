"""All code below is taken directly from https://github.com/shmsw25/FActScore and https://github.com/jiangjmj/Graph-based-Uncertainty, with minor modifications.

If you use this code, please cite:
@inproceedings{ factscore,
    title={ {FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation },
    author={ Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2023 },
    booktitle = { EMNLP },
    url={ https://arxiv.org/abs/2305.14251 }
}
"""

import json
import os
import pickle as pkl
import sqlite3
import string
import time
from collections import defaultdict
from typing import List, Optional, Tuple
import wikipedia

import numpy as np
from tqdm import tqdm
from language_models import OpenAIModel

from rich import print as rprint
import sys

FACTSCORE_CACHE_PATH = './dataset/factscore/'
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, data_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(os.path.join(FACTSCORE_CACHE_PATH, 'enwiki-20230401.db'), check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        if len(cursor.fetchall()) == 0:
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print(f"{self.db_path} is empty. start building DB from {data_path}...")
            # self.build_db(self.db_path, data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, data_path):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text) == str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip()) > 0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset + MAX_LENGTH])
                            offset += MAX_LENGTH

                psgs = [tokenizer.decode(tokens) for tokens in passages if
                        np.sum([t not in [0, 2] for t in tokens]) > 0]
                text = SPECIAL_SEPARATOR.join(psgs)
                
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        self.connection.commit()
        self.connection.close()

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        assert results is not None and len(
            results) == 1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        assert len(results) > 0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results

class RetrievalEasy(object):
    """
    This class is an easy implementation of the original FactScore Retrieval class.
    It fetches the relevant passages directly from the database.
    """
    def __init__(self, db):
        self.db = db

    def get_passages(self, topic):
        passages = self.db.get_text_from_title(topic)
        passage = " ".join([psg["text"] for psg in passages])

        return passage

class Retrieval(object):
    def __init__(self, db, cache_path, embed_cache_path,
                 retrieval_type="gtr-t5-large", batch_size=256):
        self.db = db
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type == "bm25" or retrieval_type.startswith("gtr-")

        self.encoder = None
        self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None

    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}

    def save_cache(self):
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    new_cache = json.load(f)
                self.cache.update(new_cache)

            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)

        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)

            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def get_psg_embeddings(self, topic, passages):
        if self.encoder is None:
            self.load_encoder()
        if topic in self.embed_cache:
            passage_vectors = self.embed_cache[topic]
        else:
            inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
            passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device)
            self.embed_cache[topic] = passage_vectors
            self.add_n_embed += 1
        return passage_vectors

    def get_query_embeddings(self, retrieval_query):
        if isinstance(retrieval_query, str):
            query_vectors = self.encoder.encode([retrieval_query],
                                                batch_size=self.batch_size,
                                                device=self.encoder.device)[0]
        elif isinstance(retrieval_query, list) and len(retrieval_query) > 0:
            query_vectors = self.encoder.encode(retrieval_query,
                                                batch_size=self.batch_size,
                                                device=self.encoder.device) 
        else:
            raise ValueError("Invalid retrieval query")
        return query_vectors

    def get_gtr_passages(self, topic, retrieval_query, passages, k):
        passage_vectors = self.get_psg_embeddings(topic, passages)
        query_vectors = self.get_query_embeddings(retrieval_query)
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores, axis=-1)
        if indices.ndim > 1:
            indices = indices[0, :k]
        elif indices.ndim == 1:
            indices = indices[:k]
        else:
            raise ValueError(f"Invalid indices in {__name__}")

        ranked_passages = [passages[i] for i in indices]
        return ranked_passages

    def get_passages(self, topic, question, k):
        # NOTE: argument is defined as question, but in our use case we pass in a single atomic fact
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query

        if cache_key not in self.cache:
            passages = self.db.get_text_from_title(topic)
            if self.retrieval_type == "bm25":
                self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k)
            else:
                self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)

            self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
            assert len(self.cache[cache_key]) in [k, len(passages)]
            self.add_n += 1

        ranked_passages = self.cache[cache_key]
        ranked_passage = ""
        for psg_idx, psg in enumerate(ranked_passages):
            ranked_passage += "Title: {}\nText: {}\n\n".format(
                psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))

        return ranked_passage

    def get_passages_many_claims(self, topic, claims, k):
        assert isinstance(claims, list)
        retrieval_query = [topic + " " + claim.strip() for claim in claims]
        cache_key = topic + "#" + retrieval_query[0]    # use the first claim as the cache key

        if cache_key not in self.cache:
            passages = self.db.get_text_from_title(topic)
            self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
            assert len(self.cache[cache_key]) in [k, len(passages)]
            self.add_n += 1
        
        ranked_passages = self.cache[cache_key]
        ranked_passage = ""
        for psg_idx, psg in enumerate(ranked_passages):
            ranked_passage += "Title: {}\nText: {}\n\n".format(
                psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
        
        return ranked_passage
    
def get_wiki_passage(person_name): 
    # Set the language of Wikipedia you want to search in, for example, 'en' for English
    wikipedia.set_lang('en')
    try:
        # Get the page for the person
        page = wikipedia.page(person_name)

        # Print the title of the page
        print(f"Title: {page.title}\n")
        
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error, multiple articles found: {e.options}")
    except wikipedia.exceptions.PageError:
        print("Page not found.")

class FactScorer(object):
    def __init__(self,
                 data_dir=FACTSCORE_CACHE_PATH,
                 cache_dir=FACTSCORE_CACHE_PATH,
                 batch_size=256):
        self.db = {}
        self.retrieval = {}
        self.batch_size = batch_size  # batch size for retrieval

        self.data_dir = data_dir
        self.cache_dir = cache_dir

        assert os.path.exists(cache_dir), f"Data directory {cache_dir} does not exist"
        self.eval_model = OpenAIModel(model="gpt-4o")

    def save_cache(self):
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)

    def construct_prompts_with_retrieval(
        self,
        topics: List[str],
        atomic_facts: List[List[str]],
        knowledge_source: Optional[str] = None,
        verbose: bool = True,
        dataset: str = 'factscore'
    ):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        assert len(topics) == len(atomic_facts), "`topics` and `atomic_facts` should have the same length"

        if verbose:
            topics = tqdm(topics)

        prompts = []
        n_prompts_per_topic = []
        data_to_return = defaultdict(list)

        for topic, facts in zip(topics, atomic_facts):
            data_to_return['entity'].append(topic)
            data_to_return['generated_answer'].append(facts)
            n_prompts = 0
            answer_interpretation_prompts = []
            for atom in facts:
                atom = atom.strip()
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
                assert dataset == 'factscore', "Only factscore dataset is supported for now"
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(
                        psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                prompt = "\n\n{}\n\nInput: {}.\nQuestion: Is the input Subjective or Objective? If objective, is the input True or False? Format your answer as one word as 'ANSWER: <Subjectve/True/False>'\nOutput:".format(definition.strip(), atom.strip())

                # Flattened list of prompts
                prompts.append(prompt)

                answer_interpretation_prompts.append(prompt)

                # But track how many there are per topic/entity
                n_prompts += 1

            n_prompts_per_topic.append(n_prompts)
            data_to_return['answer_interpretation_prompt'].append(answer_interpretation_prompts)

        return prompts, n_prompts_per_topic, data_to_return
    
    def fact_check_with_gpt(self,  topics: List[str], atomic_facts: List[List[str]], dataset='factscore'):
        mark_dict = {'true': 'Y', 'false': 'N', 'subjective': 'S', 'objective': ''}
        marks, raw_results = [], []
        prompts, _, data_to_return = self.construct_prompts_with_retrieval(topics=topics, atomic_facts=atomic_facts, dataset=dataset, verbose=False)
        print(atomic_facts)
        print()
        print(f"n_prompts_per_topic: {_}")
        for prompt in prompts:
            response_content = self.eval_model.query(prompt)
            mark = mark_dict[response_content.split('ANSWER:')[1].strip().lower()]
            marks.append(mark)
            raw_results.append({'return': response_content, 'prompt': prompt})
        return marks, raw_results