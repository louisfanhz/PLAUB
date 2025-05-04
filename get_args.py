import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-to-evaluate", type=str, default="meta-llama/llama-3-70b-instruct")
    parser.add_argument("--num-topics", type=int, default=3)
    parser.add_argument("--num-res-per-question", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="factscore")
    parser.add_argument("--max-concurrent-tasks", type=int, default=10)
    parser.add_argument("--compute-baselines", type=bool, default=False)
    parser.add_argument("--load-cached-generations", type=bool, default=False)
    parser.add_argument("--load-cached-analysis-results", type=bool, default=False)

    args, _ = parser.parse_known_args()

    return args