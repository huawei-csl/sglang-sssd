"""Gathers all benchmark results and compacts them into a single JSON file.
"""

import re
from python.sglang.utils import get_timestamp_str, read_json, save_json
import os
import json

def read_all_json_files(dir_path:str, sorting_fn=None) -> tuple[list[dict], list[str]]:
    """
    Reads all JSON files from a specified directory and returns a list of their contents.

    Args:
        dir_path (str): The path to the directory containing the JSON files.

    Returns:
        list: A list containing the parsed JSON data from each file.
    """
    json_data = []
    filenames = sorted(os.listdir(dir_path), key=sorting_fn) if sorting_fn else []
        
    # Iterate over all files in the directory
    for filename in filenames:
        if filename.endswith('.json'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    json_data.append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
                    
    return json_data, filenames


def get_name_and_batch_size(filename: str) -> tuple[str, int]:
    m = re.match(r'([a-zA-Z0-9\-]+)_([0-9]+)(?:_.*)?', filename)
    base = m.group(1)
    batch_size = int(m.group(2))
    
    return (base, batch_size)

def process_evaluation_results(eval_results: list[dict], eval_filenames: list[str]) -> dict:
    results = {}
    for filename, result in zip(eval_filenames, eval_results):
        method_name, batch_size = get_name_and_batch_size(filename)
        if method_name not in results:
            results[method_name] = {}
        results[method_name][batch_size] = result
    return results

def gather_results():
    results_dict = {}
    benchmark = "mt-bench"

    # evaluation results
    eval_results, eval_filenames = read_all_json_files(f"data/evaluation/{benchmark}", get_name_and_batch_size)
    results_dict[f"evaluation_{benchmark}"] = process_evaluation_results(eval_results, eval_filenames)

    # hyperparameters
    hyperparam_results, _ = read_all_json_files("data/hyperparameter_search", get_name_and_batch_size)
    hyperparam_d = {}
    for r in hyperparam_results:
        alg = r.pop("algorithm")
        batch = f"batch_{r.pop('batch_size')}"
        if alg not in hyperparam_d:
            hyperparam_d[alg] = {}
        hyperparam_d[alg][batch] = r
    results_dict["hyperparameter_search"] = hyperparam_d

    # machine specs
    results_dict["machine_specs"] = read_json("data/machine_specs.json"),

    # saving
    results_dict["submission_timestamp"] = get_timestamp_str()
    output_path = 'data/collected_results.json'
    save_json(output_path, results_dict, sort_keys=False)
    print(f"All Results saved to {output_path}.")


if __name__ == "__main__":
    gather_results()

