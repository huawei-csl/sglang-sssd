"""Gathers all benchmark results and compacts them into a single JSON file.
"""

from python.sglang.utils import get_timestamp_str, read_json, save_json
import os
import json

def read_all_json_files(dir_path:str, sorting_key = None) -> list[dict]:
    """
    Reads all JSON files from a specified directory and returns a list of their contents.

    Args:
        dir_path (str): The path to the directory containing the JSON files.
        sort_key (callable, optional): A function to sort the filenames. Defaults to None.

    Returns:
        list: A list containing the parsed JSON data from each file.
    """
    json_data = []
    
    # Check if the directory exists
    if not os.path.exists(dir_path):
        print(f"Directory '{dir_path}' does not exist.")
        return json_data
        
    # Iterate over all files in the directory
    for filename in sorted(os.listdir(dir_path), key=sorting_key):
        if filename.endswith('.json'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    json_data.append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
                    
    return json_data



def gather_results():
    results_dict = {}

    # hyperparameters
    hyperparam_results = read_all_json_files("data/hyperparameter_search", sorting_key=lambda s: int(s.split('_')[1]))
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

