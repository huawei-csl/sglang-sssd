"""Runs the main evaluation benchmark for Autoregressive, SSSD and EAGLE3"""

import argparse
import logging
import os
import multiprocessing as mp

from sglang.bench_offline_throughput import BenchArgs, throughput_test
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.sssd_utils import SSSDSpeculator
from sglang.utils import get_timestamp_str, read_json, save_json

# Default mapping for SSSD hyperparameters
# key: batch_size
default_SSSD_mapping = {
    1: {
        "speculative_eagle_topk": 5,
        "speculative_num_draft_tokens": 32,
        "speculative_num_steps": SSSDSpeculator._default_branch_func(32),
    },
    4: {
        "speculative_eagle_topk": 5,
        "speculative_num_draft_tokens": 22,
        "speculative_num_steps": SSSDSpeculator._default_branch_func(22),
    },
    8: {
        "speculative_eagle_topk": 5,
        "speculative_num_draft_tokens": 16,
        "speculative_num_steps": SSSDSpeculator._default_branch_func(16),
    },
    16: {
        "speculative_eagle_topk": 5,
        "speculative_num_draft_tokens": 8,
        "speculative_num_steps": SSSDSpeculator._default_branch_func(8),
    },
    32: {
        "speculative_eagle_topk": 5,
        "speculative_num_draft_tokens": 4,
        "speculative_num_steps": SSSDSpeculator._default_branch_func(4),
    },
    48: {
        "speculative_eagle_topk": 5,
        "speculative_num_draft_tokens": 3,
        "speculative_num_steps": SSSDSpeculator._default_branch_func(3),
    },
    64: {
        "speculative_eagle_topk": 5,
        "speculative_num_draft_tokens": 2,
        "speculative_num_steps": SSSDSpeculator._default_branch_func(2),
    },
    128: {
        "speculative_eagle_topk": 5,
        "speculative_num_draft_tokens": 1,
        "speculative_num_steps": SSSDSpeculator._default_branch_func(1),
    },
}


def add_extra_run_args(parser: argparse.ArgumentParser):
    """Adds arguments for the run"""
    parser.add_argument(
        "--hparam-file",
        type=str,
        required=False,
        help="Directory to save the results JSON file.",
    )
    parser.add_argument(
        "--use-hparam-mapping",
        action="store_true",
        help="Use our default mapping for hyperparameters.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of samples to take for each trial in the hyperparameter search. (TODO: set to 3)",
    )


def main(server_args: ServerArgs, bench_args: BenchArgs, args: argparse.Namespace):

    mp.set_start_method("spawn", force=True)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    if os.path.exists(bench_args.result_filename):
        algorithm = server_args.speculative_algorithm or "autoregressive"
        print(
            f"Results for {algorithm} with batch size {server_args.max_running_requests} already exist. Skipping..."
        )
        return

    if args.use_hparam_mapping and args.hparam_file:
        raise ValueError(
            "Cannot use both --use-hparam-mapping and --hparam-file at the same time."
        )

    if args.use_hparam_mapping:
        assert (
            server_args.speculative_algorithm == "SSSD"
        ), "Hyperparameter mapping is only available for SSSD."
        chosen_hparams = default_SSSD_mapping[server_args.max_running_requests]
        for k, v in chosen_hparams.items():
            setattr(server_args, k, v)

    if args.hparam_file:
        hparam_results = read_json(args.hparam_file)
        chosen_hparams = hparam_results["best_parameters"]
        for k, v in chosen_hparams.items():
            setattr(server_args, k, v)

    # Once on a rare occasion (<5%) an evaluation can fail due to a CUDA error.
    # We retry the evaluation to mitigate transient errors like this.
    max_attempts = args.sample_size * 2
    results = []
    final_output_path = bench_args.result_filename
    for attempt in range(max_attempts):
        print("Running with the following parameters:")
        print(
            f"speculative_num_draft_tokens: {server_args.speculative_num_draft_tokens}"
        )
        print(f"speculative_eagle_topk: {server_args.speculative_eagle_topk}")
        print(f"speculative_num_steps: {server_args.speculative_num_steps}")
        try:
            # Set up output directory for intermediate results
            result_dir = "/tmp/sssd"
            result_path = os.path.join(result_dir, "evaluation_result.json")
            os.makedirs(result_dir, exist_ok=True)
            bench_args.result_filename = result_path
            if os.path.exists(result_path):
                os.remove(result_path)

            # Run the evaluation
            print(f"Running evaluation {len(results) + 1}/{args.sample_size}...")
            p = mp.Process(target=throughput_test, args=(server_args, bench_args))
            p.start()
            p.join()

            result = read_json(result_path)
            results.append(result)

            if len(results) == args.sample_size:
                # Pick result with median latency
                results.sort(key=lambda x: x["total_latency"])
                print(f"latencies: {[r['total_latency'] for r in results]}")
                final_result = results[len(results) // 2]
                final_result["timestamp"] = get_timestamp_str()
                final_result["eval_args"] = {
                    "speculative_num_draft_tokens": server_args.speculative_num_draft_tokens,
                    "speculative_eagle_topk": server_args.speculative_eagle_topk,
                    "speculative_num_steps": server_args.speculative_num_steps,
                }

                save_json(final_output_path, final_result)
                break
        except Exception as e:
            # Handle runtime errors (e.g., CUDA out of memory).
            # Returning a large value tells Optuna this was a bad trial.
            print(f"Run failed with an error: {e}")
            print(f"{max_attempts - attempt - 1} attempts remaining...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    add_extra_run_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)
    main(server_args, bench_args, args)
