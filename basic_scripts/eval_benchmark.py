"""Runs the main evaluation benchmark for Autoregressive, SSSD and EAGLE3"""

import argparse
import os

from python.sglang.bench_offline_throughput import BenchArgs, throughput_test
from python.sglang.srt.server_args import ServerArgs
from python.sglang.srt.speculative.sssd_utils import SSSDSpeculator
from python.sglang.utils import read_json

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

def main(server_args: ServerArgs, bench_args: BenchArgs, args: argparse.Namespace):

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

    throughput_test(
        server_args,
        bench_args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    add_extra_run_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)
    main(server_args, bench_args, args)

