#!/usr/bin/env bash
set -e

## LLaMA 3, EAGLE 2 ##

python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/  \
    --cuda-graph-max-bs 32 \
    --max-running-requests 32 \
    --num-prompts 300

python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/  \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path /storage/datasets/huggingface/models/sglang-EAGLE-LLaMA3-Instruct-8B-bf16 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 5 \
    --speculative-num-draft-tokens 8 \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 300

python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/  \
    --speculative-algorithm SSSD \
    --speculative-draft-model-path /storage/users/mmarzollo/datastores/ultrachat_magpie_llama3.idx \
    --speculative-eagle-topk 5 \
    --speculative-num-steps 5 \
    --speculative-num-draft-tokens 8 \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 300


## LLaMA 3.1, EAGLE 3 ##

python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 100
    
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /storage/datasets/huggingface/models/sglang-EAGLE3-Llama-3.1-Instruct-8B-bf16 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 3 \
    --speculative-num-draft-tokens 8 \
    --cuda-graph-max-bs 2 \
    --max-running-requests 2 \
    --num-prompts 10 \
    --sharegpt-output-len 256
    # --disable-cuda-graph \

python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --speculative-algorithm SSSD \
    --speculative-draft-model-path /storage/users/mmarzollo/datastores/ultrachat_magpie_llama3.1-MT-500.idx \
    --speculative-num-steps 6 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 8 \
    --cuda-graph-max-bs 2 \
    --max-running-requests 2 \
    --num-prompts 10 \
    --sharegpt-output-len 256


# HYPERPARAMETER SEARCH

python3 -m bench_speculative --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /storage/datasets/huggingface/models/sglang-EAGLE3-Llama-3.1-Instruct-8B-bf16

# IMPORTANT PARAMETERS TO ADD FOR THE E2E EVALUATIONS

python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/  \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 80 \
    --dataset-name mt-bench \
    --sharegpt-output-len 1500 \
    --disable-ignore-eos \
    --apply-chat-template